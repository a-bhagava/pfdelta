"""
Consistency transfer test: evaluate an already-trained, FROZEN model's
self-consistency (SubproblemConsistencyLoss) across a grid of (source case,
subgraph size band) combinations -- no training, no gradient steps.

Answers: is self-consistency a property of the specific (case, size) a
model was trained/tuned on, or something more fundamental about the model
itself? E.g. does a model that's highly self-consistent on case500 cut to
3-7-bus subgraphs stay just as consistent on case118 cut to 24-36-bus
subgraphs?

Usage:
    python scripts/evaluate_consistency_transfer.py --config core/configs/consistency_transfer_eval.yaml
"""
import argparse
import copy
import json
import os
import sys

# Matches the convention every other script/ entry point in this repo uses
# (see scripts/find_best_run.py, scripts/task31_error.py, etc.) -- lets
# "core.*" resolve when this is invoked as `python scripts/foo.py` from the
# repo root, since that puts scripts/ (not the repo root) on sys.path by
# default.
sys.path.append(os.getcwd())

import torch
import yaml
from torch_geometric.loader.dataloader import DataLoader

# Trigger registry population (models/datasets register themselves on
# import) -- mirrors how main.py/trainers pull these in before using the
# registry to look them up by name.
import core.datasets.pfdelta_variants  # noqa: F401
import core.models.canos_pf  # noqa: F401
from core.utils.custom_losses import SubproblemConsistencyLoss
from core.utils.registry import registry


def load_source_config(source_run: str) -> dict:
    with open(os.path.join(source_run, "config.yaml")) as f:
        return yaml.safe_load(f)


def build_dataset(dataset_name, root_dir, case_name, split, model_dataset_name, task, add_bus_type):
    dataset_class = registry.get_dataset_class(dataset_name)
    assert dataset_class is not None, f"Dataset {dataset_name!r} not found in registry!"
    return dataset_class(
        root_dir=root_dir,
        case_name=case_name,
        split=split,
        model=model_dataset_name,
        task=task,
        add_bus_type=add_bus_type,
    )


def resolve_architecture(cfg: dict) -> dict:
    """Model architecture kwargs (name/hidden_dim/k_steps/
    include_sent_messages) -- taken directly from the eval config wherever
    given there (a `model:` block, same shape as a training config's own),
    falling back to source_run's own config.yaml ONLY for whichever keys
    are missing. Explicit values here always win, and if all four are
    given directly, source_run's config.yaml is never read at all -- so
    you don't need a source_run/config.yaml to exist if you'd rather just
    specify k_steps/hidden_dim/etc. yourself (e.g. pointing straight at a
    bare model_path)."""
    explicit = dict(cfg.get("model", {}))
    required = {"name", "hidden_dim", "k_steps", "include_sent_messages"}
    missing = required - explicit.keys()
    if missing:
        assert cfg.get("source_run"), (
            f"model: is missing {sorted(missing)} and no source_run was given to read "
            "them from -- either add a source_run (a directory with its own "
            "config.yaml), or specify all of name/hidden_dim/k_steps/"
            "include_sent_messages directly under model: in this eval config."
        )
        source_model_cfg = load_source_config(cfg["source_run"])["model"]
        for key in missing:
            assert key in source_model_cfg, (
                f"{key!r} not found in {cfg['source_run']}/config.yaml's own model: "
                f"block either -- specify it explicitly under this eval config's model: instead."
            )
            explicit[key] = source_model_cfg[key]
    return explicit


def resolve_model_path(cfg: dict) -> str:
    if cfg.get("model_path"):
        return cfg["model_path"]
    assert cfg.get("source_run"), "Need either model_path or source_run to know where to load weights from."
    return os.path.join(cfg["source_run"], "model.pt")


def resolve_output_dir(cfg: dict) -> str:
    if cfg.get("output_dir"):
        return cfg["output_dir"]
    assert cfg.get("source_run"), "Need either output_dir or source_run to know where to save results."
    return cfg["source_run"]


def build_model(architecture: dict, reference_dataset, device):
    """`architecture` is what resolve_architecture returned -- a plain
    dict of model constructor kwargs (name/hidden_dim/k_steps/
    include_sent_messages). `reference_dataset` is only used to infer
    feature dimensions at construction time (see this script's own module
    docstring for why any case's dataset works equally well for this)."""
    model_cfg = copy.deepcopy(architecture)
    model_name = model_cfg.pop("name")
    model_cfg["dataset"] = reference_dataset
    model_class = registry.get_model_class(model_name)
    assert model_class is not None, f"Model {model_name!r} not found in registry!"
    return model_class(**model_cfg).to(device)


def load_weights(model, model_path: str):
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)


def get_num_buses(dataset) -> int:
    """Bus count for this dataset's case -- constant across every sample
    in it (contingencies, if any, remove branches, never buses), so
    checking just dataset[0] is exact, not an approximation."""
    sample = dataset[0]
    if "num_nodes" in sample["bus"]:
        return int(sample["bus"].num_nodes)
    return int(sample["bus"].x.shape[0])


def degeneracy_status(min_size: int, max_size: int, num_buses: int) -> str:
    """A subgraph size band interacts with a case's own bus count in one
    of three ways -- see build_subproblem_batch/sample_bus_subset*'s own
    "stops early, returns fewer buses than requested" contract, which
    applies with NO clamping against the case's actual size:

    "degenerate": min_size alone already reaches/exceeds num_buses, so
        EVERY draw in this band is capped to the whole case -- no tie
        lines ever get cut, the second forward pass sees a copy of the
        first pass's own input, and voltage/injection/edge losses (what
        the reported .loss is made of) trivially collapse toward zero.
        Not a real signal -- skip these combinations entirely.
    "partial": max_size exceeds num_buses but min_size doesn't -- SOME
        draws are genuine cuts, others degenerate the same way as above,
        diluting (usually flattering) the average. Worth a warning, but
        there's still real signal mixed in, so this isn't skipped.
    "ok": the whole band is reachable -- no distortion from this effect.
    """
    if min_size >= num_buses:
        return "degenerate"
    if max_size >= num_buses:
        return "partial"
    return "ok"


@torch.no_grad()
def evaluate_one_combo(
    model,
    dataset,
    min_size,
    max_size,
    batch_size,
    subgraphs_per_sample,
    sampling_strategies,
    seed,
    detach_teacher,
    device,
):
    """Runs SubproblemConsistencyLoss over every batch of `dataset`, with
    subgraphs cut in [min_size, max_size], and returns the batch-averaged
    loss + each of its components. No gradients anywhere here -- this is
    pure evaluation of an already-trained model, mirroring exactly what
    calc_one_val_error does during training, just with full control over
    which case/size combination to run, outside the full trainer
    lifecycle (optimizer/scheduler/checkpointing don't apply here)."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    consistency = SubproblemConsistencyLoss(
        min_size=min_size,
        max_size=max_size,
        subgraphs_per_sample=subgraphs_per_sample,
        sampling_strategies=sampling_strategies,
        seed=seed,
        detach_teacher=detach_teacher,
    )
    consistency.model = model
    model.eval()

    totals = {"loss": 0.0, "voltage_loss": 0.0, "injection_loss": 0.0, "edge_loss": 0.0, "pbl_loss": 0.0}
    n_batches = 0
    for data in dataloader:
        data = data.to(device)
        outputs = model(data)
        loss = consistency(outputs, data)
        totals["loss"] += loss.item()
        totals["voltage_loss"] += consistency.voltage_loss.item()
        totals["injection_loss"] += consistency.injection_loss.item()
        totals["edge_loss"] += consistency.edge_loss.item()
        totals["pbl_loss"] += consistency.pbl_loss.item()
        n_batches += 1
    assert n_batches > 0, (
        f"empty dataloader for min_size={min_size}, max_size={max_size} -- "
        "check the dataset actually has data for this case/split."
    )
    return {k: v / n_batches for k, v in totals.items()}


def print_matrix(results: dict, case_names: list, band_names: list, metric: str = "loss"):
    header = "case".ljust(12) + "".join(name.rjust(14) for name in band_names)
    print(header)
    for case_name in case_names:
        row = case_name.ljust(12)
        for band_name in band_names:
            cell = results[(case_name, band_name)]
            text = f"{cell[metric]:.6f}" if cell is not None else "skipped"
            row += text.rjust(14)
        print(row)


def plot_heatmap(results: dict, case_names: list, band_names: list, out_path: str, metric: str = "loss"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Skipped (degenerate) combos become NaN -- imshow renders them via
    # cmap.set_bad rather than mapping them onto the real color scale.
    matrix = np.array(
        [
            [
                results[(c, b)][metric] if results[(c, b)] is not None else float("nan")
                for b in band_names
            ]
            for c in case_names
        ]
    )
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * len(band_names), 1.5 + 0.8 * len(case_names)))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names)
    ax.set_xlabel("Subgraph size band")
    ax.set_ylabel("Source case")
    ax.set_title(f"Consistency transfer: {metric}")
    for i in range(len(case_names)):
        for j in range(len(band_names)):
            value = matrix[i, j]
            text = "skipped" if value != value else f"{value:.3f}"  # value != value <=> NaN
            color = "black" if value != value else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def main(config_path: str) -> dict:
    """Runs the full (case x size band) sweep for the config at
    `config_path` and returns `results` (`{(case_name, band_name): metrics}`)
    -- factored out of the CLI entry point below so it can also be called
    directly (e.g. from a test) without going through argument parsing or a
    subprocess, sharing the caller's own process/registry state."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    architecture = resolve_architecture(cfg)
    model_path = resolve_model_path(cfg)
    output_dir = resolve_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    case_names = cfg["cases"]
    band_names = [b["name"] for b in cfg["size_bands"]]

    print(f"Loading reference dataset ({case_names[0]}, split={cfg['split']}) for model construction...")
    reference_dataset = build_dataset(
        cfg["dataset_name"], cfg["root_dir"], case_names[0], cfg["split"],
        cfg["model_dataset_name"], cfg["task"], cfg["add_bus_type"],
    )
    print(f"Architecture: {architecture}")
    model = build_model(architecture, reference_dataset, device)
    load_weights(model, model_path)
    print(f"Loaded weights from {model_path}")

    dataset_cache = {case_names[0]: reference_dataset}

    results = {}
    for case_name in case_names:
        if case_name not in dataset_cache:
            print(f"Loading dataset for {case_name}...")
            dataset_cache[case_name] = build_dataset(
                cfg["dataset_name"], cfg["root_dir"], case_name, cfg["split"],
                cfg["model_dataset_name"], cfg["task"], cfg["add_bus_type"],
            )
        dataset = dataset_cache[case_name]
        num_buses = get_num_buses(dataset)
        for band in cfg["size_bands"]:
            status = degeneracy_status(band["min_size"], band["max_size"], num_buses)
            if status == "degenerate":
                print(
                    f"Skipping case={case_name} ({num_buses} buses), band={band['name']} "
                    f"(min={band['min_size']}, max={band['max_size']}) -- min_size alone "
                    f"already reaches the case's own bus count, so every draw would just "
                    f"return the whole case (no real cut, no GPU time wasted on it)."
                )
                results[(case_name, band["name"])] = None
                continue
            if status == "partial":
                print(
                    f"Warning: case={case_name} ({num_buses} buses), band={band['name']} "
                    f"(min={band['min_size']}, max={band['max_size']}) -- max_size exceeds "
                    f"the case's own bus count, so some draws will trivially return the "
                    f"whole case, diluting (usually flattering) this cell's average."
                )
            print(
                f"Evaluating case={case_name}, band={band['name']} "
                f"(min={band['min_size']}, max={band['max_size']})..."
            )
            metrics = evaluate_one_combo(
                model,
                dataset,
                band["min_size"],
                band["max_size"],
                cfg.get("batch_size", 64),
                cfg.get("subgraphs_per_sample", 4),
                cfg.get("sampling_strategies"),
                cfg.get("seed"),
                cfg.get("detach_teacher", True),
                device,
            )
            results[(case_name, band["name"])] = metrics
            print(f"  -> consistency loss = {metrics['loss']:.6f}")

    print("\nConsistency loss matrix (rows=case, columns=size band):")
    print_matrix(results, case_names, band_names, metric="loss")

    out_json = os.path.join(output_dir, "consistency_transfer_eval.json")
    json.dump(
        {f"{c}|{b}": v for (c, b), v in results.items()},
        open(out_json, "w"),
        indent=2,
    )
    print(f"\nSaved full results (all components, not just .loss) to {out_json}")

    out_png = os.path.join(output_dir, "consistency_transfer_eval.png")
    plot_heatmap(results, case_names, band_names, out_png, metric="loss")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
