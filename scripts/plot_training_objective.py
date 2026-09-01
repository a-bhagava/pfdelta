"""
Purpose-built plots for a subgraph_finetune_trainer run (canos_task_3_1_
finetune.yaml / canos_task_3_1_joint_train.yaml style configs):

1. Training objective: train_loss[0]'s own value (already the correctly-
   weighted total that's actually backpropagated -- CombinedLoss applies
   each component's weight before summing, so no reconstruction needed)
   alongside that SAME weighted combination evaluated on the case500
   validation set (the one functional.consistency_val_indices actually
   runs subproblem_consistency on). val.json reports universal_power_
   balance/subproblem_consistency/subgraph-PBL as separate keys rather
   than a single pre-combined one, so this reconstructs
   1.0*PBL + w2*consistency + w3*subgraph_PBL from those three, using
   the actual w2/w3 this run used (read from its own config.yaml, not
   assumed) -- letting you see whether training is fitting the actual
   backpropagated objective on the same-distribution (case500) val set.

2. Universal power balance (PBL Mean) on every validation set separately,
   including case500 -- the actual metric that matters, independent of
   which augmentation weights were used to get there.

3. Consistency loss broken down by sampling strategy (bfs/random_walk/
   forest_fire/snowball -- see core.utils.create_subproblem.
   SAMPLING_STRATEGIES), train and case500-val, if the run's config
   actually used `sampling_strategies` (see SubproblemConsistencyLoss.
   per_strategy_loss) -- reads the "Consistency [<name>]" keys the config
   wires up via RecycleLoss's dotted recycled_parameter support. Skipped
   gracefully (with a printed note, not an error) for a run that didn't
   configure sampling_strategies, so this script still works on any
   subgraph_finetune_trainer run either way.
"""
import os
import glob
import json
import argparse

import matplotlib.pyplot as plt
import yaml


def find_run_folder(run_name):
    """Searches for the folder in 'runs' using glob and ensures the name is
    unique."""
    run_pattern = os.path.join("runs", "**", run_name)
    matching_folders = glob.glob(run_pattern, recursive=True)
    if len(matching_folders) == 0:
        print(f"Error: No folder named '{run_name}' found in 'runs'.")
        return None
    elif len(matching_folders) > 1:
        print(
            f"Error: Multiple folders with the name '{run_name}' found. "
            "Please ensure the name is unique."
        )
        return None
    return matching_folders[0]


def _find_loss_entry(losses, name=None, recycled_parameter=None, keyword=None):
    """Finds one entry in a train_loss[0]/val_loss-style list of dicts by
    name and/or (for a recycle_loss entry) its recycled_parameter/keyword --
    robust to which exact fields are present, since configs vary."""
    for entry in losses:
        if name is not None and entry.get("name") != name:
            continue
        inputs = entry.get("inputs", entry)  # some entries flatten inputs into themselves (val_loss)
        if recycled_parameter is not None and inputs.get("recycled_parameter") != recycled_parameter:
            continue
        if keyword is not None and inputs.get("keyword") != keyword:
            continue
        return entry
    return None


def load_run(run_folder):
    with open(os.path.join(run_folder, "config.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(run_folder, "train.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(run_folder, "val.json")) as f:
        val_data = json.load(f)
    return config, train_data, val_data


def get_case500_val_index(config):
    indices = config["functional"].get("consistency_val_indices", [])
    assert len(indices) == 1, (
        f"expected exactly one consistency_val_indices entry, got {indices} -- "
        "this script assumes a single designated same-distribution val set."
    )
    return indices[0]


def get_val_case_names(config):
    """dataset.datasets[1:] in order == val_num 0..N-1 in val.json's
    per-epoch lists, per calc_one_val_error's own indexing."""
    return [d["case_name"] for d in config["dataset"]["datasets"][1:]]


def get_consistency_weights(config):
    """Reads w2 (subproblem_consistency's own weight) and w3 (the
    recycle_loss pulling pbl_loss's weight) from train_loss[0]'s actual
    resolved losses list -- the real values THIS run used, not assumed
    defaults. Also returns the loss_name val.json uses for the subgraph-PBL
    component (config-defined, so read rather than hardcoded)."""
    train_loss0 = config["optim"]["train_params"]["train_loss"][0]
    losses = train_loss0["losses"]

    consistency_entry = _find_loss_entry(losses, name="subproblem_consistency")
    assert consistency_entry is not None, "train_loss[0] has no subproblem_consistency component"
    w2 = consistency_entry.get("weight", 1.0)

    subgraph_pbl_entry = _find_loss_entry(losses, name="recycle_loss", recycled_parameter="pbl_loss")
    assert subgraph_pbl_entry is not None, "train_loss[0] has no recycle_loss pulling pbl_loss"
    w3 = subgraph_pbl_entry.get("weight", 1.0)

    val_loss = config["optim"]["val_params"]["val_loss"]
    val_subgraph_pbl_entry = _find_loss_entry(val_loss, name="recycle_loss", recycled_parameter="pbl_loss")
    assert val_subgraph_pbl_entry is not None, "val_loss has no recycle_loss pulling pbl_loss"
    subgraph_pbl_val_key = val_subgraph_pbl_entry["loss_name"]

    return w2, w3, subgraph_pbl_val_key


def get_sampling_strategy_names(config):
    """Reads which sampling strategies (if any) train_loss[0]'s
    subproblem_consistency entry was configured with (see
    core.utils.create_subproblem.SAMPLING_STRATEGIES /
    custom_losses.SubproblemConsistencyLoss.sampling_strategies). Returns
    [] for a run that didn't use sampling_strategies (plain BFS only), in
    which case plot_per_strategy_consistency has nothing to plot."""
    train_loss0 = config["optim"]["train_params"]["train_loss"][0]
    losses = train_loss0["losses"]
    consistency_entry = _find_loss_entry(losses, name="subproblem_consistency")
    if consistency_entry is None:
        return []
    inputs = consistency_entry.get("inputs", consistency_entry)
    strategies = inputs.get("sampling_strategies")
    if not strategies:
        return []
    return list(strategies.keys())


def _sorted_epochs(data):
    return sorted(data.keys(), key=int)


def plot_training_objective(run_name, config, train_data, val_data, out_path, log=False):
    w2, w3, subgraph_pbl_key = get_consistency_weights(config)
    case500_idx = get_case500_val_index(config)
    case500_name = get_val_case_names(config)[case500_idx]

    train_epochs = _sorted_epochs(train_data)
    # train_loss[0]'s own key -- always train_data's own first entry, and
    # already the correctly-weighted total (CombinedLoss applies each
    # component's weight before summing -- see custom_losses.CombinedLoss),
    # so no reconstruction needed on the train side.
    train_key = list(train_data[train_epochs[0]].keys())[0]
    train_vals = [train_data[e][train_key] for e in train_epochs]

    val_epochs = _sorted_epochs(val_data)
    val_totals = []
    for e in val_epochs:
        entry = val_data[e][case500_idx]
        total = (
            1.0 * entry["PBL Mean"]
            + w2 * entry["Subproblem consistency"]
            + w3 * entry[subgraph_pbl_key]
        )
        val_totals.append(total)

    plt.figure(figsize=(10, 6))
    plt.plot(
        list(map(int, train_epochs)), train_vals,
        marker="o", linestyle="-", color="b", label="Train (train_loss[0], actual weighted total)",
    )
    plt.plot(
        list(map(int, val_epochs)), val_totals,
        marker="o", linestyle="-", color="darkorange",
        label=f"Val: {case500_name} (reconstructed, w2={w2:g}, w3={w3:g})",
    )
    plt.xlabel("Training point")
    plt.ylabel("1.0*PBL + w2*consistency + w3*subgraph-PBL")
    plt.title(f"{run_name} - training objective (train vs. same-distribution val)")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    return train_key, w2, w3, subgraph_pbl_key


def plot_universal_pbl(run_name, config, val_data, out_path, log=False):
    case_names = get_val_case_names(config)
    case500_idx = get_case500_val_index(config)
    val_epochs = _sorted_epochs(val_data)

    plt.figure(figsize=(10, 6))
    for i, case_name in enumerate(case_names):
        pbl_vals = [val_data[e][i]["PBL Mean"] for e in val_epochs]
        label = f"{case_name} (val {i})"
        if i == case500_idx:
            label += " -- same distribution as train"
        plt.plot(
            list(map(int, val_epochs)), pbl_vals,
            marker="o", linestyle="-",
            linewidth=2.5 if i == case500_idx else 1.5,
            label=label,
        )
    plt.xlabel("Training point")
    plt.ylabel("PBL Mean (universal power balance)")
    plt.title(f"{run_name} - universal PBL across validation sets")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_per_strategy_consistency(run_name, config, train_data, val_data, out_path, log=False):
    """Consistency loss (the same weighted 1.0*voltage + 1.0*injection +
    1.0*edge + pbl_weight*pbl combination as train_loss[0]'s own
    "Consistency subtotal", just broken out per sampling strategy -- see
    SubproblemConsistencyLoss.per_strategy_loss), one line per configured
    strategy, train (solid) and case500-val (dashed). Returns the list of
    strategy names actually plotted, or None if this run didn't configure
    sampling_strategies / doesn't have the corresponding train.json keys
    (an older run, or a config that didn't wire the per-strategy
    recycle_loss entries) -- in either case this is a no-op, not an error,
    so callers can just skip it (and any downstream file it would have
    produced) when it returns None.
    """
    strategy_names = get_sampling_strategy_names(config)
    if not strategy_names:
        print("No sampling_strategies configured for this run -- skipping per-strategy consistency plot.")
        return None

    train_epochs = _sorted_epochs(train_data)
    val_epochs = _sorted_epochs(val_data)
    case500_idx = get_case500_val_index(config)

    first_epoch_keys = train_data[train_epochs[0]].keys()
    available = [name for name in strategy_names if f"Consistency [{name}]" in first_epoch_keys]
    missing = [name for name in strategy_names if name not in available]
    if missing:
        print(
            f"Note: no 'Consistency [<name>]' key in train.json for: {missing} "
            "(older run, or config didn't wire per-strategy recycle_loss entries for it)"
        )
    if not available:
        print("None of the configured strategies have a matching train.json key -- skipping per-strategy consistency plot.")
        return None

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, name in enumerate(available):
        key = f"Consistency [{name}]"
        color = colors[i % len(colors)]
        train_vals = [train_data[e].get(key) for e in train_epochs]
        plt.plot(
            list(map(int, train_epochs)), train_vals,
            marker="o", linestyle="-", color=color, label=f"{name} (train)",
        )
        if all(key in val_data[e][case500_idx] for e in val_epochs):
            val_vals = [val_data[e][case500_idx].get(key) for e in val_epochs]
            plt.plot(
                list(map(int, val_epochs)), val_vals,
                marker="s", linestyle="--", color=color, alpha=0.7,
                label=f"{name} (val, case500)",
            )
    plt.xlabel("Training point")
    plt.ylabel("Consistency loss (weighted, per sampling strategy)")
    plt.title(f"{run_name} - consistency loss by sampling strategy")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    return available


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot (1) the actual weighted training objective vs. the same "
            "combination reconstructed on the case500 (same-distribution) "
            "val set, (2) universal PBL on every validation set "
            "separately, and (3) consistency loss broken down by sampling "
            "strategy (if the run configured sampling_strategies), for a "
            "subgraph_finetune_trainer run."
        )
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--log", action="store_true", default=False,
        help="Log-scale the y-axis on every plot produced.",
    )
    args = parser.parse_args()
    print(f"Searching for folder with name {args.run_name}")
    run_path = find_run_folder(args.run_name)
    assert run_path is not None, "Run not found!"
    print(f"Run name found in path: {run_path}")

    config, train_data, val_data = load_run(run_path)

    objective_path = f"{run_path}/plot_training_objective.png"
    pbl_path = f"{run_path}/plot_universal_pbl.png"
    per_strategy_path = f"{run_path}/plot_per_strategy_consistency.png"

    train_key, w2, w3, subgraph_pbl_key = plot_training_objective(
        args.run_name, config, train_data, val_data, objective_path, log=args.log
    )
    plot_universal_pbl(args.run_name, config, val_data, pbl_path, log=args.log)
    plotted_strategies = plot_per_strategy_consistency(
        args.run_name, config, train_data, val_data, per_strategy_path, log=args.log
    )

    print()
    print(f"Training objective key (train.json): {train_key!r}")
    print(f"Reconstruction weights: w2 (consistency) = {w2}, w3 (subgraph-PBL) = {w3}")
    print(f"Subgraph-PBL val.json key: {subgraph_pbl_key!r}")
    if plotted_strategies:
        print(f"Sampling strategies plotted: {plotted_strategies}")
