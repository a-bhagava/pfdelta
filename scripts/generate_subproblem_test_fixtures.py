"""
One-off generator for tests/fixtures/<case>.pt -- run this ON THE CLUSTER
(wherever the real PFDelta data actually lives; NOT part of the pytest
suite, not run automatically). Loads a real dataset batch, runs it through
a real model forward pass, and saves the (data, output_dict) pair so
tests/test_subproblem_construction.py / tests/test_subproblem_evals.py can
load it locally afterward -- no cluster/GPU access needed at test-run time,
only to generate (or regenerate) these files.

Why a real forward pass instead of feeding ground truth back in as
"predictions": ground truth is a physically self-consistent solution BY
CONSTRUCTION, so it can't exercise what happens with a real model's actual
errors -- an imperfect boundary-injection fold (built from a teacher
prediction that doesn't exactly satisfy KCL), a promoted local slack's
forced va=0 landing far from what was actually predicted there, etc. A
real forward pass -- even an untrained one -- has genuine prediction error
across every output head.

Model: UNTRAINED (randomly initialized) CANOS_PF by default -- valid
shapes, maximally "wrong" predictions, no checkpoint needed. Pass
--model_path for a real trained checkpoint instead, for smaller/more
structured errors closer to what a trained run would actually see.

Usage (on the cluster):
    python scripts/generate_subproblem_test_fixtures.py --root_dir /orcd/pool/005/donti_shared/pfdelta_data
    python scripts/generate_subproblem_test_fixtures.py --root_dir /orcd/pool/005/donti_shared/pfdelta_data --model_path runs/some_run/model.pt

Then copy tests/fixtures/*.pt back to wherever you run pytest (e.g. scp) --
conftest.py's fixtures just torch.load() them, no further cluster access
needed.
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch_geometric.loader import DataLoader

import core.datasets.pfdelta_variants  # noqa: F401 -- registers PFDeltaCANOS
import core.models.canos_pf  # noqa: F401 -- registers canos_pf
from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.models.canos_pf import CANOS_PF

CASES = ["case14", "case118"]
SAMPLES_PER_CASE = 8  # one real batch per case -- small, keeps the saved .pt files tiny
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")

HIDDEN_DIM = 256
K_STEPS = 15
INCLUDE_SENT_MESSAGES = True


def build_model(reference_dataset, model_path, device):
    model = CANOS_PF(
        dataset=reference_dataset,
        hidden_dim=HIDDEN_DIM,
        k_steps=K_STEPS,
        include_sent_messages=INCLUDE_SENT_MESSAGES,
    ).to(device)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from {model_path}")
    else:
        print("No --model_path given -- using an UNTRAINED (randomly initialized) model.")
    model.eval()
    return model


@torch.no_grad()
def generate_for_case(case_name: str, root_dir: str, model, device) -> tuple:
    dataset = PFDeltaCANOS(
        root_dir=root_dir, case_name=case_name, split="val", model="CANOS",
        task=1.3, add_bus_type=True,
    )
    dataloader = DataLoader(dataset, batch_size=SAMPLES_PER_CASE, shuffle=False)
    data = next(iter(dataloader)).to(device)
    outputs = model(data)
    # Move everything to CPU before saving -- these fixtures need to load
    # fine on a laptop with no GPU at all.
    data = data.to("cpu")
    outputs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in outputs.items()}
    return data, outputs


def main(root_dir: str, model_path: str):
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed seed for the untrained model's random init (irrelevant when
    # --model_path is given) -- so regenerating these fixtures later
    # reproduces the same "wrong" predictions instead of silently drawing
    # a fresh random model every time.
    torch.manual_seed(0)

    reference_dataset = PFDeltaCANOS(
        root_dir=root_dir, case_name=CASES[0], split="val", model="CANOS",
        task=1.3, add_bus_type=True,
    )
    model = build_model(reference_dataset, model_path, device)

    for case_name in CASES:
        data, outputs = generate_for_case(case_name, root_dir, model, device)
        out_path = os.path.join(FIXTURES_DIR, f"{case_name}.pt")
        torch.save({"data": data, "outputs": outputs}, out_path)
        print(f"Saved a {SAMPLES_PER_CASE}-sample batch for {case_name} -> {out_path}")

    print(
        f"\nDone. Copy {FIXTURES_DIR}/*.pt to wherever you run pytest "
        "(if this wasn't already local) -- tests/conftest.py loads them from there."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", type=str, required=True, help="Where the real PFDelta data lives.")
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Optional trained checkpoint (model.pt) -- omit for an untrained model.",
    )
    args = parser.parse_args()
    main(args.root_dir, args.model_path)
