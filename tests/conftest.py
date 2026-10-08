"""
Shared fixtures for test_subproblem_construction.py / test_subproblem_evals.py.

Fixtures below load REAL (data, output_dict) pairs -- an actual batched
dataset sample run through a real model forward pass, NOT ground truth --
from tests/fixtures/<case>.pt. Those files are generated once on the
cluster (where the real PFDelta data actually lives) by scripts/generate_
subproblem_test_fixtures.py and copied back here; nothing in this file
needs cluster/GPU access itself, just the already-saved files.

Why real model output, not ground truth fed back in as "predictions":
ground truth is a physically self-consistent solution by construction, so
it can't exercise what happens with a model's actual prediction error --
an imperfect boundary-injection fold, a promoted local slack's forced
va=0 landing far from what was actually predicted there, etc.

Regenerate with:
    python scripts/generate_subproblem_test_fixtures.py --root_dir <wherever the real data lives>
then copy tests/fixtures/*.pt back here (e.g. scp from the cluster).
"""
import os

import pytest
import torch

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(case_name: str) -> tuple:
    path = os.path.join(FIXTURES_DIR, f"{case_name}.pt")
    assert os.path.exists(path), (
        f"{path} doesn't exist -- generate it first (see this file's own "
        "module docstring): python scripts/generate_subproblem_test_fixtures.py "
        "--root_dir <wherever the real data lives>, then copy tests/fixtures/*.pt here."
    )
    # weights_only=False: this saves a HeteroData/dict-of-tensors pair, not
    # a plain state_dict -- torch.load's weights_only=True default (recent
    # torch versions) would refuse to unpickle those types.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return saved["data"], saved["outputs"]


@pytest.fixture(scope="session")
def case14_real_sample() -> tuple:
    """(data, output_dict) -- a real batched case14 sample run through an
    actual model forward pass (see scripts/generate_subproblem_test_
    fixtures.py's own docstring). Not ground truth -- has real prediction
    error, on every output head."""
    return _load_fixture("case14")


@pytest.fixture(scope="session")
def case118_real_sample() -> tuple:
    """Same as case14_real_sample, for case118."""
    return _load_fixture("case118")
