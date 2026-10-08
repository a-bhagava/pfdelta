# Test fixtures

`case14.pt` / `case118.pt` (not committed -- generate locally, see below)
each hold one real, batched dataset sample run through a real model
forward pass: `{"data": <HeteroData batch>, "outputs": <output_dict>}`.

Used by `tests/conftest.py`'s `case14_real_sample` / `case118_real_sample`
fixtures, consumed by `tests/test_subproblem_construction.py` and
`tests/test_subproblem_evals.py`.

## Generating / regenerating

Run on the cluster (wherever the real PFDelta data actually lives):

```
python scripts/generate_subproblem_test_fixtures.py --root_dir /orcd/pool/005/donti_shared/pfdelta_data
```

Add `--model_path runs/some_run/model.pt` to use a real trained checkpoint
instead of an untrained model (see that script's own docstring for why
untrained is the default: valid shapes, maximally "wrong" predictions, no
checkpoint needed).

Then copy the two `.pt` files it writes here (e.g. `scp` from the cluster)
to this same directory. No cluster/GPU access is needed after that --
`pytest` loads them locally from then on.
