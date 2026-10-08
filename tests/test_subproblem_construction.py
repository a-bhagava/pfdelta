"""
Tests for whether the subgraph "problem transformation" itself
(core/utils/create_subproblem.py) produces VALID subgraphs -- connectivity,
promoted-local-slack election, voltage/edge masking, boundary-injection
folding -- independent of whether the resulting loss numbers are correct
(see test_subproblem_evals.py for that side).

Fixtures (case14_real_sample / case118_real_sample, from tests/conftest.py)
are a REAL (data, output_dict) pair -- an actual model forward pass, not
ground truth -- so these tests exercise what happens with genuine
prediction error, not just a perfectly self-consistent solution. See
conftest.py's own docstring for why, and scripts/generate_subproblem_test_
fixtures.py to (re)generate them.

build_subproblem_batch is a pure function of (data, output_dict) -- no
model, no self.model -- so every test below calls it directly rather than
going through SubproblemConsistencyLoss.__call__.
"""
import pytest
import torch

from core.utils.create_subproblem import build_subproblem_batch

MIN_SIZE = 5
MAX_SIZE = 15
SUBGRAPHS_PER_SAMPLE = 4


def _connected_components(bus_batch: torch.Tensor, edge_index: torch.Tensor) -> list:
    """Union-find over `edge_index`, scoped per subgraph id (bus_batch's
    own value at each node) -- returns, for each subgraph id present, the
    number of connected components among ITS OWN nodes under ITS OWN
    (interior) edges. A fully-connected subgraph has exactly 1 per id
    (trivially true for a 0- or 1-node subgraph, nothing to disconnect)."""
    num_subgraphs = int(bus_batch.max().item()) + 1 if bus_batch.numel() else 0
    counts = []
    for subgraph_id in range(num_subgraphs):
        node_mask = bus_batch == subgraph_id
        node_positions = node_mask.nonzero(as_tuple=True)[0].tolist()
        if len(node_positions) <= 1:
            counts.append(1)
            continue
        local_id = {global_pos: i for i, global_pos in enumerate(node_positions)}
        edge_in_subgraph = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        src = edge_index[0][edge_in_subgraph].tolist()
        dst = edge_index[1][edge_in_subgraph].tolist()

        parent = list(range(len(node_positions)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, d in zip(src, dst):
            ra, rb = find(local_id[s]), find(local_id[d])
            if ra != rb:
                parent[ra] = rb

        roots = {find(i) for i in range(len(node_positions))}
        counts.append(len(roots))
    return counts


@pytest.mark.parametrize("sample_fixture", ["case14_real_sample", "case118_real_sample"])
class TestBuildSubproblemBatch:
    def test_cut_subgraph_is_connected(self, sample_fixture, request):
        data, outputs = request.getfixturevalue(sample_fixture)
        sub_data, bus_map, voltage_mask, edge_map, va_offset, strategy_per_subgraph = (
            build_subproblem_batch(
                data, outputs,
                min_size=MIN_SIZE, max_size=MAX_SIZE,
                subgraphs_per_sample=SUBGRAPHS_PER_SAMPLE,
                generator=torch.Generator().manual_seed(0),
            )
        )
        bus_batch = sub_data["bus"].batch
        edge_index = sub_data["bus", "branch", "bus"].edge_index
        components = _connected_components(bus_batch, edge_index)
        disconnected = [i for i, c in enumerate(components) if c != 1]
        assert not disconnected, (
            f"subgraph(s) {disconnected} are disconnected "
            f"(component counts: {[components[i] for i in disconnected]})"
        )

    @pytest.mark.skip(reason="TODO: not yet implemented")
    def test_promoted_local_slack_elected_when_original_slack_cut(self, sample_fixture, request):
        """When the sampled bus subset excludes the case's real slack bus,
        exactly one bus in the subgraph should end up with bus_type==3,
        its va forced to 0 (not the model's own predicted value), and
        va_offset recorded as the model's predicted va there -- see
        create_subproblem.py's own "Elect a new local slack" block. Needs
        min_size/max_size small enough (relative to the fixture's own bus
        count) to force at least one such draw -- may need a few seeded
        attempts, or a dedicated tiny min_size/max_size, to hit this case
        reliably rather than relying on luck."""

    @pytest.mark.skip(reason="TODO: not yet implemented")
    def test_no_promotion_when_original_slack_already_kept(self, sample_fixture, request):
        """When the sampled subset DOES include the case's real slack bus,
        bus_type/voltage_mask/va_offset should be unchanged from the
        teacher's own values -- no promotion logic should fire. A large
        min_size (close to the case's own bus count) makes keeping the
        real slack bus near-certain."""

    @pytest.mark.skip(reason="TODO: not yet implemented")
    def test_voltage_mask_excludes_exactly_the_promoted_bus(self, sample_fixture, request):
        """voltage_mask should be all-True except for the one promoted
        local-slack row per subgraph that needed one (if any) -- confirmed
        earlier (see this file's own module docstring / the conversation
        this suite came from) that NO other mask (edge coverage in
        particular) shares this exclusion."""

    @pytest.mark.skip(reason="TODO: not yet implemented")
    def test_boundary_injection_folds_predicted_tie_line_flows(self, sample_fixture, request):
        """A boundary bus's net injection in the subgraph should equal its
        full-grid-pass injection PLUS the (stop-gradient, if
        detach_teacher) tie-line flows the model predicted on every branch
        that got cut at that bus -- see create_subproblem.py's own
        extra_p/extra_q scatter-add derivation. This is exactly the kind
        of check that NEEDS a real (not ground-truth) prediction to be
        meaningful -- ground truth's tie-line flows are already physically
        exact, so this fixture's real (imperfect) prediction is what
        actually exercises the fold arithmetic."""

    @pytest.mark.skip(reason="TODO: not yet implemented")
    def test_only_interior_branches_appear_in_sub_data_edges(self, sample_fixture, request):
        """Every edge in sub_data["bus","branch","bus"] should have BOTH
        endpoints inside the sampled bus subset -- no cut (boundary)
        branch should appear, and edge_map should correctly index back
        into the ORIGINAL data's own edge_preds for exactly those kept
        edges."""


@pytest.mark.skip(reason="TODO: needs a real dataset + a built pool file -- not yet written")
class TestPooledSampling:
    def test_pool_path_matches_live_sampling_distribution(self):
        """A pool built by scripts/build_subgraph_pool.py for a given
        case/size band should produce subgraphs statistically similar
        (e.g. size distribution, connectivity) to live sampling with the
        same min_size/max_size/strategy -- not identical draws, but the
        same population."""

    def test_pool_case_mismatch_falls_back_to_live_sampling(self):
        """Loading a pool built for one num_nodes_per_case and then calling
        build_subproblem_batch against a DIFFERENT case's data (e.g. a val
        loop that reuses one subproblem_consistency instance across every
        val dataset) should silently fall back to live sampling for that
        call, not crash or misapply the mismatched pool -- see create_
        subproblem._load_subgraph_pool's own "applies" check."""
