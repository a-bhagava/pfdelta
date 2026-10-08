"""
Tests for whether SubproblemConsistencyLoss (core/utils/custom_losses.py)
computes CORRECT loss numbers from a given subgraph -- as opposed to
whether the subgraph itself was constructed validly in the first place
(see test_subproblem_construction.py for that side).

Two kinds of coverage, clearly separated below:

1. Unit tests (implemented, pass with no real data -- pure tensor/dict
   logic on synthetic inputs). Ported from ad hoc verification done while
   building angle_weight/magnitude_weight and collect_distributions (see
   git history on custom_losses.py around that work):
     - _per_subgraph_mse / _per_subgraph_values: the refactor that split
       the raw per-subgraph pooling out of the overall-mean/per-strategy
       reduction is behavior-preserving.
     - SubproblemConsistencyLoss.collect_distributions: off by default,
       accumulates (not overwrites) across calls, empty-batch-safe, values
       are plain detached floats, independent per instance.
     - angle_weight/magnitude_weight: the backward-compat math (neither
       overridden == exactly the old combined voltage_weight behavior,
       split 50/50; overriding one leaves the other at voltage_weight/2).

2. Integration tests (TODO, skeletoned below with pytest.mark.skip) -- go
   through SubproblemConsistencyLoss.__call__ itself, needing self.model.
   Use the case14_real_sample/case118_real_sample fixtures (tests/
   conftest.py) for `data`/`outputs`, and a stand-in for self.model (a
   trivial mock, or a real-but-UNTRAINED CANOS_PF -- these tests check
   whether the loss ARITHMETIC is correct, not whether predictions are
   GOOD, so an untrained model is fine; see conftest.py's own docstring).
"""
import pytest
import torch

from core.utils.custom_losses import (
    SubproblemConsistencyLoss,
    _per_subgraph_mse,
    _per_subgraph_values,
)

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Unit tests: _per_subgraph_mse / _per_subgraph_values
# ---------------------------------------------------------------------------


class TestPerSubgraphMSE:
    def test_values_reconstructs_overall_mean(self):
        """per_subgraph_values()'s (per_subgraph, present) pair, reduced by
        hand (per_subgraph[present].mean()), must equal _per_subgraph_mse's
        own `overall` exactly -- they're meant to be the same computation,
        just with the raw distribution exposed instead of thrown away."""
        torch.manual_seed(0)
        n = 37
        batch_idx = torch.randint(0, 6, (n,))
        student = torch.randn(n, 2)
        teacher = torch.randn(n, 2)

        overall = _per_subgraph_mse(student, teacher, batch_idx)
        per_subgraph, present = _per_subgraph_values(student, teacher, batch_idx)
        manual_overall = per_subgraph[present].mean()

        assert torch.allclose(manual_overall, overall, atol=ATOL)

    def test_with_strategy_breakdown_matches_without(self):
        """Passing strategy_per_subgraph must not change `overall` -- it
        only adds the per-strategy breakdown as a second return value."""
        torch.manual_seed(0)
        n = 37
        batch_idx = torch.randint(0, 6, (n,))
        student = torch.randn(n, 2)
        teacher = torch.randn(n, 2)
        strategies = ["bfs", "random_walk", "forest_fire", "snowball"]
        strategy_per_subgraph = [strategies[i % 4] for i in range(6)]

        overall = _per_subgraph_mse(student, teacher, batch_idx)
        overall_with_strategy, per_strategy = _per_subgraph_mse(
            student, teacher, batch_idx, strategy_per_subgraph
        )

        assert torch.allclose(overall, overall_with_strategy, atol=ATOL)
        assert set(per_strategy.keys()) <= set(strategies)

    def test_empty_batch_returns_zero(self):
        empty_student = torch.zeros(0, 2)
        empty_teacher = torch.zeros(0, 2)
        empty_batch = torch.zeros(0, dtype=torch.long)
        result = _per_subgraph_mse(empty_student, empty_teacher, empty_batch)
        assert result.item() == 0.0

    def test_empty_batch_with_strategy_returns_zero_and_empty_dict(self):
        empty_student = torch.zeros(0, 2)
        empty_teacher = torch.zeros(0, 2)
        empty_batch = torch.zeros(0, dtype=torch.long)
        overall, per_strategy = _per_subgraph_mse(
            empty_student, empty_teacher, empty_batch, strategy_per_subgraph=[]
        )
        assert overall.item() == 0.0
        assert per_strategy == {}


# ---------------------------------------------------------------------------
# Unit tests: SubproblemConsistencyLoss.collect_distributions
# ---------------------------------------------------------------------------


class TestCollectDistributions:
    def test_off_by_default_and_always_present(self):
        loss = SubproblemConsistencyLoss()
        assert loss.collect_distributions is False
        assert loss.distributions == {
            "angle": [], "magnitude": [], "injection": [], "edge": [], "pbl": [],
        }

    def test_accumulates_across_calls_not_overwrites(self):
        """3 simulated batches of 3 subgraphs each -> 9 accumulated values,
        matching how __call__ would extend this once per val/eval batch."""
        torch.manual_seed(1)
        loss = SubproblemConsistencyLoss()
        loss.collect_distributions = True
        for _ in range(3):
            student = torch.randn(6)
            teacher = torch.randn(6)
            batch_idx = torch.tensor([0, 0, 1, 1, 2, 2])
            loss._collect_distribution("magnitude", student, teacher, batch_idx)
        assert len(loss.distributions["magnitude"]) == 9
        # Other metrics are independent lists, untouched.
        assert loss.distributions["injection"] == []
        assert loss.distributions["edge"] == []
        assert loss.distributions["pbl"] == []

    def test_empty_batch_is_safe_noop(self):
        loss = SubproblemConsistencyLoss()
        loss.collect_distributions = True
        loss._collect_distribution(
            "edge", torch.zeros(0), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        )
        assert loss.distributions["edge"] == []

    def test_accumulated_values_are_plain_detached_floats(self):
        torch.manual_seed(1)
        loss = SubproblemConsistencyLoss()
        loss.collect_distributions = True
        loss._collect_distribution(
            "magnitude", torch.randn(6), torch.randn(6), torch.tensor([0, 0, 1, 1, 2, 2])
        )
        assert all(isinstance(v, float) for v in loss.distributions["magnitude"])

    def test_fresh_instance_is_independent(self):
        """Two separate SubproblemConsistencyLoss instances must not share
        distributions state -- this is what lets evaluate_subgraph_loss_
        distributions.py build a fresh instance per dataset instead of
        manually resetting one shared instance between them."""
        torch.manual_seed(1)
        loss1 = SubproblemConsistencyLoss()
        loss1.collect_distributions = True
        loss1._collect_distribution(
            "magnitude", torch.randn(6), torch.randn(6), torch.tensor([0, 0, 1, 1, 2, 2])
        )
        loss2 = SubproblemConsistencyLoss()
        assert loss2.distributions["magnitude"] == []


# ---------------------------------------------------------------------------
# Unit tests: angle_weight / magnitude_weight backward-compat math
# ---------------------------------------------------------------------------


class TestAngleMagnitudeWeights:
    def test_combined_equals_half_angle_plus_half_magnitude(self):
        """voltage_loss (the OLD combined [va, vm] MSE, kept for backward
        compat -- e.g. scripts/evaluate_consistency_transfer.py reads
        .voltage_loss directly) must equal 0.5*angle_loss + 0.5*magnitude_
        loss exactly -- this identity is what lets angle_weight/magnitude_
        weight default (None) to voltage_weight/2 each and reproduce the
        old voltage_weight*voltage_loss total bit-for-bit."""
        torch.manual_seed(0)
        n = 20
        batch_idx = torch.randint(0, 4, (n,))
        student_bus = torch.randn(n, 2)
        teacher_bus = torch.randn(n, 2)

        combined = _per_subgraph_mse(student_bus, teacher_bus, batch_idx)
        angle = _per_subgraph_mse(student_bus[:, 0], teacher_bus[:, 0], batch_idx)
        magnitude = _per_subgraph_mse(student_bus[:, 1], teacher_bus[:, 1], batch_idx)

        assert torch.allclose(combined, 0.5 * (angle + magnitude), atol=ATOL)

    def test_default_weights_reproduce_old_voltage_weight_total(self):
        torch.manual_seed(0)
        n = 20
        batch_idx = torch.randint(0, 4, (n,))
        student_bus = torch.randn(n, 2)
        teacher_bus = torch.randn(n, 2)

        combined = _per_subgraph_mse(student_bus, teacher_bus, batch_idx)
        angle = _per_subgraph_mse(student_bus[:, 0], teacher_bus[:, 0], batch_idx)
        magnitude = _per_subgraph_mse(student_bus[:, 1], teacher_bus[:, 1], batch_idx)

        voltage_weight = 3.0
        old_total = voltage_weight * combined
        # angle_weight/magnitude_weight both None (not overridden) -> each
        # resolves to voltage_weight/2, per SubproblemConsistencyLoss.loss's
        # own resolution logic.
        new_total = (0.5 * voltage_weight) * angle + (0.5 * voltage_weight) * magnitude

        assert torch.allclose(old_total, new_total, atol=ATOL)

    def test_overriding_one_leaves_the_other_at_half_voltage_weight(self):
        torch.manual_seed(0)
        n = 20
        batch_idx = torch.randint(0, 4, (n,))
        student_bus = torch.randn(n, 2)
        teacher_bus = torch.randn(n, 2)

        angle = _per_subgraph_mse(student_bus[:, 0], teacher_bus[:, 0], batch_idx)
        magnitude = _per_subgraph_mse(student_bus[:, 1], teacher_bus[:, 1], batch_idx)

        voltage_weight = 3.0
        angle_weight_override = 10.0
        magnitude_weight_resolved = 0.5 * voltage_weight  # untouched, stays at voltage_weight/2

        total = angle_weight_override * angle + magnitude_weight_resolved * magnitude
        expected = angle_weight_override * angle + 0.5 * voltage_weight * magnitude
        assert torch.allclose(total, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Integration tests (TODO) -- SubproblemConsistencyLoss.__call__ itself,
# using the real case14_real_sample/case118_real_sample fixtures (tests/
# conftest.py) for data/outputs and a stand-in self.model.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="TODO: not yet implemented")
class TestSubproblemConsistencyLossIntegration:
    def test_consistency_near_zero_when_student_equals_teacher(self):
        """Set self.model to a trivial stand-in that reads the FULL-GRID
        pass's own predictions off sub_data (mapped back via bus_map) and
        returns them as-is, rather than a real second forward pass -- this
        makes the "subgraph" prediction equal the teacher's by
        construction, which should drive angle_loss/magnitude_loss/
        injection_loss/edge_loss all near zero (NOT pbl_loss -- that's an
        independent physics residual, unrelated to student/teacher
        agreement, see test_pbl_loss_independent_of_teacher below and the
        conversation this suite came from for why)."""

    def test_pbl_loss_independent_of_teacher(self):
        """self.pbl_loss should depend only on the subgraph's OWN
        prediction (sub_outputs, sub_data) -- run __call__ twice with the
        SAME self.model but DIFFERENT fixtures' outputs standing in for
        the teacher pass (data/outputs held fixed for the student side),
        confirming pbl_loss is unchanged. Confirms it never reads the
        teacher at all."""

    def test_loss_matches_manual_weighted_sum(self):
        """self.loss should equal angle_weight*angle_loss + magnitude_
        weight*magnitude_loss + injection_weight*injection_loss +
        edge_weight*edge_loss + pbl_weight*pbl_loss exactly, for an
        explicit set of non-default weights. An UNTRAINED CANOS_PF is fine
        for self.model here -- this checks the loss's own arithmetic, not
        prediction quality."""
