"""Two-round 15-to-1 magic state distillation.

Extends Qualtran's FifteenToOne (arXiv:1905.06903) to accept an explicit
input T-state error probability (p_in).  When p_in is None the class is
identical to the upstream FifteenToOne (round-1 / raw-T behaviour).

Usage
-----
Round 1 — identical to upstream:
    r1 = FifteenToOneTwoRound(d_X=3, d_Z=3, d_m=3)
    p_out1 = r1.p_out(logical_error_model)

Round 2 — distilled T states from round 1 fed in:
    r2 = FifteenToOneTwoRound(d_X=3, d_Z=3, d_m=3, input_t_error=p_out1)
    p_out2 = r2.p_out(logical_error_model)

Two-round combined factory (for use with PhysicalCostModel):
    from qualtran.surface_code import MultiFactory
    r1_mf = MultiFactory(FifteenToOne(d_X, d_Z, d_m), n_fac_r1)
    r2_mf = MultiFactory(FifteenToOneTwoRound(d_X, d_Z, d_m, p_out1), n_fac_r2)
    two_round = TwoRoundFactory(r1_factory=r1_mf, r2_factory=r2_mf)
"""

from functools import lru_cache
import math
from typing import Optional, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen

from qualtran.surface_code.magic_state_factory import MagicStateFactory
from qualtran.surface_code.t_factory_utils import NoisyPauliRotation, storage_error

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import LogicalErrorModel


def _build_factory_with_input_error(
    *,
    d_X: int,
    d_Z: int,
    d_m: int,
    logical_error_model: 'LogicalErrorModel',
    input_t_error: Optional[float] = None,
) -> cirq.Circuit:
    """15-to-1 factory circuit with a configurable input T-state error.

    Identical to qualtran's _build_factory except that every ``phys_err/3``
    T-injection term is replaced by ``p_in/3``, where p_in defaults to
    ``phys_err`` (preserving round-1 behaviour) and can be set to the output
    error of a prior distillation round.

    Surface-code operational error terms (px, pz, pm) are unchanged.
    """
    qs = cirq.LineQubit.range(5)
    px = logical_error_model(d_X)
    pz = logical_error_model(d_Z)
    pm = logical_error_model(d_m)
    phys_err = logical_error_model.physical_error

    # p_in: per-T-state input error probability.
    # Round 1: p_in = phys_err  (raw physical T states).
    # Round 2: p_in = p_out1   (distilled T states from round 1).
    p_in = phys_err if input_t_error is None else input_t_error

    factory = cirq.Circuit.from_moments(
        cirq.H.on_each(qs),
        # 1
        NoisyPauliRotation(
            'IZIII',
            p_in / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            p_in / 3 + 0.5 * d_Z * pm,
            p_in / 3,
        )(*qs),
        # 2
        NoisyPauliRotation(
            'IIZII',
            p_in / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            p_in / 3 + 0.5 * d_Z * pm,
            p_in / 3,
        )(*qs),
        # 3
        NoisyPauliRotation(
            'IIIZI',
            p_in / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            p_in / 3 + 0.5 * d_Z * pm,
            p_in / 3,
        )(*qs),
        # 5
        NoisyPauliRotation(
            'IZZZI',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (3 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        *storage_error(
            'X',
            [
                0,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0,
            ],
            qs,
        ),
        # 6
        NoisyPauliRotation(
            'ZZZII',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 2 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 7
        NoisyPauliRotation(
            'ZZIZI',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 3 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 2 * d_Z) + (d_X + 3 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0,
            ],
            qs,
        ),
        # 8
        NoisyPauliRotation(
            'ZIZZI',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 3 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 9
        NoisyPauliRotation(
            'ZIIZZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 4
        NoisyPauliRotation(
            'IIIIZ',
            p_in / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            p_in / 3 + 0.5 * d_Z * pm,
            p_in / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 3 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 10
        NoisyPauliRotation(
            'ZZIIZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 11
        NoisyPauliRotation(
            'ZIZIZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 4 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 12
        NoisyPauliRotation(
            'ZZZZZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 13
        NoisyPauliRotation(
            'IIZZZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (3 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        *storage_error('Z', [0.5 * (d_X + 4 * d_Z) / d_X * px * d_m, 0, 0, 0, 0], qs),
        *storage_error(
            'X',
            [
                0.5 * px * (d_m + 2 * d_X),
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * (d_m + 2 * d_X),
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 14
        NoisyPauliRotation(
            'IZIZZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        # 15
        NoisyPauliRotation(
            'IZZIZ',
            p_in / 3 + 0.5 * pm * d_m,
            p_in / 3 + 0.5 * pm * d_m + 0.5 * (4 * d_Z) * d_X / d_m * pm,
            p_in / 3,
        )(*qs),
        *storage_error(
            'X',
            [
                0,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
    )
    return factory


@frozen
class FifteenToOneTwoRound(MagicStateFactory):
    """15-to-1 factory supporting an explicit input T-state error (for two-round distillation).

    When ``input_t_error`` is None this class is numerically identical to
    ``qualtran.surface_code.FifteenToOne``.  Set it to the ``p_out`` of a
    prior round to simulate chained distillation.

    References:
        [Magic State Distillation: Not as Costly as You Think](https://arxiv.org/abs/1905.06903).
    """

    d_X: int
    d_Z: int
    d_m: int
    input_t_error: Optional[float] = None

    def __attrs_post_init__(self):
        assert 0 < self.d_X <= 3 * self.d_m
        assert self.d_m > 0
        assert self.d_Z > 0

    def n_physical_qubits(self) -> int:
        return 2 * (self.d_X + 4 * self.d_Z) * 3 * self.d_X + 4 * self.d_m

    @lru_cache(8)
    def _final_state(self, logi_err_model: 'LogicalErrorModel'):
        factory = _build_factory_with_input_error(
            d_X=self.d_X,
            d_Z=self.d_Z,
            d_m=self.d_m,
            logical_error_model=logi_err_model,
            input_t_error=self.input_t_error,
        )
        return (
            cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(factory).final_density_matrix
        )

    @lru_cache(8)
    def p_fail(self, logical_error_model: 'LogicalErrorModel') -> float:
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        return np.real_if_close(
            1 - np.trace(projector @ self._final_state(logical_error_model))
        ).item()

    @lru_cache(8)
    def p_out(self, logical_error_model: 'LogicalErrorModel') -> float:
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        project_state = (
            1
            / (1 - self.p_fail(logical_error_model))
            * (projector @ self._final_state(logical_error_model) @ projector.T.conj())
        )
        T_state = np.array([1, np.exp(-1j * np.pi / 4)]).reshape((1, 2)) / np.sqrt(2)
        target_density = np.kron(T_state.T.conj() @ T_state, np.ones((16, 16)) / 16)
        return np.real_if_close(1 - np.trace(project_state @ target_density)).item()

    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        num_t = n_logical_gates.total_t_count()
        return int(np.ceil(num_t * 6 * self.d_m / (1 - self.p_fail(logical_error_model))))

    def factory_error(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> float:
        num_t = n_logical_gates.total_t_count()
        return self.p_out(logical_error_model) * num_t


@frozen
class TwoRoundFactory(MagicStateFactory):
    """Combined two-round 15-to-1 factory for use with PhysicalCostModel.

    Wraps a parallel fleet of round-1 factories (FifteenToOne) and a parallel
    fleet of round-2 factories (FifteenToOneTwoRound) as a single factory
    object.  Both fleets run simultaneously in a pipeline:

      * Round 1 must produce 15 T states per final output T state.
      * Round 2 consumes those 15 and distills one output T state per run.

    n_cycles() returns the pipeline bottleneck — the larger of:
      * ceil(15 * r1_base_cycles / n_fac_r1)   (round-1 must produce 15x)
      * ceil(r2_base_cycles / n_fac_r2)         (round-2 produces finals)

    factory_error() returns only the round-2 output error (p2 * N_T).
    Round-1 error is already encoded in p2 via the density-matrix simulation
    and must NOT be added separately.

    n_physical_qubits() = r1 fleet qubits + r2 fleet qubits.

    Attributes:
        r1_factory: MultiFactory wrapping FifteenToOne for round 1.
        r2_factory: MultiFactory wrapping FifteenToOneTwoRound for round 2.
    """

    r1_factory: MagicStateFactory
    r2_factory: MagicStateFactory

    def n_physical_qubits(self) -> int:
        return self.r1_factory.n_physical_qubits() + self.r2_factory.n_physical_qubits()

    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        # Unpack the MultiFactory to get base cycles and factory count.
        # 15 * ceil(x/n) != ceil(15x/n) in general, so we must work with
        # the raw base cycles to avoid overcounting.
        r1_base = getattr(self.r1_factory, 'base_factory', self.r1_factory)
        n_fac_r1 = getattr(self.r1_factory, 'n_factories', 1)
        r1_base_cycles = r1_base.n_cycles(n_logical_gates, logical_error_model)
        r1_pipeline = math.ceil(15 * r1_base_cycles / n_fac_r1)

        r2_base = getattr(self.r2_factory, 'base_factory', self.r2_factory)
        n_fac_r2 = getattr(self.r2_factory, 'n_factories', 1)
        r2_base_cycles = r2_base.n_cycles(n_logical_gates, logical_error_model)
        r2_pipeline = math.ceil(r2_base_cycles / n_fac_r2)

        return max(r1_pipeline, r2_pipeline)

    def factory_error(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> float:
        return self.r2_factory.factory_error(n_logical_gates, logical_error_model)
