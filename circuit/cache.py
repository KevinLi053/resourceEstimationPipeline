"""
HamLib circuit cache — persist and reload QASM3 circuits between pipeline runs.

Circuits are stored under the package's ``data/hamlib/`` directory in a
hierarchy that mirrors the HamLib source layout.  Two QASM3 files are kept
per Hamiltonian:

  evolved_<idx>.qasm   — raw Qiskit evolution circuit (U/CNOT, before BQSKit)
  bqskit_<idx>.qasm    — final Clifford+T circuit (after BQSKit + Stage-4 cleanup)

Directory layout
----------------
data/
└── hamlib/
    └── <domain>/           e.g. condensedmatter
        └── <type>/         e.g. heisenberg
            └── <stem>/     e.g. heis   (HDF5 filename without extension)
                ├── evolved_<idx>.qasm
                └── bqskit_<idx>.qasm

The path hierarchy is derived from the ``hdf5_path`` in ``HamlibConfig`` by
finding the ``hamlib`` component and taking everything that follows.  If the
path does not contain ``hamlib``, the parent directory name + file stem are
used as a two-level fallback.

``<idx>`` is the integer index of the selected Hamiltonian in the HDF5 file
(``HamlibConfig.key_index`` when ``key`` is ``None``; otherwise the position
of ``key`` in the file's key list).

Public API
----------
cache_dir_for(hdf5_path) -> Path
evolved_path(hdf5_path, idx) -> Path
final_circuit_path(hdf5_path, idx, mode) -> Path
bqskit_path(hdf5_path, idx) -> Path          (backward-compat alias for mode='bqskit')
save_circuit(circuit, path) -> None
load_circuit(path) -> QuantumCircuit
resolve_index(config) -> int
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from ..config import HamlibConfig

log = logging.getLogger(__name__)

# data/ lives one level above this file's package directory
# circuit/cache.py → circuit/ → resourceEstimationPipeline/ → data/
_DATA_ROOT: Path = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def cache_dir_for(hdf5_path: str) -> Path:
    """Return the cache directory for a HamLib HDF5 file.

    Locates the ``hamlib`` component in the resolved path and uses every
    component after it as the sub-path under ``data/hamlib/``.  The file
    extension is stripped from the final component.

    Examples
    --------
    ./../hamlib/condensedmatter/heisenberg/heis.hdf5
        → data/hamlib/condensedmatter/heisenberg/heis/

    /absolute/hamlib/discreteoptimization/maxkcut/color02/ham.hdf5
        → data/hamlib/discreteoptimization/maxkcut/color02/ham/
    """
    p = Path(hdf5_path).resolve()
    parts = p.parts

    hamlib_idx = next(
        (i for i, part in enumerate(parts) if "hamlib" in part.lower()),
        None,
    )

    if hamlib_idx is not None:
        subpath_parts = list(parts[hamlib_idx + 1:])
        if subpath_parts:
            subpath_parts[-1] = Path(subpath_parts[-1]).stem
    else:
        # Fallback: parent dir name + file stem
        subpath_parts = [p.parent.name, p.stem]

    return _DATA_ROOT / "hamlib" / Path(*subpath_parts)


def evolved_path(hdf5_path: str, idx: int) -> Path:
    """QASM3 path for the evolved (pre-transpile) circuit."""
    return cache_dir_for(hdf5_path) / f"evolved_{idx}.qasm"


_VALID_MODES = frozenset({"bqskit", "solovay_kitaev", "passthrough"})


def final_circuit_path(hdf5_path: str, idx: int, mode: str) -> Path:
    """QASM3 path for the final transpiled/synthesized circuit.

    Each synthesis mode gets its own file so switching modes never loads
    a stale result from a previous run.

    Parameters
    ----------
    mode : str
        ``'bqskit'``         — BQSKit CliffordTModel synthesis (pure Clifford+T)
        ``'solovay_kitaev'`` — Qiskit Solovay-Kitaev synthesis (pure Clifford+T)
        ``'passthrough'``    — rotation_synthesis_enabled=False (rotations preserved)
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown synthesis mode {mode!r}. Supported: {sorted(_VALID_MODES)}"
        )
    return cache_dir_for(hdf5_path) / f"{mode}_{idx}.qasm"


def bqskit_path(hdf5_path: str, idx: int) -> Path:
    """Backward-compatible alias for final_circuit_path(hdf5_path, idx, 'bqskit')."""
    return final_circuit_path(hdf5_path, idx, "bqskit")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_circuit(circuit: QuantumCircuit, path: Path) -> None:
    """Write *circuit* to *path* as QASM3, creating parent directories."""
    from qiskit.qasm3 import Exporter
    path.parent.mkdir(parents=True, exist_ok=True)
    qasm_str = Exporter().dumps(circuit)
    path.write_text(qasm_str, encoding="utf-8")
    log.info("[cache] Saved %d-qubit circuit to %s (%d bytes)",
             circuit.num_qubits, path, len(qasm_str))
    print(f"[cache] Saved → {path}")


def load_circuit(path: Path) -> QuantumCircuit:
    """Load a QuantumCircuit from a QASM3 file at *path*."""
    from qiskit.qasm3 import loads as qasm3_loads
    qasm_str = path.read_text(encoding="utf-8")
    circuit = qasm3_loads(qasm_str)
    log.info("[cache] Loaded circuit from %s", path)
    print(f"[cache] Loaded ← {path}")
    return circuit


# ---------------------------------------------------------------------------
# Index resolution
# ---------------------------------------------------------------------------

def resolve_index(config: HamlibConfig) -> int:
    """Return the integer index of the selected Hamiltonian in the HDF5 file.

    When ``config.key`` is ``None`` the value is ``config.key_index``.
    When ``config.key`` is set the key list is scanned to find its position.
    """
    if config.key is None:
        return config.key_index

    from ..loaders.hamlib_loader import get_hamlib_hdf5_keys
    all_keys, _ = get_hamlib_hdf5_keys(str(config.hdf5_path))
    try:
        return all_keys.index(config.key)
    except ValueError as exc:
        raise ValueError(
            f"Key {config.key!r} not found in {config.hdf5_path!r}. "
            f"Available keys (first 5): {all_keys[:5]}"
        ) from exc
