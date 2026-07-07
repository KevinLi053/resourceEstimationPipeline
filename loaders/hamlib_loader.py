"""
HamLib Hamiltonian loader.

Uses ``read_qiskit_hdf5`` to load a ``SparsePauliOp`` directly from a
HamLib HDF5 file.  No intermediate text parsing is performed.

Public API
----------
load_hamiltonian(config) -> HamiltonianData
    Load a SparsePauliOp and associated metadata from a HamLib HDF5 file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import h5py
from qiskit.quantum_info import SparsePauliOp

from resourceEstimationPipeline.config import HamlibConfig


# ---------------------------------------------------------------------------
# HamLib HDF5 readers
# ---------------------------------------------------------------------------

def get_hdf5_keys(fname_hdf5: str) -> List[str]:
    """Return all leaf dataset keys in a HamLib HDF5 file."""
    keys: List[str] = []
    with h5py.File(fname_hdf5, "r") as f:
        f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
    return keys


def read_qiskit_hdf5(fname_hdf5: str, key: str) -> SparsePauliOp:
    """Read a HamLib dataset directly into a Qiskit SparsePauliOp."""
    def _generate_string(term: str) -> str:
        indices = [(m.group(1), int(m.group(2))) for m in re.finditer(r"([A-Z])(\d+)", term)]
        return "".join(next((c for c, i in indices if i == j), "I") for j in range(max(i for _, i in indices) + 1))

    def _append_ids(pstrings: List[str]) -> List[str]:
        mx = max(map(len, pstrings))
        return [p + "I" * (mx - len(p)) for p in pstrings]

    with h5py.File(fname_hdf5, "r", libver="latest") as f:
        pattern = r"([\d.]+) \[([^\]]+)\]"
        matches = re.findall(pattern, f[key][()].decode("utf-8"))
        labels = [_generate_string(m[1]) for m in matches]
        coeffs = [float(m[0]) for m in matches]
    return SparsePauliOp(_append_ids(labels), coeffs)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HamiltonianData:
    """All information extracted for a single HamLib dataset."""

    hdf5_path: str
    key: str

    # Parsed SparsePauliOp
    pauli_op: SparsePauliOp

    # Metadata from HDF5 attributes
    nqubits: int
    one_norm: Optional[float]
    n_terms: Optional[int]

    def __repr__(self) -> str:
        return (
            f"HamiltonianData(key={self.key!r}, "
            f"nqubits={self.nqubits}, "
            f"n_terms={self.n_terms}, "
            f"one_norm={self.one_norm})"
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_hamiltonian(config: HamlibConfig) -> HamiltonianData:
    """
    Load a Hamiltonian from a HamLib HDF5 file according to `config`.

    Steps
    -----
    1. Open the HDF5 file and collect all dataset keys.
    2. Select a key: use ``config.key`` if set; otherwise pick by ``config.key_index``.
    3. Read metadata attributes (nqubits, one_norm, terms) with fallback aliases.
    4. Build a ``SparsePauliOp`` via ``read_qiskit_hdf5``.
    5. Return a :class:`HamiltonianData` with the op and all metadata.

    Parameters
    ----------
    config : HamlibConfig

    Returns
    -------
    HamiltonianData
    """
    path = str(config.hdf5_path)

    # 1. Collect keys
    all_keys = get_hdf5_keys(path)
    if not all_keys:
        raise ValueError(f"No datasets found in {path!r}")

    # 2. Select key
    if config.key is not None:
        key = config.key
    else:
        if config.key_index >= len(all_keys):
            raise IndexError(
                f"key_index {config.key_index} out of range "
                f"(file has {len(all_keys)} datasets)"
            )
        key = all_keys[config.key_index]

    # 3. Read metadata attributes
    nqubits: Optional[int] = None
    one_norm: Optional[float] = None
    n_terms: Optional[int] = None

    with h5py.File(path, "r") as f:
        attrs = dict(f[key].attrs)
        if "nqubits" in attrs:
            nqubits = int(attrs["nqubits"])
        for _attr in ("one_norm", "one-norm", "onenorm", "lambda"):
            if _attr in attrs:
                one_norm = float(attrs[_attr])
                break
        for _attr in ("terms", "n_terms", "num_terms", "nterms"):
            if _attr in attrs:
                n_terms = int(attrs[_attr])
                break

    # 4. Build SparsePauliOp via read_qiskit_hdf5
    pauli_op = read_qiskit_hdf5(path, key)
    if nqubits is None:
        nqubits = pauli_op.num_qubits

    return HamiltonianData(
        hdf5_path=path,
        key=key,
        pauli_op=pauli_op,
        nqubits=nqubits,
        one_norm=one_norm,
        n_terms=n_terms,
    )
