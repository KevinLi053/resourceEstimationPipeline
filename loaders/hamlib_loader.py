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
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ..config import HamlibConfig


# ---------------------------------------------------------------------------
# HamLib HDF5 readers
# ---------------------------------------------------------------------------

def _parse_hdf5_recursive(func):
    """Decorator that recursively iterates through HDF5 file and performs
    some action that can be specified by `func` on the internal and leaf
    nodes in the file."""
    def wrapper(obj, path='/', key=None):
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for ky in obj.keys():
                func(obj, path, key=ky, leaf=False)
                wrapper(obj=obj[ky], path=path + ky + '/', key=ky)
        elif type(obj) is h5py._hl.dataset.Dataset:
            func(obj, path, key=None, leaf=True)
    return wrapper


def get_hierarchical_hdf5_keys(fname_hdf5):
    """Get list of full path keys in hdf5 file. (Applicable to any
    "hierarchical" HamLib hdf5 file)."""
    all_keys = []

    @_parse_hdf5_recursive
    def action(obj, path='/', key=None, leaf=False):
        if leaf is True:
            all_keys.append(path)

    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])

    return all_keys

def get_hdf5_keys(fname_hdf5):
    """Get list of keys in hdf5 file. (Applicable to any "flat" HamLib hdf5
    file)"""

    with h5py.File(fname_hdf5, 'r') as f:
        keys = list(f.keys())

    return keys

def get_hamlib_hdf5_keys(fname_hdf5: str):
    """
    Automatically detect HamLib HDF5 layout and return the appropriate keys.

    Uses:
        - get_hierarchical_hdf5_keys() for hierarchical HDF5 files
        - get_hdf5_keys() for flat HDF5 files

    Returns:
        tuple:
            keys (List[str]): extracted dataset keys
            layout (str): detected layout type
    """

    with h5py.File(fname_hdf5, "r") as f:
        has_group = False
        has_dataset = False

        def check_layout(name, obj):
            nonlocal has_group, has_dataset

            if isinstance(obj, h5py.Group):
                has_group = True
            elif isinstance(obj, h5py.Dataset):
                has_dataset = True

        f.visititems(check_layout)

    # Decide layout
    if has_group:
        layout = "hierarchical"
        keys = get_hierarchical_hdf5_keys(fname_hdf5)
    else:
        layout = "flat"
        keys = get_hdf5_keys(fname_hdf5)

    return keys, layout


def read_qiskit_hdf5(fname_hdf5: str, key: str):
    """
    Read the operator object from HDF5 at specified key to qiskit SparsePauliOp
    format.
    """
    def _generate_string(term):
        # change X0 Z3 to XIIZ; empty term = identity
        indices = [
            (m.group(1), int(m.group(2)))
            for m in re.finditer(r'([A-Z])(\d+)', term)
        ]
        if not indices:
            return 'I'  # identity term; _append_ids pads to full width
        return ''.join(
            [next((char for char, idx in indices if idx == i), 'I')
             for i in range(max(idx for _, idx in indices) + 1)]
        )

    def _append_ids(pstrings):
        # append Ids to strings
        return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]

    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        # Match both plain-real coefficients (e.g. "1.0", "-0.5") used in
        # Heisenberg-style files, and complex coefficients in parentheses
        # (e.g. "(-0.5+0j)") used in Fermi-Hubbard/BoseHubbard/etc. files.
        # [^\]]* (zero-or-more) instead of [^\]]+ also captures the identity
        # term written as [] with empty brackets.
        pattern = r"(?:\(([^)]+)\)|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))\s+\[([^\]]*)\]"
        data = f[key][()]
        if isinstance(data, bytes):
            text = data.decode("utf-8")
        elif isinstance(data, str):
            text = data
        elif isinstance(data, np.ndarray):
            # Decode elements if necessary, then join into one string
            if data.dtype.kind == "S":  # byte strings
                text = "\n".join(x.decode("utf-8") for x in data.flat)
            else:  # already Unicode strings
                text = "\n".join(map(str, data.flat))
        else:
            text = str(data)

        matches = re.findall(pattern, text)

        labels = []
        coeffs = []
        for complex_part, real_part, term in matches:
            labels.append(_generate_string(term))
            coeffs.append(complex(complex_part) if complex_part else float(real_part))

        op = SparsePauliOp(_append_ids(labels), coeffs)
    return op

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
    all_keys = get_hamlib_hdf5_keys(path)[0]
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
