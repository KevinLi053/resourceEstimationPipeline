#!/usr/bin/env python3
"""List every Hamiltonian found under the HamLib tree, sorted by number of qubits.

Walks all .hdf5 files in ~/Documents/resourceEstimation/hamlib (skipping .venv)
and extracts nqubits / one_norm / n_terms from HDF5 attributes only.
Prints a markdown table with index per Hamiltonian within its HDF5 group,
sorted ascending by number of qubits. Shows full directory path from hamlib root.
"""
from __future__ import annotations

import h5py
import sys
from pathlib import Path

HAMLIB_DIR = Path(__file__).resolve().parent.parent / "hamlib"


def find_hdf5_files(root: Path) -> list[Path]:
    """Find all .hdf5 files under *root*, skipping .venv and __pycache__."""
    skip = {"__pycache__", ".venv"}
    return sorted(
        [p for p in root.rglob("*.hdf5") if not any(s in p.parts for s in skip)],
        key=lambda p: str(p.relative_to(root)),
    )

# Attr name aliases — try each in order
_NQUBITS = ["nqubits"]
_ONORM = ["one_norm", "one-norm", "onenorm", "lambda"]
_TERMS = ["terms", "n_terms", "num_terms", "nterms"]


def _get_attr(attrs, aliases):
    for a in aliases:
        if a in attrs:
            return attrs[a]
    return None


def main():
    root = HAMLIB_DIR
    if not root.is_dir():
        print(f"Error: hamlib dir not found at {root}", file=sys.stderr)
        sys.exit(1)

    files = find_hdf5_files(root)
    n = len(files)
    print(f"Found {n} HDF5 files, scanning ...", file=sys.stderr)

    rows = []  # (full_path, key, index, nqubits, one_norm, n_terms)
    ok = 0
    err = 0

    for i, h5 in enumerate(files):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n} scanned ({ok} ok, {len(rows)} results, {err} err)", file=sys.stderr)

        full = str(h5)
        try:
            with h5py.File(full, "r") as f:
                ds_at_root = []
                g_at_root = []
                for k in f.keys():
                    obj = f[k]
                    if isinstance(obj, h5py.Dataset):
                        ds_at_root.append(k)
                    else:
                        g_at_root.append(k)

                if ds_at_root and not g_at_root:
                    for idx, k in enumerate(ds_at_root):
                        attrs = f[k].attrs
                        nq = _get_attr(attrs, _NQUBITS)
                        if nq is None:
                            continue
                        on = _get_attr(attrs, _ONORM)
                        nt = _get_attr(attrs, _TERMS)
                        rows.append((full, k, idx, int(nq),
                                      float(on) if on is not None else None,
                                      int(nt) if nt is not None else None))
                    ok += 1

                elif g_at_root:
                    for idx, gk in enumerate(g_at_root):
                        grp = f[gk]
                        inner_ds = []
                        for ik in grp.keys():
                            obj = grp[ik]
                            if isinstance(obj, h5py.Dataset):
                                inner_ds.append((ik, obj))
                            else:
                                for ik2 in obj.keys():
                                    obj2 = obj[ik2]
                                    if isinstance(obj2, h5py.Dataset):
                                        inner_ds.append((f"{ik}/{ik2}", obj2))

                        for jdx, (dname, d) in enumerate(inner_ds):
                            attrs = d.attrs
                            nq = _get_attr(attrs, _NQUBITS)
                            if nq is None:
                                continue
                            on = _get_attr(attrs, _ONORM)
                            nt = _get_attr(attrs, _TERMS)
                            rows.append((full, gk + "/" + dname, jdx, int(nq),
                                          float(on) if on is not None else None,
                                          int(nt) if nt is not None else None))
                    ok += 1

        except Exception as e:
            err += 1
            print(f"  Warning: could not read {full}: {e}", file=sys.stderr)

    if n % 200 != 0:
        print(f"  {n}/{n} scanned ({ok} ok, {len(rows)} results, {err} err)", file=sys.stderr)

    # Sort by qubits ascending (None goes last)
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] else 0))

    # Print table with full path + key within file, sorted by number of qubits
    print(f"| {'idx':>5} | {'qubits':>7} | {'one_norm':>12} | {'n_terms':>8} | Hamiltonian")
    print(f"|{'-'*7}|{'-'*9}|{'-'*14}|{'-'*10}|{'-'*60}|")
    for full, key, idx, nq, on, nt in rows:
        path_key = f"{full}/{key}"
        nq_str = str(nq) if nq is not None else "-"
        on_str = f"{on:.6f}" if on is not None else "-"
        nt_str = str(nt) if nt is not None else "-"
        print(f"| {idx:>5} | {nq_str:>7} | {on_str:>12} | {nt_str:>8} | {path_key}")

    print(f"\nTotal: {len(rows)} Hamiltonians (with nqubits attr) "
          f"across {err} errors")


if __name__ == "__main__":
    main()
