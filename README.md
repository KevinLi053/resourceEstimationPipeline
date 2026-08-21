# QDK vs. Qualtran Resource Estimation Comparison

This project provides a unified pipeline to compare the resource estimation capabilities of two major quantum computing frameworks: **Microsoft Azure QDK** and **Google Qualtran**. 

The goal is to perform an "apples-to-apples" comparison of physical qubit counts and runtime estimates for Hamiltonian-based quantum circuits, using a shared, canonical Clifford+T basis.

## Overview

The pipeline automates the entire workflow of generating, transpiling, and estimating the resources required for complex quantum circuits:

1.  **Load Hamiltonian**: Loads Hamiltonian data from **HamLib** HDF5 files.
2.  **Circuit Construction**: Builds a Pauli evolution circuit based on the loaded Hamiltonian.
3.  **Transpilation**: Transpiles the raw evolution circuit into a canonical **Clifford+T+Rz** basis with some support for **Clifford+T** gates (uses **BQSKit**).
4.  **Estimation (Azure QDK)**: Runs the Azure QDK resource estimator to determine physical qubit and runtime requirements.
5.  **Estimation (Qualtran)**: Runs the Google Qualtran resource estimator using the same Clifford+T circuit.
6.  **Comparison**: Generates a side-by-side report comparing metrics like physical qubits, runtime, and gate counts.

## Setup and Installation

### Prerequisites
* Python 3.10 or higher.

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd qdk-qualtran-comparison
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -r requirements.txt
   ./third_party/setup_local_deps.sh # To get error splits and change source code
   ```

## Getting Hamiltonians
The link to the full dataset can be found here <https://portal.nersc.gov/cfs/m888/dcamps/hamlib/>
to download Hamiltonians.

Run `list_hamlib_by_qubits.py` to get qubit and key breakdown of hamlib files.

## How to Run

The pipeline is designed to be used as a library mainly for Jupyter notebooks.

### Running via Jupyter notebooks
Use `notebooks/comparison.ipynb` to get a step by step breakdown of entire process.

### Configuration
All major parameters—including Hamiltonian paths, evolution time, transpilation strategy, and estimator-specific knobs (like error budgets and code distances)—are centralized in:
`src/qdk_qualtran_comparison/config.py`

### Experiments and Analysis
You can find various test cases and experimental workflows in the following directories:
* `experiments/`: Jupyter notebooks for testing specific circuit generation and comparison logic.

## Repository Structure

* `src/qdk_qualtran_comparison/`: The core package containing the pipeline, estimators, and circuit logic.
* `data/`: Contains Hamiltonian data (in QASM format) used for the experiments.
* `experiments/`: A collection of Jupyter notebooks demonstrating the pipeline's usage and testing different configurations.
* `notebooks/`: Analytical notebooks for visualizing and interpreting the comparison results.