# Pennylane IQM

Unofficial, experimental and partially vibe-coded connector for Pennylane to submit to IQM quantum computers.

This project tries to bridge the gap between Pennylane and IQM Client when writing and submitting Pennylane quantum circuits. It skips Qiskit and IQM Qiskit entirely, differentiating itself from the typical `pennylane → pennylane-qiskit → qiskit-iqm` flow.

## Installation

The package is not on PyPI yet. Install from a git checkout using [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install pennylane-iqm
```

Submit a circuit:

```python
import pennylane as qml
from pennylane_iqm import IQMDevice

dev = IQMDevice(server_url="<IQM_SERVER_URL>")  # token from $IQM_TOKEN

@qml.qnode(dev, shots=1024)
def bell():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.expval(qml.Z(0) @ qml.Z(1))

print(bell())
```

## Development

Python version `3.11` and `3.12` is supported but `3.11` is recommended to compatibility with IQM Client.

### Setup

```bash
uv sync --all-groups          # main + dev + docs
uv run pre-commit install     # enable formatting/linting on commit
```

All checks run through `tox`:

```bash
uv run tox -e ruff,pyrefly    # lint, format check, type check
uv run tox -e py311,py312     # unit tests on both Python versions
uv run tox                    # everything in env_list
```

### Release

To make a new release edit the `CHANGELOG.md`.
