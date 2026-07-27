## [0.4.0] - 27.07.2026

- Add `_optimize_single_qubit_gates` in `translate.py` to merge single qubit gate operations when translating pennylane tapes to IQM circuits.

## [0.3.0] - 27.07.2026

- Add `_translate_circuit` to `device.py` to translate a preprocessed pennylane circuit
- Add `device.to_iqm_circuits` to convert a PennyLane tape into IQM circuits without executing it
- Add a Demo notebook demonstrating how pennylane-iqm can be used to convert a Pennylane tape into a Pulla playlist and visualise it.


## [0.2.0] - 23.07.2026

- Fix: Add decomposition transformation to fix additional SWAPs being added
- Fix: Clearer error message when unsupported op reaches translator
- Add broadcast expand transformation to convert Pennylane parameter broadcasting into separate circuit descriptions.
  - Ensure 1 IQM job is submitted with batched circuits

## [0.1.0] - 22.06.2026

Initial version of Pennylane IQM targeting IQM Client 34.0.3.
