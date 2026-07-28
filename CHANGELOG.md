## [0.5.0] - 28.07.2026

- Add `optimize_layout` in `IQMDevice` which selects a pennylane-circuit aware initial layout based on the DQA, prioritising reducing the number of SWAPs and then number of CZ gates. Defaults to True.
  - Searching possible layouts uses Rustworkx `vf2_mapping` which is added as a dependency.
- Add `use_metrics` in IQM which uses IQM's experimental API for getting quality metrics. This selects the initial layout using calibration quality metrics. Defaults to False.

## [0.4.1] - 27.07.2026

- Fix wire mapping issue that caused the translation to use qubits that didn't exist on the backend.

## [0.4.0] - 27.07.2026

- Add `_optimize_single_qubit_gates` in `translate.py` to merge single qubit gate operations when translating pennylane tapes to IQM circuits. Circuits submitted should now contain fewer gates and a shorter duration.

## [0.3.0] - 27.07.2026

- Add `_translate_circuit` to `device.py` to translate a preprocessed pennylane circuit
- Add `device.to_iqm_circuits` to convert a PennyLane tape into IQM circuits without executing it
- Add a Demo notebook demonstrating how pennylane-iqm can be used to convert a Pennylane tape into a Pulla playlist and visualise it.


## [0.2.0] - 23.07.2026

- Fix: Decompose SWAPs inserted by topology routing before IQM translation
- Fix: Clearer error message when unsupported op reaches translator
- Add broadcast expand transformation to convert Pennylane parameter broadcasting into separate circuit descriptions.
  - Ensure 1 IQM job is submitted with batched circuits

## [0.1.0] - 22.06.2026

Initial version of Pennylane IQM targeting IQM Client 34.0.3.
