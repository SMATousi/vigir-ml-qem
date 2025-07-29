import os
import json
from pathlib import Path
from typing import List, Dict
from concurrent.futures import TimeoutError as FuturesTimeoutError
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.qpy import load
from qiskit_aer import AerSimulator
from qiskit import transpile
from tqdm import tqdm
# from qiskit_ibm_runtime.fake_provider import   # keep your chosen fake backend

def simulate_and_store_z_expectations_json(
    qpy_folder: str,
    output_json_path: str,
    shots: int = 1024,
    per_circuit_timeout_s: float = 30.0,
) -> None:
    """
    Load QPY circuits, simulate (ideal & noisy), compute <Z> per qubit, and
    save to a JSON file. Skips any circuit whose simulation exceeds the timeout.

    Parameters
    ----------
    qpy_folder : str
        Folder path containing .qpy files.
    output_json_path : str
        Path to save the output JSON.
    shots : int
        Number of shots for each simulation.
    per_circuit_timeout_s : float
        Seconds to wait for each simulator job before skipping.
    """

    def counts_to_z_expectation(counts: Dict[str, int], n_qubits: int) -> List[float]:
        """Supports hex ('0x...') and binary ('0101') keys."""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return [0.0] * n_qubits

        expectations = [0.0] * n_qubits
        for key, cnt in counts.items():
            if key.startswith("0x") or key.startswith("0X"):
                val = int(key, 16)
            else:
                # treat as binary bitstring like '0101'
                # if something else, this will raise which is fine
                val = int(key, 2)
            bitstring = bin(val)[2:].zfill(n_qubits)[::-1]  # little-endian
            for i in range(n_qubits):
                z = 1 if bitstring[i] == "0" else -1
                expectations[i] += z * cnt

        return [round(e / total_shots, 6) for e in expectations]

    def unique_key(preferred: str, fallback_base: str, taken: set) -> str:
        base = preferred.strip() if preferred and preferred.strip() else fallback_base
        if base not in taken:
            taken.add(base)
            return base
        k = 2
        while True:
            candidate = f"{base}#{k}"
            if candidate not in taken:
                taken.add(candidate)
                return candidate
            k += 1

    results_dict: Dict[str, Dict[str, List[float]]] = {}

    fake_backend = FakeFez()
    noisy_sim = AerSimulator.from_backend(fake_backend)
    ideal_sim = AerSimulator()

    taken_names = set()

    for file in tqdm(sorted(os.listdir(qpy_folder))):
        if not file.endswith(".qpy"):
            continue

        file_path = os.path.join(qpy_folder, file)
        try:
            with open(file_path, "rb") as f:
                circuits = load(f)  # returns a list of QuantumCircuit objects
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
            continue

        for idx, qc in enumerate(circuits):
            n_qubits = qc.num_qubits
            # circuit_key = unique_key(qc.name, f"{Path(file).stem}_{idx}", taken_names)
            circuit_key = file
            # print(f"• Simulating '{circuit_key}' ({n_qubits} qubits)...")

            try:
                tqc_ideal = transpile(qc, backend=ideal_sim)
                tqc_noisy = transpile(qc, backend=noisy_sim)

                job_ideal = ideal_sim.run(tqc_ideal, shots=shots)
                job_noisy = noisy_sim.run(tqc_noisy, shots=shots)

                ideal_result = job_ideal.result(timeout=per_circuit_timeout_s)
                noisy_result = job_noisy.result(timeout=per_circuit_timeout_s)

                # Use the first experiment's counts (each run has one circuit here)
                ideal_counts = ideal_result.results[0].data.counts
                noisy_counts = noisy_result.results[0].data.counts

                z_ideal = counts_to_z_expectation(ideal_counts, n_qubits)
                z_noisy = counts_to_z_expectation(noisy_counts, n_qubits)

                results_dict[circuit_key] = {
                    "z_ideal": z_ideal,
                    "z_noisy": z_noisy,
                }


            except FuturesTimeoutError:
                print(f"⏱️ Timeout (> {per_circuit_timeout_s:.0f}s). Skipping '{circuit_key}'.")
                continue
            except Exception as e:
                print(f"❌ Error simulating '{circuit_key}': {e}")
                continue
        # break

            

    with open(output_json_path, "w") as json_out:
        json.dump(results_dict, json_out, indent=2)

    print(f"\n✅ Results saved to {output_json_path}")


simulate_and_store_z_expectations_json(
    qpy_folder="/home/macula/SMATousi/projects/quantum/andrew/ExecutionResults/StoredCircuits/",
    output_json_path="z_expectations.json",
    shots=1024
)

