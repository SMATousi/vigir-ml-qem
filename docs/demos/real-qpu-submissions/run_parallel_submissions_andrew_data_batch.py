import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_ibm_runtime.runtime_job import (
    RuntimeJobMaxTimeoutError,
    RuntimeJobFailureError,
    RuntimeInvalidStateError,
)

import qiskit.qpy as qpy  # for loading .qpy circuits


def simulate_and_store_z_expectations_real_qpu_json(
    qpy_folder: str,
    output_json_path: str,
    shots: int = 1024,
    per_circuit_timeout_s: float = 60.0,
    backend_name: str = "ibm_fez",
    batch_size: int = 100,
    service: Optional[QiskitRuntimeService] = None,
) -> None:
    """
    Load all QPY circuits, run ideal Aer ⟨Z⟩, and execute real hardware in batches of `batch_size`
    concurrent jobs. Saves per-circuit shot-level JSONs and a summary JSON with z_ideal,
    z_hardware, and hardware_job_id.
    """

    def counts_to_z_expectation(counts: Dict[str, int], n_qubits: int) -> List[float]:
        total = sum(counts.values())
        if total == 0:
            return [0.0] * n_qubits
        exp = [0.0] * n_qubits
        for key, cnt in counts.items():
            if key.startswith(("0x", "0X")):
                val = int(key, 16)
            else:
                val = int(key, 2)
            bits = bin(val)[2:].zfill(n_qubits)[::-1]  # little-endian
            for i in range(n_qubits):
                z = 1 if bits[i] == "0" else -1
                exp[i] += z * cnt
        return [round(e / total, 6) for e in exp]

    def unique_key(preferred: str, fallback_base: str, taken: Set[str]) -> str:
        base = preferred.strip() if preferred and preferred.strip() else fallback_base
        if base not in taken:
            taken.add(base)
            return base
        k = 2
        while True:
            cand = f"{base}#{k}"
            if cand not in taken:
                taken.add(cand)
                return cand
            k += 1

    def has_measurements(qc) -> bool:
        return any(inst.name == "measure" for inst, _, _ in qc.data)

    def sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def get_job_id(job):
        try:
            return job.job_id()
        except Exception:
            return getattr(job, "job_id", None)

    if service is None:
        service = QiskitRuntimeService()  # user manages authentication externally

    # retrieve backend object
    backends = service.backends(name=backend_name)
    if not backends:
        raise RuntimeError(f"No backend found with name '{backend_name}'")
    backend_obj = backends[0]

    ideal_sim = AerSimulator()

    results_dict: Dict[str, Dict[str, object]] = {}
    taken_names: Set[str] = set()

    output_base = Path(output_json_path)
    shot_dir = output_base.parent / (output_base.stem + "_shot_data")
    shot_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: gather all circuits, run ideal, prepare hardware-transpiled circuits ===
    hw_queue: List[Dict] = []  # entries: circuit_key, safe_key, n_qubits, tqc_hw, z_ideal

    all_files = os.listdir(qpy_folder)
    all_files = all_files[:200]
    for file in tqdm(sorted(all_files), desc="Loading QPY circuits"):
        if not file.endswith(".qpy"):
            continue
        file_path = os.path.join(qpy_folder, file)
        try:
            with open(file_path, "rb") as f:
                circuits = qpy.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
            continue

        for idx, qc in enumerate(circuits):
            n_qubits = qc.num_qubits
            #circuit_key = unique_key(qc.name, f"{Path(file).stem}_{idx}", taken_names)
            #safe_key = sanitize(circuit_key)
            circuit_key = file
            safe_key = file
            # === Ideal Aer run ===
            qc_ideal = qc.copy()
            if not has_measurements(qc_ideal):
                qc_ideal = qc_ideal.copy()
                qc_ideal.measure_all()

            try:
                tqc_ideal = transpile(qc_ideal, backend=ideal_sim)
                job_ideal = ideal_sim.run(tqc_ideal, shots=shots, memory=True)
                ideal_result = job_ideal.result()
                ideal_counts = ideal_result.get_counts()
                z_ideal = counts_to_z_expectation(ideal_counts, n_qubits)
                try:
                    ideal_bitstrings = ideal_result.get_memory()
                except Exception:
                    ideal_bitstrings = []
                    for b, c in ideal_counts.items():
                        ideal_bitstrings.extend([b] * c)

                # write ideal shot-level file
                ideal_payload = {
                    "z": z_ideal,
                    "counts": ideal_counts,
                    "bitstrings": ideal_bitstrings,
                }
                with open(shot_dir / f"{safe_key}__ideal_shots.json", "w") as f_out:
                    json.dump(ideal_payload, f_out, indent=2)
            except Exception as e:
                print(f"❌ Aer (ideal) failed for '{circuit_key}': {e}")
                continue  # skip this circuit entirely

            # === Transpile for hardware ===
            qc_hw = qc.copy()
            if not has_measurements(qc_hw):
                qc_hw = qc_hw.copy()
                qc_hw.measure_all()

            try:
                tqc_hw = transpile(qc_hw, backend=backend_obj)
            except Exception as e:
                print(f"❌ Transpile-to-hardware failed for '{circuit_key}': {e}")
                continue

            hw_queue.append(
                {
                    "circuit_key": circuit_key,
                    "safe_key": safe_key,
                    "n_qubits": n_qubits,
                    "tqc_hw": tqc_hw,
                    "z_ideal": z_ideal,
                }
            )

    if not hw_queue:
        print("No circuits prepared for hardware execution.")
        return

    # === Phase 2: hardware submission in batches of batch_size ===
    with Session(backend_obj) as session:  # reuse one session for all batches. :contentReference[oaicite:0]{index=0}
        sampler = Sampler(mode=session)  # SamplerV2 in session mode. :contentReference[oaicite:1]{index=1}

        # process in chunks
        for i in range(0, len(hw_queue), batch_size):
            batch = hw_queue[i : i + batch_size]
            jobs_info = []  # to collect submitted jobs for this batch

            # submit all in this batch
            for entry in batch:
                circuit_key = entry["circuit_key"]
                safe_key = entry["safe_key"]
                n_qubits = entry["n_qubits"]
                tqc_hw = entry["tqc_hw"]
                z_ideal = entry["z_ideal"]

                try:
                    job_hw = sampler.run([tqc_hw], shots=shots)
                    hw_job_id = get_job_id(job_hw)  # record job id. :contentReference[oaicite:2]{index=2}
                    jobs_info.append(
                        {
                            "circuit_key": circuit_key,
                            "safe_key": safe_key,
                            "n_qubits": n_qubits,
                            "job_hw": job_hw,
                            "z_ideal": z_ideal,
                            "hardware_job_id": hw_job_id,
                        }
                    )
                    print(f"Submitted hardware job for '{circuit_key}' id={hw_job_id}")
                except Exception as e:
                    print(f"❌ Submission failed for '{circuit_key}': {e}")
                    continue

            # wait for all in the batch to finish and collect
            for job_entry in jobs_info:
                circuit_key = job_entry["circuit_key"]
                safe_key = job_entry["safe_key"]
                n_qubits = job_entry["n_qubits"]
                job_hw = job_entry["job_hw"]
                z_ideal = job_entry["z_ideal"]
                hardware_job_id = job_entry["hardware_job_id"]

                try:
                    pub_result = job_hw.result(timeout=per_circuit_timeout_s)[0]  # blocking. :contentReference[oaicite:3]{index=3}

                    # extract counts: preferred current SamplerV2 pattern. :contentReference[oaicite:4]{index=4}
                    try:
                        hw_counts = pub_result.data.meas.get_counts()
                    except Exception:
                        try:
                            hw_counts = pub_result.join_data().get_counts()
                        except Exception:
                            hw_counts = {}

                    if not hw_counts:
                        print(f"⚠️ No counts extracted for hardware run '{circuit_key}'; skipping.")
                        continue

                    z_hardware = counts_to_z_expectation(hw_counts, n_qubits)

                    # per-shot bitstrings if available
                    try:
                        hw_bitstrings = pub_result.data.meas.get_strings()
                    except Exception:
                        hw_bitstrings = []
                        if isinstance(hw_counts, dict):
                            for b, c in hw_counts.items():
                                hw_bitstrings.extend([b] * c)

                    # write hardware shot-level file
                    hw_payload = {
                        "z": z_hardware,
                        "counts": hw_counts,
                        "bitstrings": hw_bitstrings,
                    }
                    with open(shot_dir / f"{safe_key}__hardware_shots.json", "w") as f_out:
                        json.dump(hw_payload, f_out, indent=2)

                    # summary
                    results_dict[circuit_key] = {
                        "z_ideal": z_ideal,
                        "z_hardware": z_hardware,
                        "hardware_job_id": hardware_job_id,
                    }

                except RuntimeJobMaxTimeoutError as e:
                    print(f"⏱️ Hardware timeout for '{circuit_key}' (job {hardware_job_id}): {e}")
                except (RuntimeJobFailureError, RuntimeInvalidStateError) as e:
                    print(f"❌ Hardware job error for '{circuit_key}' (job {hardware_job_id}): {e}")
                except Exception as e:
                    print(f"❌ Unexpected retrieval error for '{circuit_key}' (job {hardware_job_id}): {e}")

    # === Phase 3: write summary JSON ===
    try:
        with open(output_json_path, "w") as out_f:
            json.dump(results_dict, out_f, indent=2)
        print(f"\n✅ Aggregated summary saved to {output_json_path}")
        print(f"📁 Shot-level details under {shot_dir}")
    except Exception as e:
        print(f"❌ Failed to write summary JSON: {e}")


simulate_and_store_z_expectations_real_qpu_json(
    qpy_folder="/home/macula/SMATousi/projects/quantum/andrew/ExecutionResults/StoredCircuits/",
    output_json_path="real_z_expectations.json",
    shots=100,
    per_circuit_timeout_s=14400
    # service=service
)
