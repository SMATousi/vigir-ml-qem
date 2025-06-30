import numpy as np
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator, noise
from qiskit.test.mock import FakeLima
from qiskit.quantum_info import Statevector, Pauli



def generate_random_circuits(num_qubits, num_circuits, circuit_depth):
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Params
    # ──────────────────────────────────────────────────────────────────────────────
    num_qubits    = num_qubits
    num_circuits  = num_circuits
    circuit_depth = circuit_depth
    master_seed   = 1234
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Build your noise model
    # ──────────────────────────────────────────────────────────────────────────────
    backend      = FakeLima()
    noise_model  = noise.NoiseModel.from_backend(backend)
    basis_gates  = noise_model.basis_gates
    coupling_map = backend.configuration().coupling_map
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Define simulators
    # ──────────────────────────────────────────────────────────────────────────────
    sim_ideal = AerSimulator(method='statevector',
                            seed_transpiler=master_seed,
                            seed_simulator=master_seed)
    
    sim_noise = AerSimulator(
        method='density_matrix',
        noise_model=noise_model,
        # you can omit basis_gates/coupling_map here if it keeps causing trouble
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        seed_transpiler=master_seed,
        seed_simulator=master_seed
    )
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Prepare the observable Z⊗5
    # ──────────────────────────────────────────────────────────────────────────────
    z_paulis = [
        Pauli('I'*i + 'Z' + 'I'*(num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    mz_matrices = [p.to_matrix() for p in z_paulis]
    
    
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Generate raw random circuits (no save_* calls yet)
    # ──────────────────────────────────────────────────────────────────────────────
    raw_circuits = []
    for i in range(num_circuits):
        # seed each circuit differently but predictably
        seed_i = master_seed + i
        qc = random_circuit(
            num_qubits,
            circuit_depth,
            measure=False,
            seed=seed_i
        )
        raw_circuits.append(qc)
    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Build the IDEAL set (with save_statevector)
    # ──────────────────────────────────────────────────────────────────────────────
    ideal_circuits = []
    for qc in raw_circuits:
        qc_i = qc.copy()
        qc_i.save_statevector()       # instruct Aer to record the final statevector
        ideal_circuits.append(qc_i)
    
    # Transpile and run
    tcirc_ideal = transpile(ideal_circuits, sim_ideal)
    res_ideal   = sim_ideal.run(tcirc_ideal).result()
    
    # Extract statevectors and compute ⟨Z⟩ on each qubit
    statevecs = [res_ideal.get_statevector(i) for i in range(num_circuits)]
    expect_ideal_per_qubit = [
        [Statevector(sv).expectation_value(p).real for p in z_paulis]
        for sv in statevecs
    ]
    
    
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Build the NOISY set (now with save_density_matrix)
    # ──────────────────────────────────────────────────────────────────────────────
    noise_circuits = []
    for qc in raw_circuits:
        qc_n = qc.copy()
        qc_n.save_density_matrix(label='rho')   # record the final density matrix
        noise_circuits.append(qc_n)
    
    # Transpile and run
    tcirc_noise = transpile(noise_circuits, sim_noise)
    res_noise   = sim_noise.run(tcirc_noise).result()
    
    # Extract density matrices and compute ⟨Z⟩ on each qubit
    dens_mats = [res_noise.data(i)['rho'] for i in range(num_circuits)]
    expect_noise_per_qubit = [
        [np.real(np.trace(rho @ mz)) for mz in mz_matrices]
        for rho in dens_mats
    ]
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Compare
    # ──────────────────────────────────────────────────────────────────────────────
    # for i in range(num_circuits):
    #     print(f"Circuit {i:2d}:")
    #     print(f"  Ideal per-qubit Z: {expect_ideal_per_qubit[i]}")
    #     print(f"  Noisy per-qubit Z: {expect_noise_per_qubit[i]}")
    #     print()

    return raw_circuits, expect_ideal_per_qubit, expect_noise_per_qubit


