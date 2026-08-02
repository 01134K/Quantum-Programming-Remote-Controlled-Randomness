import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile

def run_remote_randomness_simulation(shots=100000, phase_degrees=45.0):
    # 1. Create directory for images if it doesn't exist
    os.makedirs('img_results', exist_ok=True)
    
    # 2. Initialize Quantum Circuit with 2 Qubits and 2 Classical Bits
    # q[0] = Qubit a (Control)
    # q[1] = Qubit b (Target)
    q = QuantumRegister(2, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)

    # 3. Apply Gates on Qubit a
    qc.h(q[0])

    # 4. Apply Gates on Qubit b (H -> Phase(theta) -> H)
    qc.h(q[1])
    phase_radians = np.deg2rad(phase_degrees)
    qc.p(phase_radians, q[1])  # Phase shift in radians
    qc.h(q[1])

    # 5. Apply CNOT (Control: a -> Target: b)
    qc.cx(q[0], q[1])

    # 6. Measure Qubits
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # 7. Draw and save circuit diagram
    print("Saving circuit diagram to img_results/circuit.png...")
    qc.draw('mpl', filename='img_results/circuit.png')

    # 8. Run simulation
    print(f"Running simulation with {shots:,} shots at phase {phase_degrees}°...")
    simulator = AerSimulator()
    qc_transpiled = transpile(qc, simulator)
    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts()
    
    # 9. Analyze conditional probabilities
    # Raw counts keys are binary strings 'c[1]c[0]' representing 'b a'
    a_0_total = 0
    a_0_b_0 = 0
    a_0_b_1 = 0
    
    a_1_total = 0
    a_1_b_0 = 0
    a_1_b_1 = 0
    
    for outcome, count in counts.items():
        b_val = outcome[0]
        a_val = outcome[1]
        
        if a_val == '0':
            a_0_total += count
            if b_val == '0':
                a_0_b_0 += count
            else:
                a_0_b_1 += count
        else:
            a_1_total += count
            if b_val == '0':
                a_1_b_0 += count
            else:
                a_1_b_1 += count

    # Calculate percentages
    pct_b0_a0 = (a_0_b_0 / a_0_total) * 100 if a_0_total > 0 else 0
    pct_b1_a0 = (a_0_b_1 / a_0_total) * 100 if a_0_total > 0 else 0
    pct_b0_a1 = (a_1_b_0 / a_1_total) * 100 if a_1_total > 0 else 0
    pct_b1_a1 = (a_1_b_1 / a_1_total) * 100 if a_1_total > 0 else 0

    print("\n================== Results Analysis ==================")
    print(f"Raw Counts: {counts}")
    print(f"\nCase a = 0 (Total: {a_0_total:,} shots):")
    print(f"  - b = 0: {a_0_b_0:,} times ({pct_b0_a0:.2f}%)")
    print(f"  - b = 1: {a_0_b_1:,} times ({pct_b1_a0:.2f}%)")
    print(f"\nCase a = 1 (Total: {a_1_total:,} shots):")
    print(f"  - b = 0: {a_1_b_0:,} times ({pct_b0_a1:.2f}%)")
    print(f"  - b = 1: {a_1_b_1:,} times ({pct_b1_a1:.2f}%)")
    print("======================================================\n")

    # 10. Plot and save results histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ['b=0 given a=0', 'b=1 given a=0', 'b=0 given a=1', 'b=1 given a=1']
    percentages = [pct_b0_a0, pct_b1_a0, pct_b0_a1, pct_b1_a1]
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    
    bars = ax.bar(scenarios, percentages, color=colors, width=0.5)
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'Conditional Probabilities of b given a\n(Total Shots: {shots:,} | Phase: {phase_degrees}°)')
    ax.set_ylim(0, 110)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2, f"{height:.2f}%", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(f'img_results/histogram_{shots}_shots.png')
    print(f"Saving statistics histogram to img_results/histogram_{shots}_shots.png...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remote-Controlled Randomness Quantum Simulator")
    parser.add_argument('--shots', type=int, default=100000, help="Number of simulation runs (default: 100000)")
    parser.add_argument('--phase', type=float, default=45.0, help="Phase angle of qubit b in degrees (default: 45.0)")
    args = parser.parse_args()
    
    run_remote_randomness_simulation(shots=args.shots, phase_degrees=args.phase)
