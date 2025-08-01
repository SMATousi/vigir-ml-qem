#!/usr/bin/env python3
"""
Script to add expected Z values to JSON files containing quantum measurement results.

This script iterates through all JSON files in the extracted_data directory,
calculates the expected Z value for each qubit based on the measurement counts,
and adds this information to each JSON file.

Expected Z value for qubit i is calculated as:
<Z_i> = (count_0 - count_1) / total_shots
where count_0 is the number of measurements with qubit i in state |0⟩
and count_1 is the number of measurements with qubit i in state |1⟩
"""

import json
import os
import glob
from typing import Dict, List, Tuple

def calculate_expected_z_values(counts: Dict[str, int]) -> List[float]:
    """
    Calculate expected Z values for each qubit from measurement counts.
    
    Args:
        counts: Dictionary mapping bit strings to their counts
        
    Returns:
        List of expected Z values, one for each qubit position
    """
    if not counts:
        return []
    
    # Get the number of qubits from the first bit string
    first_bitstring = next(iter(counts.keys()))
    num_qubits = len(first_bitstring)
    
    # Initialize counters for each qubit
    qubit_0_counts = [0] * num_qubits
    qubit_1_counts = [0] * num_qubits
    total_shots = sum(counts.values())
    
    # Count 0s and 1s for each qubit position
    for bitstring, count in counts.items():
        for i, bit in enumerate(bitstring):
            if bit == '0':
                qubit_0_counts[i] += count
            else:
                qubit_1_counts[i] += count
    
    # Calculate expected Z values: <Z> = (count_0 - count_1) / total_shots
    expected_z_values = []
    for i in range(num_qubits):
        expected_z = (qubit_0_counts[i] - qubit_1_counts[i]) / total_shots
        expected_z_values.append(expected_z)
    
    return expected_z_values

def process_json_file(filepath: str) -> bool:
    """
    Process a single JSON file to add expected Z values.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if expected_z already exists
        if 'expected_z' in data:
            print(f"Skipping {os.path.basename(filepath)}: expected_z already exists")
            return True
        
        # Calculate expected Z values
        if 'counts' in data:
            expected_z_values = calculate_expected_z_values(data['counts'])
            data['expected_z'] = expected_z_values
            
            # Write back to the file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Processed {os.path.basename(filepath)}: Added expected_z for {len(expected_z_values)} qubits")
            return True
        else:
            print(f"Warning: {os.path.basename(filepath)} does not contain 'counts' field")
            return False
            
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return False

def main():
    """
    Main function to process all JSON files in the extracted_data directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_data_dir = os.path.join(script_dir, 'extracted_data')
    
    if not os.path.exists(extracted_data_dir):
        print(f"Error: Directory {extracted_data_dir} does not exist")
        return
    
    # Find all JSON files
    json_pattern = os.path.join(extracted_data_dir, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {extracted_data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        if process_json_file(json_file):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Total files: {len(json_files)}")

if __name__ == "__main__":
    main()
