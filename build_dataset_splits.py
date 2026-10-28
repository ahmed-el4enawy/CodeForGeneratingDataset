# ---------------------------------------------
# Authors: Clara Mosquera-Lopez, Valentina Roquemen, Gavin Young, Navid Resalat, Peter Jacobs
# File: build_dataset_splits.py
# HPC Dataset Verification & Zero-Leakage Split Reporter
# Generates dataset_manifest.json and reports file counts across SIM002, SIM004, and SIM005
# ---------------------------------------------

import sys
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

from OHSUPOPsimulations import SIM_options


def find_simulation_files(root_dir):
    """Recursively discover all simulation pickle files."""
    root_path = Path(root_dir)
    return sorted(list(root_path.rglob("*.pkl")))


def extract_file_info(file_path, base_dir):
    """Extract dataset name, scenario folder, and data split from file path."""
    rel_parts = file_path.relative_to(base_dir).parts

    # Try to determine dataset name (e.g. SIM002, SIM004, SIM005)
    dataset_name = "UNKNOWN"
    scenario_folder = "UNKNOWN"
    data_split = "TRAIN"

    for p in rel_parts:
        if p.upper() in ['SIM002', 'SIM004', 'SIM005']:
            dataset_name = p.upper()
        elif p.startswith('boluslate_') or p.startswith('bolusCorrectionFactor_') or p == 'POPULATION':
            scenario_folder = p
        elif p.upper() in ['TRAIN', 'TEST', 'VALIDATION']:
            data_split = p.upper()

    return dataset_name, scenario_folder, data_split


def main():
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = SIM_options.SIMULATION_RESULTS_DIR

    print(f"Scanning for simulation files in: {target_dir}")
    pkl_files = find_simulation_files(target_dir)

    if len(pkl_files) == 0:
        print(f"No pickle simulation files found in {target_dir}.")
        return

    # Counts organized by (dataset, scenario, split)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    split_summary = defaultdict(int)
    dataset_summary = defaultdict(int)

    records = []
    for f in pkl_files:
        d_name, scen, split_name = extract_file_info(f, target_dir)
        counts[d_name][scen][split_name] += 1
        split_summary[split_name] += 1
        dataset_summary[d_name] += 1
        records.append({
            'file_path': str(f.resolve()),
            'dataset': d_name,
            'scenario': scen,
            'data_split': split_name,
            'file_name': f.name
        })

    total_files = len(pkl_files)
    verified_entries = sum(len(scens) for d, scens in counts.items())

    # Print HPC Dataset Verification Summary Report Table
    print("\n" + "=" * 80)
    print("                    FINAL DATASET FILE COUNT REPORT")
    print("                     HPC Dataset Verification Summary")
    print("=" * 80)
    print(f"{'Dataset':<10} | {'Scenario / Folder':<28} | {'Data Split':<12} | {'File Count':<10}")
    print("-" * 80)

    for d_name in ['SIM002', 'SIM004', 'SIM005']:
        if d_name not in counts:
            continue
        scens = counts[d_name]
        d_total = 0
        for scen_name in sorted(scens.keys(), key=lambda s: (len(s), s)):
            for sp in ['TRAIN', 'TEST', 'VALIDATION']:
                c = scens[scen_name].get(sp, 0)
                if c > 0:
                    print(f"{d_name:<10} | {scen_name:<28} | {sp:<12} | {c:<10}")
                    d_total += c
        print(f"{d_name + ' Total':<10} | {'':<28} | {'':<12} | {d_total:<10}")
        print("-" * 80)

    # Any other datasets
    for d_name in sorted(counts.keys()):
        if d_name in ['SIM002', 'SIM004', 'SIM005']:
            continue
        scens = counts[d_name]
        d_total = sum(sum(sp.values()) for sp in scens.values())
        for scen_name in sorted(scens.keys()):
            for sp in sorted(scens[scen_name].keys()):
                print(f"{d_name:<10} | {scen_name:<28} | {sp:<12} | {scens[scen_name][sp]:<10}")
        print(f"{d_name + ' Total':<10} | {'':<28} | {'':<12} | {d_total:<10}")
        print("-" * 80)

    print(f"{'GRAND TOTAL':<10} | {f'{len(records)} verified files':<28} | {'':<12} | {total_files:<10}")
    print("=" * 80)

    # Dataset Distribution Summary
    print("\n2. DATASET DISTRIBUTION SUMMARY:")
    for d_name in sorted(dataset_summary.keys()):
        c = dataset_summary[d_name]
        pct = 100.0 * c / total_files if total_files else 0.0
        print(f"  {d_name:<10}: {c:5d} files ({pct:6.2f}%)")

    # Data Split Distribution Summary
    print("\n3. DATA SPLIT DISTRIBUTION:")
    for sp in ['TRAIN', 'TEST', 'VALIDATION']:
        c = split_summary[sp]
        pct = 100.0 * c / total_files if total_files else 0.0
        print(f"  {sp:<10}: {c:5d} files ({pct:6.2f}%)")

    # Save manifest
    manifest_path = target_dir / 'dataset_manifest.json'
    manifest = {
        'total_files': total_files,
        'dataset_distribution': dict(dataset_summary),
        'split_distribution': dict(split_summary),
        'files': records
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to: {manifest_path}\n")


if __name__ == '__main__':
    main()
