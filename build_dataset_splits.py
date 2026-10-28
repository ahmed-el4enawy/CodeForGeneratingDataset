# ---------------------------------------------
# Authors: Clara Mosquera-Lopez, Valentina Roquemen, Gavin Young, Navid Resalat, Peter Jacobs
# File: build_dataset_splits.py
# Splits dataset into 60/20/20 train/val/test subsets without daily meal scenario leakage
# and flags rare events (TBR > 20% or glucose > 250 mg/dL > 40%) per paper Sec 2.2.1.
# ---------------------------------------------

import sys
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from OHSUPOPsimulations import SIM_options


def find_simulation_files(root_dir):
    """Recursively discover all simulation pickle files."""
    root_path = Path(root_dir)
    return sorted(list(root_path.rglob("*.pkl")))


def analyze_simulation_file(file_path):
    """
    Extract metadata, meal hashes, and rare-event flags from a simulation file.
    """
    try:
        df = pd.read_pickle(str(file_path))
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return None

    # Target glucose for models is output_glucose (= Q1 in mg/dL)
    glucose = df['output_glucose'].values
    tbr_pct = float(100.0 * np.mean(glucose < 70.0))
    tar250_pct = float(100.0 * np.mean(glucose > 250.0))
    is_rare_event = bool((tbr_pct > 20.0) or (tar250_pct > 40.0))

    subject_id = str(df['subject_id'].iloc[0]) if 'subject_id' in df else file_path.stem.split('_')[0]
    meal_scenario = int(df['meal_scenario'].iloc[0]) if 'meal_scenario' in df else -1
    late_bolus_min = int(df['late_bolus_min'].iloc[0]) if 'late_bolus_min' in df else 0
    init_cgm = float(df['initCGM'].iloc[0]) if 'initCGM' in df else 0.0

    if 'meal_day_ids' in df:
        meal_day_ids = sorted(list(set(df['meal_day_ids'].dropna().tolist())))
    else:
        # Fallback to subject-scenario identifier if meal_day_ids not present
        meal_day_ids = [f"{subject_id}_mealScen_{meal_scenario}"]

    n_dosing = df['dosing_problem_id'].nunique() if 'dosing_problem_id' in df else 11
    sim_days = df['sim_day'].nunique() if 'sim_day' in df else 7
    total_days = n_dosing * sim_days

    return {
        'file_path': str(file_path.resolve()),
        'relative_path': str(file_path),
        'file_name': file_path.name,
        'subject_id': subject_id,
        'meal_scenario': meal_scenario,
        'late_bolus_min': late_bolus_min,
        'initCGM': init_cgm,
        'total_samples': len(df),
        'total_sim_days': total_days,
        'tbr_pct': round(tbr_pct, 2),
        'tar250_pct': round(tar250_pct, 2),
        'rare_event': is_rare_event,
        'meal_day_ids': meal_day_ids
    }


def partition_clusters(file_records, split_ratios=(0.60, 0.20, 0.20), seed=42):
    """
    Cluster simulation files sharing meal days and distribute clusters into
    train (60%), validation (20%), and test (20%) sets without data leakage.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build adjacency between meal_day_id and file records
    meal_to_files = defaultdict(list)
    file_to_meals = {}

    for idx, rec in enumerate(file_records):
        m_ids = rec['meal_day_ids']
        file_to_meals[idx] = m_ids
        for m in m_ids:
            meal_to_files[m].append(idx)

    # Graph connected components: two files in same cluster if they share any meal_day_id
    visited_files = set()
    clusters = []

    for idx in range(len(file_records)):
        if idx in visited_files:
            continue

        cluster_file_indices = set()
        queue = [idx]
        cluster_file_indices.add(idx)
        visited_files.add(idx)

        while queue:
            curr_file = queue.pop()
            for m in file_to_meals[curr_file]:
                for neighbor_file in meal_to_files[m]:
                    if neighbor_file not in visited_files:
                        visited_files.add(neighbor_file)
                        cluster_file_indices.add(neighbor_file)
                        queue.append(neighbor_file)

        clusters.append(list(cluster_file_indices))

    # Shuffle clusters randomly
    random.shuffle(clusters)

    # Target fractions
    total_files = len(file_records)
    target_train = split_ratios[0] * total_files
    target_val = split_ratios[1] * total_files
    target_test = split_ratios[2] * total_files

    train_indices = []
    val_indices = []
    test_indices = []

    # Sort clusters descending by size to place largest components first (best fit)
    clusters.sort(key=lambda c: len(c), reverse=True)

    for cluster in clusters:
        c_len = len(cluster)
        rem_train = target_train - len(train_indices)
        rem_val = target_val - len(val_indices)
        rem_test = target_test - len(test_indices)

        # Assign to split with greatest remaining demand
        demands = [
            (rem_train, 0),
            (rem_val, 1),
            (rem_test, 2)
        ]
        demands.sort(key=lambda x: x[0], reverse=True)
        chosen_split = demands[0][1]

        if chosen_split == 0:
            train_indices.extend(cluster)
        elif chosen_split == 1:
            val_indices.extend(cluster)
        else:
            test_indices.extend(cluster)

    train_records = [file_records[i] for i in train_indices]
    val_records = [file_records[i] for i in val_indices]
    test_records = [file_records[i] for i in test_indices]

    # Verify 0% meal-day leakage
    train_meals = set().union(*(r['meal_day_ids'] for r in train_records)) if train_records else set()
    val_meals = set().union(*(r['meal_day_ids'] for r in val_records)) if val_records else set()
    test_meals = set().union(*(r['meal_day_ids'] for r in test_records)) if test_records else set()

    leakage_train_val = train_meals.intersection(val_meals)
    leakage_train_test = train_meals.intersection(test_meals)
    leakage_val_test = val_meals.intersection(test_meals)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError("FATAL: Meal day leakage detected across subsets!")

    return train_records, val_records, test_records


def main():
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = SIM_options.SIMULATION_RESULTS_DIR / SIM_options.SIMULATION_ID / 'POPULATION'

    print(f"Scanning for simulation files in: {target_dir}")
    pkl_files = find_simulation_files(target_dir)

    if len(pkl_files) == 0:
        print(f"No pickle simulation files found in {target_dir}.")
        print("Run 'python datasetPop.py' first to generate simulation data.")
        return

    print(f"Found {len(pkl_files)} simulation files. Analyzing metadata and rare-event rates...")
    records = []
    for f in pkl_files:
        rec = analyze_simulation_file(f)
        if rec is not None:
            records.append(rec)

    print(f"Successfully analyzed {len(records)} simulation files.")

    train_recs, val_recs, test_recs = partition_clusters(records, split_ratios=(0.60, 0.20, 0.20))

    total_files = len(records)
    train_pct = 100.0 * len(train_recs) / total_files if total_files else 0
    val_pct = 100.0 * len(val_recs) / total_files if total_files else 0
    test_pct = 100.0 * len(test_recs) / total_files if total_files else 0

    total_days = sum(r['total_sim_days'] for r in records)
    train_days = sum(r['total_sim_days'] for r in train_recs)
    val_days = sum(r['total_sim_days'] for r in val_recs)
    test_days = sum(r['total_sim_days'] for r in test_recs)

    rare_total = sum(1 for r in records if r['rare_event'])
    rare_train = sum(1 for r in train_recs if r['rare_event'])
    rare_val = sum(1 for r in val_recs if r['rare_event'])
    rare_test = sum(1 for r in test_recs if r['rare_event'])

    manifest = {
        'summary': {
            'total_files': total_files,
            'total_sim_days': total_days,
            'splits_file_counts': {
                'train': len(train_recs),
                'validation': len(val_recs),
                'test': len(test_recs)
            },
            'splits_file_percentages': {
                'train': round(train_pct, 1),
                'validation': round(val_pct, 1),
                'test': round(test_pct, 1)
            },
            'splits_sim_days': {
                'train': train_days,
                'validation': val_days,
                'test': test_days
            },
            'rare_events': {
                'total': rare_total,
                'train': rare_train,
                'validation': rare_val,
                'test': rare_test,
                'criteria': 'TBR > 20% or time > 250 mg/dL > 40%'
            },
            'leakage_verified_free': True
        },
        'splits': {
            'train': train_recs,
            'validation': val_recs,
            'test': test_recs
        }
    }

    manifest_path = target_dir / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Also save file lists for easy training consumption
    for split_name, recs in [('train', train_recs), ('validation', val_recs), ('test', test_recs)]:
        list_file = target_dir / f"{split_name}_files.txt"
        with open(list_file, 'w') as f:
            for r in recs:
                f.write(r['file_path'] + '\n')

    print("\n" + "=" * 50)
    print("DATASET SPLITS SUMMARY (Zero Leakage Verified)")
    print("=" * 50)
    print(f"Total Simulation Files : {total_files}")
    print(f"Total Simulated Days   : {total_days:,}")
    print(f"Train Split (60%)      : {len(train_recs)} files ({train_pct:.1f}%), {train_days:,} days")
    print(f"Validation Split (20%) : {len(val_recs)} files ({val_pct:.1f}%), {val_days:,} days")
    print(f"Test Split (20%)       : {len(test_recs)} files ({test_pct:.1f}%), {test_days:,} days")
    print(f"Rare-Event Traces      : {rare_total} total ({rare_train} train, {rare_val} val, {rare_test} test)")
    print(f"Manifest written to    : {manifest_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
