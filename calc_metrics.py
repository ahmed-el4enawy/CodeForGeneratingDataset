# ---------------------------------------------
# Clinical metrics calculation & paper benchmark comparison
# Reference: Roquemen-Echeverri et al. (Sec 2.2.1, 2.6.2, Table 3, Table 6, Page 12)
# ---------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Paper benchmark values for the ODE simulator (Roquemen-Echeverri et al., Sec. 3, Page 12)
PAPER_ODE_BENCHMARK = {
    'TIR':  '72.9 ± 35.2 %',
    'TAR':  '23.1 ± 36.3 %',
    'TBR':  '4.0 ± 8.2 %',
    'LBGI': '1.5 ± 1.7',
    'HBGI': '4.2 ± 6.2',
    'MG':   '132.6 ± 48.2 mg/dL'
}


def compute_blood_glucose_indices(glucose_vals):
    """
    Compute LBGI and HBGI using Kovatchev's standard formula.
    f(g) = 1.509 * (ln(g)^1.084 - 5.381)
    r(g) = 10 * f(g)^2
    rl(g) = r(g) if f(g) < 0 else 0
    rh(g) = r(g) if f(g) > 0 else 0
    """
    g = np.asarray(glucose_vals, dtype=float)
    valid_mask = g > 10.0
    g = g[valid_mask]
    if len(g) == 0:
        return 0.0, 0.0

    log_g = np.log(np.maximum(g, 1e-3))
    f = 1.509 * (np.power(log_g, 1.084) - 5.381)
    risk = 10.0 * (f ** 2)

    rl = np.where(f < 0.0, risk, 0.0)
    rh = np.where(f > 0.0, risk, 0.0)

    lbgi = float(np.mean(rl))
    hbgi = float(np.mean(rh))
    return lbgi, hbgi


def compute_metrics_for_array(glucose_arr):
    g = np.asarray(glucose_arr, dtype=float)
    valid = g[~np.isnan(g)]
    if len(valid) == 0:
        return {'TIR': 0.0, 'TAR': 0.0, 'TBR': 0.0, 'LBGI': 0.0, 'HBGI': 0.0, 'MG': 0.0}

    tir = float(100.0 * np.mean((valid >= 70.0) & (valid <= 180.0)))
    tar = float(100.0 * np.mean(valid > 180.0))
    tbr = float(100.0 * np.mean(valid < 70.0))
    mg = float(np.mean(valid))
    lbgi, hbgi = compute_blood_glucose_indices(valid)

    return {
        'TIR': tir,
        'TAR': tar,
        'TBR': tbr,
        'LBGI': lbgi,
        'HBGI': hbgi,
        'MG': mg
    }


def compute_sequence_metrics(glucose_arr, window_len=60):
    n_windows = len(glucose_arr) // window_len
    if n_windows == 0:
        return [compute_metrics_for_array(glucose_arr)]

    seq_metrics = []
    for i in range(n_windows):
        window = glucose_arr[i * window_len:(i + 1) * window_len]
        seq_metrics.append(compute_metrics_for_array(window))
    return seq_metrics


def evaluate_simulation_files(files_list, target_col='output_glucose'):
    all_5h_windows = []
    subject_summaries = {}

    for fpath in files_list:
        df = pd.read_pickle(str(fpath))
        sub_id = str(df['subject_id'].iloc[0]) if 'subject_id' in df else Path(fpath).stem.split('_')[0]

        if sub_id not in subject_summaries:
            subject_summaries[sub_id] = []

        for prob_id, prob_df in df.groupby('dosing_problem_id'):
            glucose = prob_df[target_col].values
            w_metrics = compute_sequence_metrics(glucose, window_len=60)
            all_5h_windows.extend(w_metrics)
            subject_summaries[sub_id].extend(w_metrics)

    def aggregate(metrics_list):
        if not metrics_list:
            return {}
        res = {}
        for k in ['TIR', 'TAR', 'TBR', 'LBGI', 'HBGI', 'MG']:
            vals = [m[k] for m in metrics_list]
            res[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals))
            }
        return res

    windows_agg = aggregate(all_5h_windows)
    subject_agg = {sid: aggregate(m_list) for sid, m_list in subject_summaries.items()}

    return windows_agg, subject_agg


def print_comparison_table(windows_agg, subject_agg, n_files):
    print("\n" + "=" * 65)
    print("      SIMULATION RESULTS vs. PAPER (Roquemen-Echeverri et al.)")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Your Simulation':<18} | {'Paper (T1DSim_ODE)'}")
    print("-" * 65)

    metric_names = {
        'TIR':  ('Time in Range (70-180 mg/dL)', '%'),
        'TAR':  ('Time Above Range (>180 mg/dL)', '%'),
        'TBR':  ('Time Below Range (<70 mg/dL)', '%'),
        'MG':   ('Mean Glucose', 'mg/dL'),
        'HBGI': ('High BG Index (HBGI)', ''),
        'LBGI': ('Low BG Index (LBGI)', '')
    }

    for k in ['TIR', 'TAR', 'TBR', 'MG', 'HBGI', 'LBGI']:
        label, u = metric_names[k]
        val_str = f"{windows_agg[k]['mean']:.1f} ± {windows_agg[k]['std']:.1f} {u}".strip()
        paper_str = PAPER_ODE_BENCHMARK[k]
        print(f"{label:<25} | {val_str:<18} | {paper_str}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate simulation metrics and compare with paper.")
    parser.add_argument('path', nargs='?', default=None, help="Directory containing .pkl simulation files")
    parser.add_argument('--target', default='output_glucose', choices=['output_glucose', 'output_cgm'],
                        help="Target glucose signal")
    args = parser.parse_args()

    if args.path:
        search_dir = Path(args.path)
    else:
        from OHSUPOPsimulations import SIM_options
        search_dir = SIM_options.SIMULATION_RESULTS_DIR / 'small_experiment'

    files = list(search_dir.rglob("*.pkl"))
    if not files:
        print(f"No .pkl simulation files found in: {search_dir}")
        return

    windows_agg, subject_agg = evaluate_simulation_files(files, target_col=args.target)
    print_comparison_table(windows_agg, subject_agg, len(files))


if __name__ == '__main__':
    main()
