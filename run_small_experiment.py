# ---------------------------------------------
# Quick runner for small cohort simulations & metric evaluation
# ---------------------------------------------

import argparse
import time
import shutil
from pathlib import Path
import multiprocessing

from OHSUPOPsimulations import SIM_options, SIM_load_virtualpop_data
from datasetPop import simulate_scenarios
from calc_metrics import evaluate_simulation_files, print_comparison_table


def run_experiment(subject_ids=None, n_subjects=3, delays=None, n_meal_scenarios=3,
                   n_init_cgm=5, output_dir=None, parallel=False, clean=True):
    if output_dir is None:
        output_dir = SIM_options.SIMULATION_RESULTS_DIR / 'small_experiment'
    output_dir = Path(output_dir)

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if delays is None:
        delays = [0]  # bolus at meal onset

    # Configure options
    SIM_options.SIMULATION_N_MEAL_SCENARIOS = n_meal_scenarios
    SIM_options.SIMULATION_N_CGM_INITIALIZATION = n_init_cgm

    vp_data = SIM_load_virtualpop_data.load_virtual_subjects(str(SIM_options.SIMULATION_VP_PKL))

    # Select target subjects (normalized to 6 digits with leading zeros)
    if subject_ids is not None:
        target_ids = [str(s).strip().zfill(6) for s in subject_ids]
    else:
        target_ids = SIM_options.SIMULATION_SUBJECTS_INCLUDED[:n_subjects]

    print("=" * 65)
    print("STARTING EXPERIMENT (3 REPRESENTATIVE PATIENTS)")
    print(f"Target Subjects: {target_ids}")
    print(f"Bolus Delays   : {delays} min")
    print("=" * 65)

    TDIR = SIM_options.SIMULATION_TDIR
    basalTDIRRatio = SIM_options.SIMULATION_BASAL_TDIR_RATIO
    modParamsOUT = SIM_options.SIMULATION_POPULATION_PARAMS[:15]
    body_weight = SIM_options.SIMULATION_POPULATION_PARAMS[-1]
    cf = 35.0

    t_start = time.time()
    generated_files = []

    for late_bolus_min in delays:
        subfolder = output_dir / f"boluslate_{late_bolus_min}min"
        subfolder.mkdir(parents=True, exist_ok=True)

        items = []
        for sub_idx in range(len(vp_data['subject'])):
            sid = str(vp_data['subject'][sub_idx]['subjectID']).zfill(6)
            if sid in target_ids:
                items.append([
                    sub_idx, sid, vp_data, modParamsOUT, body_weight,
                    TDIR, basalTDIRRatio, str(subfolder), late_bolus_min, cf
                ])

        if parallel and len(items) > 1:
            num_proc = min(multiprocessing.cpu_count(), len(items))
            with multiprocessing.Pool(processes=num_proc) as pool:
                pool.map(simulate_scenarios, items)
        else:
            for item in items:
                simulate_scenarios(item)

        new_files = list(subfolder.glob("*.pkl"))
        generated_files.extend(new_files)

    elapsed = time.time() - t_start
    print(f"Simulation completed in {elapsed:.1f} s ({len(generated_files)} files).")

    if not generated_files:
        print("ERROR: No files generated.")
        return

    windows_agg, subject_agg = evaluate_simulation_files(generated_files, target_col='output_glucose')
    print_comparison_table(windows_agg, subject_agg, len(generated_files))


def main():
    parser = argparse.ArgumentParser(description="Run simulation on a small cohort and compare with paper.")
    parser.add_argument('--n_subjects', type=int, default=3, help="Number of subjects (default: 3)")
    parser.add_argument('--subjects', type=str, default='001033,001034,037008', help="Comma-separated subject IDs")
    parser.add_argument('--delays', type=str, default='0', help="Comma-separated delays in min")
    parser.add_argument('--n_meals', type=int, default=3, help="Meal scenarios per subject (default: 3)")
    parser.add_argument('--n_cgm', type=int, default=5, help="Init CGM samples per scenario (default: 5)")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory")
    parser.add_argument('--parallel', action='store_true', help="Use parallel CPU execution")
    parser.add_argument('--no-clean', dest='clean', action='store_false', help="Do not clean output dir")
    args = parser.parse_args()

    s_ids = [x.strip() for x in args.subjects.split(',')] if args.subjects else None
    d_list = [int(x.strip()) for x in args.delays.split(',')]

    run_experiment(
        subject_ids=s_ids,
        n_subjects=args.n_subjects,
        delays=d_list,
        n_meal_scenarios=args.n_meals,
        n_init_cgm=args.n_cgm,
        output_dir=args.out_dir,
        parallel=args.parallel,
        clean=args.clean
    )


if __name__ == '__main__':
    main()
