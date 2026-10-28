# ---------------------------------------------
# Authors: Clara Mosquera-Lopez, Valentina Roquemen, Gavin Young, Navid Resalat, Peter Jacobs
#          Oregon Health & Science University
# File: datasetPop.py
# Glucoregulatory Population Simulator - Exact HPC Dataset (SIM002, SIM004, SIM005)
# ---------------------------------------------

import argparse
import hashlib
import multiprocessing
from pathlib import Path
import random
import numpy as np
import pandas as pd

# Original random seeds
random.seed(0)
np.random.seed(0)

# OHSU simulator imports
import OHSUPOPsimulations.create_SHP as createSH
import OHSUPOPsimulations.test_SHP as testSH

# SIM imports
from OHSUPOPsimulations import SIM_options, SIM_generate_dosing_problems, SIM_load_virtualpop_data
from correction_bolus import Correction_Ins_Alg


def OHSU_SIM_simulate_subject(simDays, initCGM, modParams0, bodyWeight, TDIR, basalTDIRRatio, mealScenario,
                              ICR_problem, BASAL_problem, CF_problem, sub_idx, late_bolus_min, CF, problem,
                              subject_id, meal_scenario_id, meal_day_ids_list, split_name,
                              correction_bolus_enabled=False,
                              age=35, sex='UNK', initStates=None, Ts=5):
    """
    Simulate a single subject under a given scenario using pre-allocated NumPy arrays.
    """
    modParams = np.append(modParams0, bodyWeight)  # 16-element array
    simTime = int(simDays * 1440 / Ts)            # Total samples (2016 for 7 days)

    mealVector = mealScenario.copy()
    IpVector = np.ones(len(mealVector))

    # Calculate prandial insulin boluses
    ICR = 1700.0 / TDIR / 3.0                     # ICR: Insulin to Carb Ratio
    mealBoluses = 1000.0 * IpVector * (mealVector / (ICR * ICR_problem)) / (bodyWeight * Ts)

    # Delay meal bolus and truncate back to simTime
    if late_bolus_min > 0:
        delay_steps = late_bolus_min // 5
        mealBoluses = np.concatenate([np.zeros(delay_steps), mealBoluses])
    mealBoluses = mealBoluses[:simTime]

    # Calculate constant basal insulin
    uBasal = BASAL_problem * np.array([(TDIR / basalTDIRRatio / (24 * 60)) * 1000.0 / bodyWeight])

    # Initial conditions
    if initStates is not None:
        xmPlant = initStates.copy()
    else:
        xmPlant, _ = createSH.get_plant_steady_state(initCGM, modParams)
        Q1 = initCGM / 18.0 * modParams[1]  # mg/dL to mmol/kg
        xmPlant[0] = Q1

    # Pre-allocate simulation arrays
    sim_day_arr = np.zeros(simTime, dtype=int)
    sim_hour_arr = np.zeros(simTime, dtype=int)
    input_insulin_arr = np.zeros(simTime, dtype=float)
    input_meal_carbs_arr = mealVector.copy().astype(float)
    output_glucose_arr = np.zeros(simTime, dtype=float)
    output_cgm_arr = np.zeros(simTime, dtype=float)
    states_arr = np.zeros((simTime, 9), dtype=float)
    correction_bolus_arr = np.zeros(simTime, dtype=float)

    samples_per_day = (24 * 60) // Ts
    meal_day_ids_arr = np.array([meal_day_ids_list[t // samples_per_day] for t in range(simTime)])

    v_cgm_noise = []
    mealAmt, mealTime = [], []
    rescueCounter = 40.0 / Ts
    rescueTime = np.array([], dtype=int)
    UrPlant, UrModel, DrPlant, DrModel, rescueTriggerCounter = 0, 0, 0, 0, 0
    rescueInsulinAdj = np.ones((simTime,))

    time_corr = []
    corr_switch = 0
    Bolus_C = 0.0

    mePIU, mePGU, meHGP = 1, 1, 1

    for t in range(simTime):
        sim_day = (t // samples_per_day) + 1
        sim_hour_of_day = (t % samples_per_day) // (60 // Ts)
        sim_day_arr[t] = sim_day
        sim_hour_arr[t] = sim_hour_of_day

        # Optional correction bolus
        if correction_bolus_enabled:
            Bolus_C, corr_switch, time_corr = Correction_Ins_Alg(
                output_cgm_arr[:t], mealTime, t, SIM_options.SIMULATION_HIGH_TARGET,
                time_corr, bodyWeight, corr_switch, CF, rescueTime, sim_hour_of_day,
                mealBoluses, output_cgm_arr[:t]
            )

        # Total insulin dose
        if t == 0:
            uTotal = uBasal[0]
        else:
            uTotal = rescueInsulinAdj[t] * uBasal[0] + mealBoluses[t] + Bolus_C

        # Meal response
        UgPlant, mealVectorPlant, mealTime, mealAmt = testSH.get_meal_response(
            mealAmt, mealTime, mealVector, t, modParams
        )

        # Circadian insulin sensitivity adjustment if enabled
        if SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_CHANGE:
            sf_adjusted = testSH.apply_insulin_sensitivity_circadian_changes(
                t, sub_idx, modParams0[-3:],
                SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_AMPLITUDE,
                SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_PHASE,
                SIM_options.SIMULATION_TS_MINUTES
            )
            modParams[-4:-1] = sf_adjusted

        # Plant ODE step
        Ap, Bp, Cp, Dp = createSH.plant(xmPlant, modParams, mePIU, mePGU, meHGP)
        xmPlant = np.matmul(Ap, xmPlant) + Bp * uTotal + Dp + mealVectorPlant * (UgPlant + UrPlant) * Ts
        yPlant = float(np.squeeze(np.matmul(Cp, xmPlant) * 18.0 / modParams[1]))

        # Sensor CGM reading with noise
        if t > 0:
            yCGM, v_cgm_noise = testSH.add_noise_to_cgm(
                yPlant, output_glucose_arr[t - 1], v_cgm_noise, sim_day - 1
            )
        else:
            yCGM, v_cgm_noise = testSH.add_noise_to_cgm(yPlant, None, v_cgm_noise, sim_day - 1)

        # Rescue carbs check
        UrPlant, UrModel, DrPlant, DrModel, rescueCounter, rescueTime, rescueTriggerCounter, rescueInsulinAdj = \
            testSH.get_rescue_carb(
                yPlant, UrPlant, UrModel, DrPlant, DrModel, rescueInsulinAdj, t, modParams,
                rescueCounter, rescueTime, rescueTriggerCounter
            )

        # Store timestep outputs
        input_insulin_arr[t] = uTotal * bodyWeight * 60.0 / 1000.0  # U/h
        output_glucose_arr[t] = yPlant
        output_cgm_arr[t] = float(yCGM[0])
        states_arr[t, :] = np.append(xmPlant, UgPlant + UrPlant)
        correction_bolus_arr[t] = Bolus_C * bodyWeight * 60.0 / 1000.0

    # Add 20g rescue carbs at trigger timestamps
    valid_rescues = rescueTime[rescueTime < simTime].astype(int)
    if len(valid_rescues) > 0:
        input_meal_carbs_arr[valid_rescues] += 20.0

    df = pd.DataFrame({
        'time': np.arange(simTime),
        'dosing_problem_id': int(problem),
        'sim_day': sim_day_arr,
        'initCGM': float(initCGM),
        'sim_hour_of_day': sim_hour_arr,
        'input_insulin': input_insulin_arr,
        'input_meal_carbs': input_meal_carbs_arr,
        'output_glucose': output_glucose_arr,
        'output_cgm': output_cgm_arr,
        'state_Q1': states_arr[:, 0],
        'state_Q2': states_arr[:, 1],
        'state_S1': states_arr[:, 2],
        'state_S2': states_arr[:, 3],
        'state_I': states_arr[:, 4],
        'state_X1': states_arr[:, 5],
        'state_X2': states_arr[:, 6],
        'state_X3': states_arr[:, 7],
        'state_Ug': states_arr[:, 8],
        'correction_bolus': correction_bolus_arr,
        'subject_id': str(subject_id),
        'meal_scenario': int(meal_scenario_id),
        'late_bolus_min': int(late_bolus_min),
        'meal_day_ids': meal_day_ids_arr,
        'data_split': split_name
    })

    return df, xmPlant


def simulate_scenarios(item):
    """
    Run 7-day simulations for a given scenario item.
    """
    (sub, subjectID, vp_data, modParamsOUT, body_weight, TDIR, basalTDIRRatio,
     target_split_dir, late_bolus_min, cf, split_name, corr_bolus_enabled) = item

    print(f"INFO: Processing virtual subject {subjectID} ({split_name}, delay: {late_bolus_min}m)")

    age = vp_data['subject'][sub].get('age', 35)
    simDays = SIM_options.SIMULATION_N_DAYS_WEEKS * SIM_options.SIMULATION_N_WEEKS
    samples_per_day = (24 * 60) // SIM_options.SIMULATION_TS_MINUTES

    meal_data_days = SIM_load_virtualpop_data.get_meal_scenario_days(
        vp_data, sub, arms=SIM_options.SIMULATION_SCENARIOS_ARMS
    )

    if len(meal_data_days) == 0:
        return None

    n_meal_scenarios = SIM_options.SIMULATION_N_MEAL_SCENARIOS
    mealScenarioOUT_array = SIM_load_virtualpop_data.get_meal_scenario_simulation(
        meal_data_days, n_meal_scenarios * simDays, random_sample=SIM_options.SIMULATION_SHUFFLE_MEALS
    )
    mealScenarioOUT_array = mealScenarioOUT_array.reshape(n_meal_scenarios, -1)

    target_dir = Path(target_split_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    mu_last = None

    for nsim, mealScenarioOUT in enumerate(mealScenarioOUT_array):
        # In original OHSU generator, get_initCGM is called separately for each meal scenario
        initCGM_array, mu = SIM_load_virtualpop_data.get_initCGM(vp_data, sub)
        if len(initCGM_array) == 0:
            continue
        mu_last = mu

        day_hashes = [
            hashlib.sha256(mealScenarioOUT[d * samples_per_day:(d + 1) * samples_per_day].tobytes()).hexdigest()[:12]
            for d in range(simDays)
        ]

        for ii, initCGM in enumerate(initCGM_array):
            dfs_dosing = []

            # 1. No dosing problem (case 0)
            if SIM_options.SIMULATION_NO_DOSING_PROBLEMS:
                ICR_problem = np.ones(simDays * samples_per_day, dtype=float)
                CF_problem = ICR_problem.copy()
                BASAL_problem = 1.0

                df_0, _ = OHSU_SIM_simulate_subject(
                    simDays, initCGM, modParamsOUT, body_weight,
                    TDIR, basalTDIRRatio, mealScenarioOUT,
                    ICR_problem, BASAL_problem, CF_problem, sub,
                    late_bolus_min, cf, 0, subjectID, nsim, day_hashes, split_name,
                    correction_bolus_enabled=corr_bolus_enabled,
                    age=age, sex='UNK'
                )
                dfs_dosing.append(df_0)

            # 2. Dosing problems 1 through 10
            for problem in SIM_options.SIMULATION_DOSING_PROBLEMS_DIC.keys():
                if problem not in SIM_options.SIMULATION_DOSING_PROBLEMS_SELECTED:
                    continue

                # In original generator, generate_problems_matrix is called separately for each problem
                probmatrix = SIM_generate_dosing_problems.generate_problems_matrix(
                    SIM_options.SIMULATION_DOSING_PROBLEMS_DIC
                )

                BASAL_problem, ICR_problem_day, CF_problem_day = (
                    SIM_generate_dosing_problems.generate_sim_day_problems(problem, probmatrix)
                )
                ICR_problem = np.tile(ICR_problem_day, simDays)
                CF_problem = np.tile(CF_problem_day, simDays)

                df_prob, _ = OHSU_SIM_simulate_subject(
                    simDays, initCGM, modParamsOUT, body_weight,
                    TDIR, basalTDIRRatio, mealScenarioOUT,
                    ICR_problem, BASAL_problem, CF_problem, sub,
                    late_bolus_min, cf, problem, subjectID, nsim, day_hashes, split_name,
                    correction_bolus_enabled=corr_bolus_enabled,
                    age=age, sex='UNK'
                )
                dfs_dosing.append(df_prob)

            if len(dfs_dosing) > 0:
                df_sub_sim = pd.concat(dfs_dosing, ignore_index=True)
                # Exact original filename format: {subjectID}mealScenario_{nsim}_initCGM_{ii}_{int(initCGM)}.pkl
                out_name = f"{subjectID}mealScenario_{nsim}_initCGM_{ii}_{int(initCGM)}.pkl"
                out_path = target_dir / out_name
                df_sub_sim.to_pickle(str(out_path))

    return mu_last


def run_dataset_generation(dataset_ids=None, filter_subjects=None, parallel=False, output_base=None):
    """
    Generate dataset(s) adhering exactly to the HPC dataset layout.
    """
    if dataset_ids is None:
        dataset_ids = ['SIM002']
    elif isinstance(dataset_ids, str):
        dataset_ids = ['SIM002', 'SIM004', 'SIM005'] if dataset_ids.upper() == 'ALL' else [dataset_ids]

    vp_data = SIM_load_virtualpop_data.load_virtual_subjects(str(SIM_options.SIMULATION_VP_PKL))
    results_base = Path(output_base) if output_base else SIM_options.SIMULATION_RESULTS_DIR

    TDIR = SIM_options.SIMULATION_TDIR
    basalTDIRRatio = SIM_options.SIMULATION_BASAL_TDIR_RATIO
    modParamsOUT = SIM_options.SIMULATION_POPULATION_PARAMS[:15]
    body_weight = SIM_options.SIMULATION_POPULATION_PARAMS[-1]

    for sim_id in dataset_ids:
        if sim_id not in SIM_options.HPC_DATASET_CONFIGS:
            print(f"Unknown dataset ID: {sim_id}. Choose from SIM002, SIM004, SIM005, or ALL.")
            continue

        cfg = SIM_options.HPC_DATASET_CONFIGS[sim_id]
        print("\n" + "=" * 70)
        print(f"GENERATING DATASET: {sim_id} ({cfg['description']})")
        print("=" * 70)

        for scen in cfg['scenarios']:
            folder = scen['folder']
            delay = scen['delay']
            use_splits = scen['use_splits']
            subjects_mode = scen['subjects_mode']
            corr_bolus = scen['correction_bolus']
            cf = scen['cf']

            # Target subjects for this scenario
            if filter_subjects is not None:
                target_subjects = [s.zfill(6) for s in filter_subjects]
            elif subjects_mode == 'TRAIN_18':
                target_subjects = SIM_options.SIMULATION_SUBJECTS_TRAIN
            else:
                target_subjects = SIM_options.SIMULATION_SUBJECTS_INCLUDED

            # Build items
            items = []
            for sub in range(len(vp_data['subject'])):
                sid = str(vp_data['subject'][sub]['subjectID']).zfill(6)
                if sid not in target_subjects:
                    continue

                if use_splits:
                    if sid in SIM_options.SIMULATION_SUBJECTS_TRAIN:
                        split_name = 'TRAIN'
                    elif sid in SIM_options.SIMULATION_SUBJECTS_TEST:
                        split_name = 'TEST'
                    elif sid in SIM_options.SIMULATION_SUBJECTS_VALIDATION:
                        split_name = 'VALIDATION'
                    else:
                        split_name = 'TRAIN'
                else:
                    split_name = 'TRAIN'

                target_split_dir = results_base / sim_id / folder / split_name

                items.append([
                    sub, sid, vp_data, modParamsOUT, body_weight,
                    TDIR, basalTDIRRatio, str(target_split_dir), delay, cf,
                    split_name, corr_bolus
                ])

            print(f"\n---> {sim_id} / {folder} : {len(items)} subjects to simulate")

            if parallel and len(items) > 1:
                num_processes = min(multiprocessing.cpu_count(), len(items))
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.map(simulate_scenarios, items)
            else:
                for item in items:
                    simulate_scenarios(item)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset matching HPC structure (SIM002, SIM004, SIM005).")
    parser.add_argument('--dataset', default='SIM002', choices=['SIM002', 'SIM004', 'SIM005', 'ALL'],
                        help="Dataset ID to generate (default: SIM002). Use 'ALL' for full 5,010-file dataset.")
    parser.add_argument('--subjects', default=None, help="Optional comma-separated list of subject IDs for test runs")
    parser.add_argument('--output-dir', default=None, help="Optional custom output directory base")
    parser.add_argument('--parallel', action='store_true', help="Enable multi-process parallelism")
    args = parser.parse_args()

    filter_subs = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    run_dataset_generation(dataset_ids=args.dataset, filter_subjects=filter_subs,
                           parallel=args.parallel, output_base=args.output_dir)


if __name__ == '__main__':
    main()
