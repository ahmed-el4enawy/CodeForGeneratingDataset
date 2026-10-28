from pathlib import Path
import numpy as np

# Simulation Execution Mode
SIMULATION_SERIAL = True
SIMULATION_ID = 'SIM005'
SIMULATION_TYPES = ['POPULATION', 'PERSONALIZATION']
SIMULATION_TYPE_SELECTED = 0

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SIMULATION_RESULTS_DIR = REPO_ROOT / 'simulation_results'
SIMULATION_VP_PKL = REPO_ROOT / 'all_subject_t1dexi_pilot_06222023.pkl'

# Virtual Subjects Included in Cohort (28 subjects, Roquemen-Echeverri et al. Sec 2.2.1)
SIMULATION_SUBJECTS_INCLUDED = [
    '001003', '001004', '001008', '001012', '001016', '001017', '001026', '001032', '001033', '001034',
    '037001', '037002', '037003', '037004', '037006', '037007', '037008', '037014',
    '042006', '042007', '042011', '042012', '042013',
    '062005', '064003', '064005', '064008', '064011'
]

# Simulation Time & Sampling
SIMULATION_N_DAYS_WEEKS = 7
SIMULATION_N_WEEKS = 1
SIMULATION_TS_MINUTES = 5

# Meals & Initial Glucose
SIMULATION_SHUFFLE_MEALS = True
SIMULATION_SCENARIOS_ARMS = ['df']
SIMULATION_N_MEAL_SCENARIOS = 3
SIMULATION_N_CGM_INITIALIZATION = 5
SIMULATION_count_min = 1
SIMULATION_count_max = 9
SIMULATION_amount_min = 30
SIMULATION_amount_max = 359

# Bolus Timing Delays (0 to 45 min in 5 min steps)
SIMULATION_LATE_BOLUS_MIN = list(range(0, 50, 5))

# Model Parameters (Resalat et al. 2019 population values)
SIMULATION_BASAL_TDIR_RATIO = 1.78
SIMULATION_TDIR = 47
SIMULATION_POPULATION_PARAMS = np.array([
    0.00970, 0.16000, 0.06600, 0.80000, 40.0000,
    0.01610, 55.0000, 0.13800, 0.12000, 0.00600,
    0.06000, 0.03000, 0.00210, 0.00035, 0.02140, 76
])

# Dosing Problems (0 = normal, 1-10 = dosing problems)
SIMULATION_NO_DOSING_PROBLEMS = True
SIMULATION_DOSING_PROBLEMS_ADJUSTMENT_TOLERANCE = 0.7  # Up to 70% of max adjustment
SIMULATION_DOSING_PROBLEMS_SELECTED = list(np.arange(10) + 1)
SIMULATION_DOSING_PROBLEMS_DIC = {  # problem_idx: active timeframes
    1: [2, 4],
    2: [2, 4],
    3: [2, 4],
    4: [2, 4],
    5: [],
    6: [],
    7: [2, 4],
    8: [2, 4],
    9: [2, 4],
    10: [2, 4]
}

# Correction Bolus (Off by default per paper Sec 2.2.1)
SIMULATION_CORRECTION_BOLUS = False
SIMULATION_HIGH_TARGET = 120
SIMULATION_GLUCOSE_THRESHOLD_CORRECTION_BOLUS = 200

# Circadian Insulin Sensitivity Variation
SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_CHANGE = False
SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_AMPLITUDE = 0.2
SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_PHASE = 0.77
