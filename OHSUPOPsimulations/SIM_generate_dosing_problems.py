import numpy as np
from OHSUPOPsimulations import SIM_options

N_PROBLEMS = 10
N_TIMEFRAMES = 4  # Relevant for ICR

# Original OHSU timeframes (with original definition and bounds)
TIMEFRAMES_DIC = {
    1: np.arange(7, 12),                             # 7-11AM
    2: np.arange(11, 16),                            # 11AM-3PM
    3: np.arange(15, 21),                            # 3PM-8PM
    4: np.append(np.arange(0, 7), np.arange(20, 23))  # 8PM-7AM
}

PROBLEM_REQ_TIMEFRAME_INPUT = [1, 2, 3, 4, 7, 8, 9, 10]
PROBLEM_ADJUSTMENT_FACTOR_BASAL = 0.4
PROBLEM_ADJUSTMENT_FACTOR_ICR = 0.4


def get_adjustment(max_adjustment, tolerance_below=SIM_options.SIMULATION_DOSING_PROBLEMS_ADJUSTMENT_TOLERANCE):
    """Original adjustment sampling returning 1D array of size 1."""
    adj_tolerance = max_adjustment * tolerance_below
    return np.random.uniform(low=adj_tolerance, high=max_adjustment, size=1)


def generate_problems_matrix(problems_dic):
    """
    Generate matrix of problem adjustments with original per-problem random consumption.
    Columns: index 0 is Basal multiplier; indices 1-4 correspond to timeframes 1-4 for ICR multiplier.
    """
    problems_matrix = np.ones((N_PROBLEMS, 1 + N_TIMEFRAMES))
    for pp in problems_dic.keys():
        if (pp <= 0) or (pp > N_PROBLEMS):
            continue
        timeframe = np.asarray(problems_dic[pp])
        timeframe = timeframe[(timeframe > 0) & (timeframe <= N_TIMEFRAMES)]
        if len(timeframe) == 0 and (pp in PROBLEM_REQ_TIMEFRAME_INPUT):
            continue

        # Implementing problems matching original random call order
        if pp == 1:  # Post-prandial hypoglycemia --> ICR low
            problems_matrix[pp - 1, timeframe] = 1 - (get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR) / 2)
        elif pp == 2:  # Post-prandial severe hypoglycemia --> ICR very low
            problems_matrix[pp - 1, timeframe] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        elif pp == 3:  # Post-prandial hyperglycemia --> ICR high
            problems_matrix[pp - 1, timeframe] = 1 + (get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR) / 2)
        elif pp == 4:  # Post-prandial severe hyperglycemia --> ICR very high
            problems_matrix[pp - 1, timeframe] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        elif pp == 5:  # Aggressive basal
            problems_matrix[pp - 1, 0] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
        elif pp == 6:  # Insufficient basal
            problems_matrix[pp - 1, 0] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
        elif pp == 7:  # Basal high, ICR high
            problems_matrix[pp - 1, 0] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
            problems_matrix[pp - 1, timeframe] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        elif pp == 8:  # Basal high, ICR low
            problems_matrix[pp - 1, 0] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
            problems_matrix[pp - 1, timeframe] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        elif pp == 9:  # Basal low, ICR high
            problems_matrix[pp - 1, 0] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
            problems_matrix[pp - 1, timeframe] = 1 + get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        else:  # Basal low, ICR low (problem 10)
            problems_matrix[pp - 1, 0] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)
            problems_matrix[pp - 1, timeframe] = 1 - get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)

    return problems_matrix


def generate_sim_day_problems(problem_idx, problems_matrix):
    n_repeats = (1 * 60) // SIM_options.SIMULATION_TS_MINUTES

    # Basal problem
    BASAL_problem = problems_matrix[problem_idx - 1, 0]

    # ICR problem
    ICR_problem = np.ones(24)
    for ii in TIMEFRAMES_DIC.keys():
        ICR_problem[TIMEFRAMES_DIC[ii]] = problems_matrix[problem_idx - 1, ii]

    # Correction factor CF problem --> Currently implemented using same problems in ICR
    CF_problem = ICR_problem

    return BASAL_problem, np.repeat(ICR_problem, n_repeats), np.repeat(CF_problem, n_repeats)

