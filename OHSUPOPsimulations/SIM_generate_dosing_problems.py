import numpy as np
from OHSUPOPsimulations import SIM_options

N_PROBLEMS = 10
N_TIMEFRAMES = 4  # Relevant for ICR

# Clean 24-hour partition (no overlapping or missing hours)
# 1: 7AM - 11AM (hours 7, 8, 9, 10)
# 2: 11AM - 3PM (hours 11, 12, 13, 14)
# 3: 3PM - 8PM (hours 15, 16, 17, 18, 19)
# 4: 8PM - 7AM (hours 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6)
TIMEFRAMES_DIC = {
    1: np.arange(7, 11),
    2: np.arange(11, 15),
    3: np.arange(15, 20),
    4: np.append(np.arange(20, 24), np.arange(0, 7))
}

PROBLEM_REQ_TIMEFRAME_INPUT = [1, 2, 3, 4, 7, 8, 9, 10]
PROBLEM_ADJUSTMENT_FACTOR_BASAL = 0.4
PROBLEM_ADJUSTMENT_FACTOR_ICR = 0.4


def get_adjustment(max_adjustment, tolerance_below=SIM_options.SIMULATION_DOSING_PROBLEMS_ADJUSTMENT_TOLERANCE):
    """Return a scalar float adjustment factor."""
    adj_tolerance = max_adjustment * tolerance_below
    return float(np.random.uniform(low=adj_tolerance, high=max_adjustment))


def generate_problems_matrix(problems_dic):
    """
    Generate matrix of problem adjustments.
    Columns: index 0 is Basal multiplier; indices 1-4 correspond to timeframes 1-4 for ICR multiplier.
    """
    problems_matrix = np.ones((N_PROBLEMS, 1 + N_TIMEFRAMES), dtype=float)
    for pp in problems_dic.keys():
        if pp <= 0 or pp > N_PROBLEMS:
            continue
        timeframe = np.asarray(problems_dic[pp], dtype=int)
        timeframe = timeframe[(timeframe > 0) & (timeframe <= N_TIMEFRAMES)]
        if len(timeframe) == 0 and (pp in PROBLEM_REQ_TIMEFRAME_INPUT):
            continue

        adj_icr = get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_ICR)
        adj_basal = get_adjustment(PROBLEM_ADJUSTMENT_FACTOR_BASAL)

        # Implementing problems
        if pp == 1:  # Post-prandial hypoglycemia --> ICR low (over-bolusing)
            problems_matrix[pp - 1, timeframe] = 1.0 - (adj_icr / 2.0)
        elif pp == 2:  # Post-prandial severe hypoglycemia --> ICR very low
            problems_matrix[pp - 1, timeframe] = 1.0 - adj_icr
        elif pp == 3:  # Post-prandial hyperglycemia --> ICR high (under-bolusing)
            problems_matrix[pp - 1, timeframe] = 1.0 + (adj_icr / 2.0)
        elif pp == 4:  # Post-prandial severe hyperglycemia --> ICR very high
            problems_matrix[pp - 1, timeframe] = 1.0 + adj_icr
        elif pp == 5:  # Aggressive basal
            problems_matrix[pp - 1, 0] = 1.0 + adj_basal
        elif pp == 6:  # Insufficient basal
            problems_matrix[pp - 1, 0] = 1.0 - adj_basal
        elif pp == 7:  # Basal high, ICR high
            problems_matrix[pp - 1, 0] = 1.0 + adj_basal
            problems_matrix[pp - 1, timeframe] = 1.0 + adj_icr
        elif pp == 8:  # Basal high, ICR low
            problems_matrix[pp - 1, 0] = 1.0 + adj_basal
            problems_matrix[pp - 1, timeframe] = 1.0 - adj_icr
        elif pp == 9:  # Basal low, ICR high
            problems_matrix[pp - 1, 0] = 1.0 - adj_basal
            problems_matrix[pp - 1, timeframe] = 1.0 + adj_icr
        elif pp == 10:  # Basal low, ICR low
            problems_matrix[pp - 1, 0] = 1.0 - adj_basal
            problems_matrix[pp - 1, timeframe] = 1.0 - adj_icr

    return problems_matrix


def generate_sim_day_problems(problem_idx, problems_matrix):
    """
    Generate the 1-day basal, ICR, and CF vectors for a given problem index.
    Returns:
        BASAL_problem: scalar float
        ICR_problem_day: array of shape (288,)
        CF_problem_day: array of shape (288,)
    """
    n_repeats = (1 * 60) // SIM_options.SIMULATION_TS_MINUTES  # 12 samples per hour

    # Basal problem
    BASAL_problem = float(problems_matrix[problem_idx - 1, 0])

    # ICR problem across 24 hours
    ICR_problem = np.ones(24, dtype=float)
    for ii in TIMEFRAMES_DIC.keys():
        ICR_problem[TIMEFRAMES_DIC[ii]] = float(problems_matrix[problem_idx - 1, ii])

    # Correction factor problem
    CF_problem = ICR_problem.copy()

    return BASAL_problem, np.repeat(ICR_problem, n_repeats), np.repeat(CF_problem, n_repeats)
