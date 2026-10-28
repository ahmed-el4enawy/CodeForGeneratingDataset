# ---------------------------------------------
# Authors: Nichole Tyler (Original in Matlab), Valentina Roquemen (Translate to python),
# Date: February 23, 2024
# Updated for linear IOB & paper conformance
# ---------------------------------------------

import numpy as np
from OHSUPOPsimulations import SIM_options


def IOB_Linear_Calculation(IIR_mat):
    """
    Calculate Insulin On Board (IOB) using a 4-hour (48 sample @ 5-min) linear decay.
    IIR_mat: insulin infusion rate in U/h (converted to U/sample by dividing by 12)
             or bolus series. Can be a 1D slice of history (up to 48 samples) or longer.
    """
    arr = np.asarray(IIR_mat, dtype=float).flatten()
    units_per_sample = arr / 12.0  # Convert U/h to Units per 5-min sample
    li = len(units_per_sample)
    if li == 0:
        return 0.0

    if li <= 48:
        # Time since each bolus in samples: age in 1..li
        ages = li - np.arange(li)
        decay = np.maximum(0.0, (48.0 - ages) / 48.0)
        return float(np.sum(units_per_sample * decay))
    else:
        # Full time series calculation
        kernel = (48.0 - np.arange(1, 49)) / 48.0
        padded = np.pad(units_per_sample, (48, 0), mode='constant')
        iob_series = np.convolve(padded, kernel, mode='valid')[:li]
        return iob_series


def Correction_Ins_Alg(Output, meal_time, kk, high_target, time_corr, Weight, corr_switch, CF, time_resc, thour, u_bolus, cgm):
    """
    Checks for hyperglycemia and calculates smart correction bolus using IOB and trend.
    Output: CGM history list/array in mg/dL
    kk: current sample index
    high_target: target glucose (mg/dL)
    Weight: subject body weight (kg)
    corr_switch: cooldown switch
    CF: correction factor
    """
    if kk <= 0:
        return 0.0, 0, time_corr

    # Daytime vs Nighttime correction threshold
    if thour > 21 or thour < 7:
        thr_corr = 280.0
    else:
        thr_corr = float(getattr(SIM_options, 'SIMULATION_GLUCOSE_THRESHOLD_CORRECTION_BOLUS', 200.0))

    prev_cgm = Output[kk - 1]
    Bolus_C = 0.0

    # Test 1: Check threshold, post-meal delay (>= 4 hours), and switch
    if (corr_switch == 0) and (prev_cgm > thr_corr) and len(meal_time) > 0 and ((kk - meal_time[-1]) >= 48):
        uiob = u_bolus[max(0, kk - 48):kk]
        iob_vals = IOB_Linear_Calculation(uiob * Weight * 60.0 / 1000.0)

        # Rate of change over past ~1 hour
        start_trend = max(0, kk - 14)
        cgm_trend_slice = cgm[start_trend:kk]
        if len(cgm_trend_slice) > 1:
            de_val = float(np.sum(np.diff(cgm_trend_slice)) / (5.0 * 12.0))
        else:
            de_val = 0.0

        # Trend-offset adjustment
        if de_val > 3.0:
            offset = 100.0
        elif de_val > 2.0:
            offset = 75.0
        elif de_val > 1.0:
            offset = 50.0
        elif de_val < -3.0:
            offset = -100.0
        elif de_val < -2.0:
            offset = -75.0
        elif de_val < -1.0:
            offset = -50.0
        else:
            offset = 0.0

        cf_val = (prev_cgm + offset - high_target) / CF - iob_vals
        if cf_val < 0.0:
            cf_val = 0.0

        Bolus_C = cf_val * 1000.0 / (Weight * 5.0)  # in mU/kg/min

        # Don't correct if rescue carbs administered in the last hour (12 samples)
        if len(time_resc) > 0 and ((kk - time_resc[-1]) < 12):
            Bolus_C = 0.0

        if Bolus_C > 0.0:
            time_corr.append(kk)
            corr_switch = 1

    # Test 2: Reset switch after 3 hours (36 samples) between correction boluses
    if len(time_corr) > 0 and ((kk - time_corr[-1]) > 36):
        corr_switch = 0

    return Bolus_C, corr_switch, time_corr
