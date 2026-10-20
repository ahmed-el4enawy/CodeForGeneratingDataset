# ---------------------------------------------
# Authors: Gavin Young, Navid Resalat, Peter Jacobs
#         Oregon Health & Science University
# Date: January 29, 2021
#
# File: test_SHP.py
# Function: This module contains functions to test a population of virtual patients
#           represented by a mathematical glucoregulatory model.  The model includes
#           a representation of insulin kinetics, insulin dynamics, and carbohydrate
#           kinetics.
#
# Code adapted from:
# A STATISTICAL VIRTUAL PATIENT POPULATION FOR THE GLUCOREGULATORY SYSTEM
# IN TYPE 1 DIABETES WITH INTEGRATED EXERCISE MODEL
# Resalat N, El Youssef J, Tyler N, Castle J, Jacobs PG, PlOS One 2019
#
# Free to use for academic research and teaching applications
# For commercial use, please contact the authors for licensing.
#
# Needs work:
# TODO: Document get_meal_response()
# TODO: Document get_rescue_carb()
# TODO: Document get_exercise_response()
# TODO: Static type function parameters?
#
# ---------------------------------------------
# Imports

import numpy as np
from sympy import Function, Eq, dsolve, symbols
from numba import jit

# ---------------------------------------------
# Parameters

# ---------------------------------------------


# @jit(nopython=True, parallel=True)
def get_exercise_response(t, exercise_start_time, exercise_duration, pvo2_max=60.0, pamm=0.5, pgua_1_active=0.0, ts=5):
    """
    Hernandez aerobic exercise model

    :param t: Current time index
    :param pvo2_max: Percentage of maximum oxygen consumption [0-100]
    :param pamm: Percentage of active muscular mass [0-1]
    :param exercise_start_time: Time start of exercise (in minutes)
    :param exercise_duration: Duration of exercise activity (in minutes)
    :param pgua_1_active: Initial periphery glucose uptake at active tissues (default: 0)
    :param ts: Sample period (default: 5 min)
    :return:
    """

    # Exercise model parameters
    exercise_start_idx = exercise_start_time / ts
    exercise_end_idx = (exercise_start_time + exercise_duration) / ts
    mean_pgua_1 = 0.006 * pvo2_max ** 2 + 1.2264 * pvo2_max - 10.1958
    t_idx = t

    # Exercise response
    if exercise_start_idx <= t_idx <= exercise_end_idx:                               # During exercise

        t, f = symbols('t'), Function('f')
        diff_eq = Eq(f(t).diff(t), -1 / 30 * f(t) + 1 / 30 * mean_pgua_1)           # Define equality
        yy_1 = dsolve(diff_eq, hint='1st_linear', ics={f(0): 0.1})                  # Solve diff-eq
        t_now = ts * (t_idx - exercise_start_idx)                                   # Calculate current time
        pgua_1_active = yy_1.subs(t, t_now).rhs                                     # Plug in time and compute

        m_e_pgu = 1 + pgua_1_active * pamm / 35
        m_e_piu = 1 + 2.4 * pamm
        m_e_hgp = 1 + pgua_1_active * pamm / 155

    elif exercise_end_idx < t_idx < (exercise_end_idx + 5 * 60 / ts):                 # Immediately post-exercise

        t, f = symbols('t'), Function('f')
        diff_eq = Eq(f(t).diff(t), -1 / 30 * f(t))                                  # Define equality
        zz_1 = dsolve(diff_eq, hint='1st_linear', ics={f(0): pgua_1_active})        # Solve diff-eq
        t_now = ts * (t_idx - exercise_end_idx)                                     # Calculate current time
        pgua_1_non_active = zz_1.subs(t, t_now).rhs                                 # Plug in time and compute

        m_e_pgu = 1 + pgua_1_non_active * pamm / 35
        m_e_piu = 1
        m_e_hgp = 1 + pgua_1_non_active * pamm / 155

    else:                                                                               # Non-exercise period

        m_e_pgu = 1
        m_e_piu = 1
        m_e_hgp = 1

    return m_e_piu, m_e_pgu, m_e_hgp, pgua_1_active


def get_meal_response(meal_amount, meal_time, meal_vector, t_idx, model_parameters, ts=5, n_states=8, snack_tmax_carbs = None):
    """
    This function calculates the impact of a meal using the meal model described in Hovorka et al. 2004

    :param meal_amount: Amount of carbs consumed in last meal consumed (in grams)
    :param meal_time: Time index last meal was consumed
    :param meal_vector: Vector of meal amounts
    :param t_idx: Current sample index
    :param model_parameters: 16-element vector of model parameters
    :param ts: Sample period (default: 5 min)
    :param n_states: Number of states in plant model (default: 8)
    :return:
    """

    # Extract model parameters
    a_g = model_parameters[3]
    body_weight = model_parameters[15]
    if snack_tmax_carbs is None:
        tmax_carbs = model_parameters[4] # Changed this for long-lasting
    else:
        tmax_carbs = snack_tmax_carbs

    # Add new meal to running array of consumed meals
    if t_idx > 0:
        current_meal_size = meal_vector[t_idx]
        if current_meal_size != 0:
            meal_time += [t_idx]
            meal_amount += [current_meal_size]

    # Absorb meal carbohydrate into model
    if len(meal_time) == 0:
        ug_plant = 0
    else:
        dg_plant = meal_amount / body_weight / 0.18
        ug_plant_vector = dg_plant * a_g * ts * (t_idx - np.array(meal_time)) * \
                          np.exp(-ts * (t_idx - np.array(meal_time)) / tmax_carbs) / tmax_carbs ** 2
        ug_plant = ug_plant_vector.sum()
    plant_meal_vector = np.zeros(shape=(n_states, 1))
    plant_meal_vector[0] = 1

    return ug_plant, plant_meal_vector, meal_time, meal_amount


def get_rescue_carb(y_plant, ur_plant, ur_model, dr_plant, dr_model, rescue_insulin_adjusted, t_idx, model_parameters,
                    rescue_counter, rescue_time, rescue_trigger_counter, ts=5, rescue_threshold=70, rescue_carbs=20,
                    rescue_insulin_window=40, rescue_iir_reduction=1.0, rescue_insulin_timer=40, rescue_tmax=20,
                    rescue_carb_delay=20):
    """
    This function implements an adjustment to the states following a hypoglycemic event whereby rescue carbohydrates
    are given to the patient. This function returns the amount of updated carbohydrates taken by the patient
    as well as handles the model for absorption of these carbohydrates in the gut into the blood.

    :param y_plant:
    :param ur_plant:
    :param ur_model:
    :param dr_plant:
    :param dr_model:
    :param rescue_insulin_adjusted:
    :param t_idx:
    :param ts:
    :param model_parameters: 16-element vector of model parameters
    :param rescue_counter: A control flag to identify the onset of the next glucose measurement after a rescue carb
    :param rescue_time:
    :param rescue_trigger_counter:
    :param rescue_threshold: Glucose threshold below which rescue carbs are given (default: 70 mg/dL)
    :param rescue_carbs: Grams of rescue carbohydrates given once glucose is below threshold (default: 20 g)
    :param rescue_insulin_window: Length of time insulin dosing is reduced following rescue carbs (default: 40 min)
    :param rescue_iir_reduction: Magnitude by which insulin doses are reduced following rescue carbs (default: 1.0)
                                 Note: prior value was 0.25
    :param rescue_insulin_timer: Length of time insulin dosing is reduced following rescue carbs (default: 40 min)
    :param rescue_tmax: time-to-maximum rescue carb absorption (default: 20)
    :param rescue_carb_delay: Delay in consuming rescue carb after crossing hypoglycemia threshold (default: 20 min)
    :return:
    """

    a_g = model_parameters[3]
    body_weight = model_parameters[15]
    rescue_window = rescue_insulin_window // ts
    rescue_timer = rescue_insulin_timer // ts

    if (y_plant < rescue_threshold) and rescue_counter == rescue_window:
        # TODO: check window conditions below (missing conditions in Ravi's code?)
        if not rescue_time.size == 0 and (t_idx - rescue_time[-1]) > 60 / ts:
            rescue_delay = rescue_carb_delay // ts
        elif not rescue_time.size == 0 and (t_idx - rescue_time[-1]) <= rescue_window:
            rescue_delay = 0 // ts
        else:
            rescue_delay = rescue_carb_delay // ts
        rescue_insulin_adjusted[(t_idx + rescue_delay):(t_idx + rescue_delay + rescue_timer)] = rescue_iir_reduction
        rescue_trigger_counter = 1
        rescue_time = np.append(rescue_time, [t_idx + rescue_delay])
        dr_plant = rescue_carbs / body_weight / 0.18
        dr_model = rescue_carbs / body_weight * 1000
        rescue_counter = 0

    # Update plant and controller model with new rescue carbs
    if rescue_trigger_counter == 1:
        ur_plant = np.multiply(dr_plant * a_g * ts * (t_idx - rescue_time + 1),
                               np.exp(-ts * (t_idx - rescue_time + 1) / rescue_tmax) / rescue_tmax ** 2)
        ur_plant[ur_plant < 0] = 0
        ur_model = np.multiply(dr_model * a_g * ts * (t_idx - rescue_time + 1),
                               np.exp(-ts * (t_idx - rescue_time + 1) / rescue_tmax) / rescue_tmax ** 2)
        ur_model[ur_model < 0] = 0
        ur_plant = ur_plant.sum()
        ur_model = ur_model.sum()

    if t_idx > 0:
        if not len(rescue_time) == 0 and (t_idx - rescue_time[-1]) >= (rescue_window - 1):
            rescue_counter = rescue_window

    return (ur_plant, ur_model, dr_plant, dr_model, rescue_counter, rescue_time,
            rescue_trigger_counter, rescue_insulin_adjusted)


def add_noise_to_cgm(plantout_curr, plantout_prev, v_cgm_noise, n_sim_days, Ts=5):  # Implemented by Nichole Tyler
    # Vettoretti et al., Sensors 2019, 19(23), 5320; https://doi.org/10.3390/s19235320
    # Noise model parameters
    tau = 3.78  # minutes
    a0 = 0.95  # unitless
    a1 = 0.004  # 1/days
    a2 = 0  # 1/(days^2)
    b0 = 6.35  # mg/dL
    alpha_1 = 1.30  # unitless
    alpha_2 = -0.42  # unitless
    sigma = 3.19  # mg/dL

    w = np.random.normal(loc=0.0, scale=sigma, size=1)

    if len(v_cgm_noise) >= 2:
        IG_t = (Ts/tau)*(plantout_curr - plantout_prev) + plantout_prev
        a = a0 + a1 * n_sim_days + a2 * np.power(n_sim_days,2)
        b = b0
        IG_s = a*IG_t + b
        v_curr = alpha_1*v_cgm_noise[-1] + alpha_2*v_cgm_noise[-2] + w
    else:
        IG_t = plantout_curr
        IG_s = a0 * IG_t + b0
        v_curr = w

    v_cgm_noise.append(v_curr)
    return IG_s + v_curr, v_cgm_noise

def apply_insulin_sensitivity_circadian_changes(sim_t, subject_index, insulin_sensitivity_params, amplitude, phase, Ts=5):
    adjustment = 1 + amplitude * np.sin(2 * np.pi / (24 * 60 // Ts) * sim_t + 2 * np.pi * phase * subject_index)
    return insulin_sensitivity_params*adjustment