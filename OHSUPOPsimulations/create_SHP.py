# ---------------------------------------------
# Authors: Gavin Young, Navid Resalat, Peter Jacobs
#         Oregon Health & Science University
# Date: January 22, 2021
#
# File: create_SHP.py
# Function: This module contains functions to generate a population of virtual patients
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
# TODO: Test create_virtual_patient_population()
# TODO: Test solve_ss()
# TODO: Test prune_population()
# TODO: Static type function parameters?
#
# ---------------------------------------------
# Imports

import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import expm

# ---------------------------------------------
# Parameters

# ---------------------------------------------


def create_virtual_patient_population(hovorka_parameters, n_population=100, insulin_reduction_factor=1, seed=0):
    """
    :param hovorka_parameters: 15-element column vector of hovorka model parameters
    :param n_population: [>1] max number of virtual subjects to generate
    :param insulin_reduction_factor: [0-1] insulin reduction factor
    :param seed: numpy random seed value
    :return: 15 x n array of all virtual subject model parameters, array of randomly sampled body weights
    """

    np.random.seed(seed)
    std_ste = 1
    sf2_sf1_std_ratio = 1 - 0.75
    sf2_sf3_std_ratio = 1 - 0.25

    # Randomize the Sf2 parameter while ensuring that Sf1, Sf2, and Sf3 correlate
    sf2_a = std_ste * 0.00032 * insulin_reduction_factor
    sf2_b = 0.00082 * insulin_reduction_factor
    rnd1 = np.random.normal(size=(n_population, 1))
    rnd2 = np.random.uniform(size=(n_population, 1))
    sf2_samples = sf2_a * rnd1 + sf2_b

    # Randomize the Sf1 parameter while ensuring that Sf1, Sf2, and Sf3 correlate
    sf1_a = std_ste * 0.00144 * insulin_reduction_factor
    sf1_b = 0.00512 * insulin_reduction_factor
    sf1_w1 = -sf2_sf1_std_ratio
    sf1_w2 = sf2_sf1_std_ratio
    sf1_samples = sf1_a * (rnd1 + (sf1_w1 + (sf1_w2 - sf1_w1) * rnd2)) + sf1_b

    # Randomize the Sf3 parameter while ensuring that Sf1, Sf2, and Sf3 correlate
    sf3_a = std_ste * 0.01375 * insulin_reduction_factor
    sf3_b = 0.052 * insulin_reduction_factor
    sf3_w1 = -sf2_sf3_std_ratio
    sf3_w2 = sf2_sf3_std_ratio
    sf3_samples = sf3_a * (rnd1 + (sf3_w1 + (sf3_w2 - sf3_w1) * rnd2)) + sf3_b

    # Drop sampled virtual subject parameters less than zero
    sampled_param_pairs = np.hstack((sf1_samples, sf2_samples, sf3_samples))
    valid_param_pairs = sampled_param_pairs[np.all(sampled_param_pairs > 0, axis=1)]

    # Generate full parameter array for all virtual subjects
    population_parameters = np.tile(hovorka_parameters, reps=(1, len(valid_param_pairs)))
    population_parameters[12, :] = valid_param_pairs[:, 0]          # Sf1
    population_parameters[13, :] = valid_param_pairs[:, 1]          # Sf2
    population_parameters[14, :] = valid_param_pairs[:, 2]          # Sf3

    # Randomize population body weights
    population_body_weights = (15.9 * np.random.normal(size=(n_population, 1)) + 50).T
    population_parameters[15, :] = population_body_weights          # Body weight

    return population_parameters, population_body_weights


def solve_ss(x, q1, model_parameters):
    """
    This function describes the Hovorka glucose compartment model

    :param x: 8-element state vector at current time step
    :param q1: current glucose concentration in Q1 compartment
    :param model_parameters: Hovorka model parameters
    :return: state vector at next time step
    """

    # Model parameters
    if model_parameters.shape[0] == 16:
        fc01, vd_g, k12, a_g, tmax_g, egp0, \
            tmax_i, ke, vd_i, ka1, ka2, ka3, sf1, sf2, sf3, kg = model_parameters.ravel()
    else:
        fc01, vd_g, k12, a_g, tmax_g, egp0, \
            tmax_i, ke, vd_i, ka1, ka2, ka3, sf1, sf2, sf3, \
            kg1, kge1, kge2, vd_gg, sf_gg, kc, kg3, kg2 = model_parameters.ravel()

        kg = ((10 ^ 6) * kc * sf_gg) / vd_gg

    # TODO: Check against Navid's code: swap [x2, x3, x4] for [x5, x6, x7] below?
    f = np.array([[-x[5] * q1 - fc01 + k12 * x[1] + egp0 * (1 - x[7])],     # x_0
                  [x[5] * q1 - k12 * x[1] - x[6] * x[1]],                   # x_1
                  [x[0] - x[2] / tmax_i],                                   # x_2
                  [x[2] / tmax_i - x[3] / tmax_i],                          # x_3
                  [x[3] / (tmax_i * vd_i) - ke * x[4]],                     # x_4
                  [-ka1 * x[5] + (sf1 * ka1) * x[4]],                       # x_5
                  [-ka2 * x[6] + (sf2 * ka2) * x[4]],                       # x_6
                  [-ka3 * x[7] + (sf3 * ka3) * x[4]]]).reshape(-1, )        # x_7

    return f


def get_plant_steady_state(target_glucose, model_parameters):
    """
    This function solves for the model states and basal insulin required
    to bring the virtual patient to a target_glucose level (e.g. 115 mg/dL)

    :param target_glucose: glucose in mg/dL to solve for at steady state
    :param model_parameters: parameters of the glucoregulatory model
    :return: States of the model at steady-state, States of the model at steady-state
    """

    v_dg = model_parameters[1, 0] if model_parameters.ndim == 2 else model_parameters[1, ]

    # Convert target glucose from mg/dl to mmol/kg
    set_point_init = target_glucose / 18 * v_dg

    # Find steady state conditions
    x_x0 = np.zeros(shape=[8, 1])
    xx = fsolve(func=solve_ss,
                x0=x_x0,
                args=(set_point_init, model_parameters))

    plant_states_ss = np.append(set_point_init, xx[1:, ]).reshape((8, 1))

    return plant_states_ss, xx


def get_single_hormone_model_parameters(body_weight):
    """
    This function defines the model parameters for a single-hormone glucoregulatory model.

    :param body_weight: subject body weight in kilograms
    :return: 15-element column vector of model parameters
    """

    # Glucose kinetics
    fc01 = 0.0097
    v_dg = 0.16
    k12 = 0.066
    a_g = 0.8
    tmax_g = 40
    egp0 = 0.0161

    # Insulin kinetics from Hovorka et al. 2004
    tmax_i = 55
    k_e = 0.138
    v_di = 0.12

    # Insulin dynamics from Wilinska et al. 2005
    ka1 = 0.006
    ka2 = 0.06
    ka3 = 0.03
    sf1 = 51.2e-4
    sf2 = 8.2e-4
    sf3 = 520e-4

    model_parameters = np.array([[fc01, v_dg, k12, a_g, tmax_g, egp0,
                                  tmax_i, k_e, v_di,
                                  ka1, ka2, ka3, sf1, sf2, sf3, body_weight]]).transpose()

    return model_parameters


def plant(x, model_parameters, m_e_piu=1.0, m_e_pgu=1.0, m_e_hgp=1.0, mode='default'):
    """
    This function implements the glucoregulatory model for state X and model parameters ModPar.
    The outputs of the model are in the form:               \n
    x(t+1) = Ad * x(t) + Bd * u(t) + Dd                     \n
    y(t+1) = Cd * x(t+1)                                    \n

    :param mode:
    :param m_e_hgp:
    :param m_e_pgu:
    :param m_e_piu:
    :param x: model state vector
    :param model_parameters: 15-element vector of model parameters
    :return: state-space matrices {a_d: system matrix, b_d: input matrix, c_d: output matrix, d_d: feedforward matrix}
    """

    t_s = 5     # seconds

    # Model parameters
    fc01, v_dg, k12, a_g, tmax_g, egp0, \
        tmax_i, k_e, v_di, \
        ka1, ka2, ka3, sf1, sf2, sf3, body_weight = model_parameters.ravel()

    # Exercise effect
    sf1 *= m_e_piu * m_e_pgu
    sf2 *= m_e_piu * m_e_pgu
    sf3 *= m_e_hgp

    kb1 = sf1 * ka1     # L / mU
    kb2 = sf2 * ka2     # L / mU
    kb3 = sf3 * ka3     # L / mU - min ^ -1

    # State variables
    q1, q2, s1, s2, i, x1, x2, x3 = x.ravel()
    g = q1 / v_dg

    # Disposal from Q1
    if g < 4.5:
        kc01 = fc01/(4.5 * v_dg)
        k_r = 0
        k_d = 0
    elif 4.5 <= g < 9:
        kc01 = 0
        k_r = 0
        k_d = fc01
    else:
        kc01 = 0
        k_r = 0.003
        k_d = fc01 - 0.003 * 9 * v_dg

    # Assemble state space representation
    a_p = np.array([[-x1 - kc01 - k_r,  k12,         0,         0,                 0,    -q1,  0,    -egp0],
                    [x1,                -(k12 + x2), 0,         0,                 0,    q1,   -q2,  0],
                    [0,                 0,           -1/tmax_i, 0,                 0,    0,    0,    0],
                    [0,                 0,           1/tmax_i,  -1/tmax_i,         0,    0,    0,    0],
                    [0,                 0,           0,         1/(tmax_i * v_di), -k_e, 0,    0,    0],
                    [0,                 0,           0,         0,                 kb1,  -ka1, 0,    0],
                    [0,                 0,           0,         0,                 kb2,  0,    -ka2, 0],
                    [0,                 0,           0,         0,                 kb3,  0,    0,    -ka3]])

    b_p = np.array([[0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])

    c_p = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])

    d_p = np.array([[x1 * q1 - k_d + egp0],
                    [-x1 * q1 + x2 * q2],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])

    # Solve system
    if mode == 'ZOH':
        # TODO: Need to speed up expm() (nearly 5 times slower than matlab!)
        s = expm(np.vstack((np.hstack((a_p, b_p)) * t_s,
                            np.zeros(shape=(1, 9)))))
        a_d = s[:8, :8]
        b_d = s[:8, 8:]
    else:
        a_d = a_p * t_s + np.identity(8)
        b_d = b_p * t_s
    c_d = c_p
    d_d = d_p * t_s

    return a_d, b_d, c_d, d_d


def prune_population(hovorka_parameters, body_weights):
    """
    This function will prune a virtual patient population by requiring that each of the patients,
    with parameters given by HovParams passes a set of physiologic tests.

    :param hovorka_parameters: Two-dimensional array of model parameters for each virtual patient
    :param body_weights: List of body weights (in kg) of each virtual patient
    :return:
    """

    # Initialize arrays, and indicate the time duration over which to run (nominal 1000 samples)
    n_par = hovorka_parameters.shape[0]
    revised_params_first_pass, revised_params_first_fail = np.empty(shape=(n_par, 0)), np.empty(shape=(n_par, 0))
    weights_data_first_pass, weights_data_first_fail = np.array([]), np.array([])
    revised_params_second_pass, revised_params_second_fail = np.empty(shape=(n_par, 0)), np.empty(shape=(n_par, 0))
    weights_data_second_pass, weights_data_second_fail = np.array([]), np.array([])
    n_sim = 1000                                                    # number of samples to simulate
    init_gluc = 160                                                 # Starting glucose level to run the system in to

    # Test 1: Withhold insulin and ensure glucose levels rise above 300 mg/dL
    for vp_hovorka_params, vp_bw in zip(hovorka_parameters.T, body_weights.T):

        vp_params = np.append(vp_hovorka_params[:-1], vp_bw)        # Select model parameters for virtual patient
        x_m = np.zeros((8, 1))                                      # Initialize the states of the model
        v_dg = vp_hovorka_params[1]                                 # Volume of distribution (Vdg)

        # Across n_sim time stamps, run the model presuming no insulin is given to the patient
        for t in range(1, n_sim + 1):
            a_pr, b_pr, c_pr, d_pr = plant(x_m, vp_params, mode='ZOH')
            u_insulin = 0                                           # Insulin (in U) given to the patient
            x_m = np.matmul(a_pr, x_m) + b_pr * u_insulin + d_pr
            y = c_pr * x_m

        # The virtual patient only passed this test if their glucose is greater than 300 mg/dL
        # after giving no insulin across the simulation window
        if (x_m[0, 0] * 18 / v_dg) >= 300:                                                          # Pass Test 1
            revised_params_first_pass = np.concatenate((revised_params_first_pass,
                                                        vp_hovorka_params[np.newaxis].T), axis=1)
            weights_data_first_pass = np.append(weights_data_first_pass, vp_bw)
        else:                                                                                       # Fail Test 1
            revised_params_first_fail = np.concatenate((revised_params_first_fail,
                                                        vp_hovorka_params[np.newaxis].T), axis=1)
            weights_data_first_fail = np.append(weights_data_first_fail, vp_bw)

    # Test 2: Administer a large amount of insulin and ensure glucose drops below 100 mg/dL
    for vp_hovorka_params, vp_bw in zip(revised_params_first_pass.T, weights_data_first_pass):

        # Initialize the states of the model
        ss_state, x_m = get_plant_steady_state(init_gluc, vp_hovorka_params[np.newaxis].T)
        x_m = x_m.reshape(8, 1)
        vp_params = np.append(vp_hovorka_params[:-1], vp_bw)        # Select model parameters for virtual patient
        v_dg = vp_hovorka_params[1]                                 # Volume of distribution (Vdg)

        # Across n_sim time stamps, run the model presuming no insulin is given to the patient
        for t in range(1, n_sim + 1):
            a_pr, b_pr, c_pr, d_pr = plant(x_m, vp_params, mode='ZOH')
            u_insulin = 1000 * (15 / 60 / vp_bw)                    # Dose a large amount of insulin (in U)
            x_m = np.matmul(a_pr, x_m) + b_pr * u_insulin + d_pr
            y = c_pr * x_m
            x_m[x_m < 0] = 0

        # The virtual patient only passed this test if their glucose drops below 100 mg/dL after receiving insulin
        if (x_m[0, 0] * 18 / v_dg) <= 100:                                                          # Pass Test 2
            revised_params_second_pass = np.concatenate((revised_params_second_pass,
                                                         vp_hovorka_params[np.newaxis].T), axis=1)
            weights_data_second_pass = np.append(weights_data_second_pass, vp_bw)
        else:                                                                                       # Fail Test 2
            revised_params_second_fail = np.concatenate((revised_params_second_fail,
                                                         vp_hovorka_params[np.newaxis].T), axis=1)
            weights_data_second_fail = np.append(weights_data_second_fail, vp_bw)

    return revised_params_second_pass, weights_data_second_pass, revised_params_second_fail, weights_data_second_fail