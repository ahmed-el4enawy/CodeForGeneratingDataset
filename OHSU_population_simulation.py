# ---------------------------------------------
# Authors: Clara Mosquera-Lopez, Valentina Roquemen, Gavin Young, Navid Resalat, Peter Jacobs
#         Oregon Health & Science University
# Date: November 22, 2022
# Updated for NH simulations on April 5, 2023
# File: datasetPop.py
# Function: This script tests the sample virtual patient population with glucose data for a sample meal scenario
#           using the model reported in Resalat et al. 2019. The glucose data shown here is computed with
#           open-loop basal insulin delivery. The insulin input can be replaced with any controllers' output
#           (MPC vs PID vs Fuzzy logic) for AP in-silico analyses.
#           Note: the states of the single hormone system are  [Q1 Q2 S1 S2 I x1 x2 x3]
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
#
# ---------------------------------------------
# Imports
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import multiprocessing
import time
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle

# OHSU simulator imports
import OHSUPOPsimulations.create_SHP as createSH
import OHSUPOPsimulations.test_SHP as testSH

# SIM imports
from OHSUPOPsimulations import SIM_options, SIM_generate_dosing_problems,SIM_load_virtualpop_data

from correction_bolus import IOB_Linear_Calculation,Correction_Ins_Alg

def OHSU_SIM_simulate_subject(simDays, initCGM, modParams0, bodyWeight,TDIR, basalTDIRRatio, mealScenario,
                              ICR_problem, BASAL_problem, CF_problem, sub_idx,late_bolus_min,CF,problem,
                              age = 35, sex = 'UNK',
                              initStates = None, Ts = 5):

    #%% Build simulation scenario
    modParams = np.append(modParams0,bodyWeight) # 16-element array

    # Select a sample meal scenario
    simTime = int(simDays * 1440 / Ts)                                  # Simulation samples

    mealVector = mealScenario

    #mealVector_snack_fast = np.zeros(len(mealVector))
    #mealVector_snack_slow = np.zeros(len(mealVector))

    IpVector = np.ones(len(mealVector))

    # Calculate prandial insulin boluses
    ICR = 1700 / TDIR / 3                               # ICR: Insulin to Carb Ratio
    mealBoluses = 1000 * IpVector * (mealVector / (ICR * ICR_problem)) / (bodyWeight * Ts)    # Pre-meal bolus insulin (mU/kg/min)

    mealBoluses = np.concatenate([np.zeros(late_bolus_min//5),mealBoluses])  # Late bolus

    # Calculate patient's constant basal insulin
    uBasal = BASAL_problem * np.array([(TDIR/basalTDIRRatio/ (24 * 60)) * 1000 / bodyWeight])     # basal insulin (mU/kg/min)
    #%% Calculate model initial conditions
    if initStates is not None:
        xmPlant = initStates
    else:
        xmPlant, ssIIRKgSubj = createSH.get_plant_steady_state(initCGM, modParams)
        Q1 = initCGM / 18 * modParams[1]        # conversion to (mmol/kg); (mg/dl) divided by 18 (mmol/L) times Vdg (L/kg)
        xmPlant[0] = Q1                         # Replace Q1 compartment with the starting glucose level

    #%% Simulate
    v_cgm_noise = []
    mealAmt, mealTime = [], [] # Keep for slow/fast carbs (call meal response twice)
    mealAmt_snack_fast, mealTime_snack_fast = [], []
    mealAmt_snack_slow, mealTime_snack_slow = [], []
    rescueCounter = 40 / Ts
    rescueTime = np.array([])
    UrPlant, UrModel, DrPlant, DrModel, rescueTriggerCounter = 0, 0, 0, 0, 0
    rescueInsulinAdj = np.ones((simTime, ))

    time_corr = [] # Correction boluses
    corr_val = []
    corr_switch = 0

    df = pd.DataFrame(columns=['time','dosing_problem_id','sim_day','initCGM','sim_hour_of_day','input_insulin','input_meal_carbs','output_glucose','output_cgm','state_Q1','state_Q2','state_S1','state_S2','state_I','state_X1', 'state_X2','state_X3','state_Ug'])
    df['time'] = np.arange(simTime)

    #MIN_CGM_past_nights_split_1 = []
    #MIN_CGM_past_nights_split_2 = [] #Keep history for NH prediction feature extraction

    is_change_hour = False
    for t in range(simTime):
        # Simulation hour of day
        sim_day = (t // (24 * 60 // Ts)) + 1
        sim_hour_of_day = (t % (24 * 60 // Ts))//(60//Ts)

        if t > 0:
            is_change_hour = sim_hour_of_day != df.loc[t-1,'sim_hour_of_day'] # This is to check for a change in hour. Useful to predict NH only once

        if SIM_options.SIMULATION_CORRECTION_BOLUS:
            Output = list(df['output_cgm'].values)

            Bolus_C, corr_switch, time_corr = Correction_Ins_Alg(Output,mealTime,t,SIM_options.SIMULATION_HIGH_TARGET,time_corr,bodyWeight,corr_switch,CF,
                               rescueTime,sim_hour_of_day,mealBoluses,Output)
            if Bolus_C>0:
                corr_val.append(Bolus_C)

        # Determine insulin dose
        uTotal = uBasal[0] if (t == 0) else rescueInsulinAdj[t] * uBasal[0] + mealBoluses[t] + Bolus_C# Don't give insulin boluses/rescue insulin if it is nighttime?

        # Meal response
        UgPlant, mealVectorPlant, mealTime, mealAmt = testSH.get_meal_response(mealAmt, mealTime, mealVector, t,
                                                                               modParams)  # from meal scenarios


        # Adjust insulin sensitivity for circadian changes
        if SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_CHANGE:
            sf_adjusted = testSH.apply_insulin_sensitivity_circadian_changes(t, sub_idx, modParams0[-3:],
                                                               SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_AMPLITUDE,
                                                               SIM_options.SIMULATION_CIRCADIAN_INSULIN_SENSITIVITY_PHASE,
                                                               SIM_options.SIMULATION_TS_MINUTES)
            modParams[-4:-1] = sf_adjusted

        # Generate plant output
        mePIU = 1
        mePGU = 1
        meHGP = 1

        # Plant model
        #TODO: Check with Nichole if this is correct
        Ap, Bp, Cp, Dp = createSH.plant(xmPlant, modParams, mePIU, mePGU, meHGP)
        xmPlant = np.matmul(Ap, xmPlant) + Bp * uTotal + Dp + (
            mealVectorPlant) * ((UgPlant) + UrPlant) * Ts  # in mmol/kg

        yPlant = float(np.matmul(Cp, xmPlant) * 18 / modParams[1])

        # Generate cgm with sensor noise
        if t > 0:
            yCGM, v_cgm_noise = testSH.add_noise_to_cgm(yPlant, df.loc[t-1,'output_glucose'], v_cgm_noise,
                                                        sim_day - 1)
        else:
            yCGM, v_cgm_noise = testSH.add_noise_to_cgm(yPlant, None, v_cgm_noise,sim_day - 1)

        # Rescue carbs
        if True:#not is_nighttime:
            UrPlant, UrModel, DrPlant, DrModel, rescueCounter, rescueTime, rescueTriggerCounter, rescueInsulinAdj = \
                testSH.get_rescue_carb(yPlant, UrPlant, UrModel, DrPlant, DrModel, rescueInsulinAdj, t, modParams,
                                       rescueCounter, rescueTime, rescueTriggerCounter)

        df.loc[t,'sim_day'] = sim_day
        df.loc[t,'sim_hour_of_day'] = sim_hour_of_day
        df.loc[t,'input_insulin'] = uTotal * bodyWeight * 60 / 1000

        df.loc[t,'input_meal_carbs'] = mealVector[t]

        df.loc[t,'output_glucose'] = yPlant
        df.loc[t,'output_cgm'] = yCGM[0]
        df.loc[t,[i for i in df if i.startswith('state')]] = np.append(xmPlant,(UgPlant) + UrPlant)

        df.loc[t,'correction_bolus'] = Bolus_C * bodyWeight * 60 / 1000
    df['initCGM'] = initCGM
    df.loc[rescueTime[rescueTime<simTime],'input_meal_carbs'] +=20

    '''
    plt.subplot(311)
    plt.plot(df.output_cgm)
    plt.title('CGM [mg/dL]')

    plt.subplot(312)
    plt.plot(df.input_insulin)
    plt.plot(df.correction_bolus,label='Correction bolus')
    plt.title('Insulin [U/h]')
    plt.legend()
    plt.subplot(313)
    plt.plot(df.input_meal_carbs)
    plt.title('Carbs [g]')

    plt.tight_layout()
    plt.savefig('test'+str(problem)+'.png')
    plt.close()
    #exit()
    '''

    return df, xmPlant
# ---------------------------------------------
# Parameters
# Model Parameters:
# Fc01: Non-insulin mediated glucose uptake above 4.5 mmol/L ([mmol/kg]/min)
# Vdg: Volume of distribution of glucose (L/kg)r
# k12: Rate constant for glucose transfer from Q2 to Q1 (min^-1)
# Ag: Carb bioavailability (unitless)
# tmaxG: time-to-maximum of carb absorption (min)
# EGP0: Endogenous glucose production maximum ([mmol/kg]/min)
# tmaxI: time-to-maximum of rapid-acting insulin absorption
# Ke: Elimination rate of insulin (min^-1)
# VdI: Volume of distribution of insulin (L/kg)
# ka1: Rate constant for elimination of insulin effect from x1 (min^-1)
# ka2: Rate constant for elimination of insulin effect from x2 (min^-1)
# ka3: Rate constant for elimination of insulin effect from x3 (min^-1)
# Sf1: Sensitivity factor for glucose distribution (x1) ([mU.L.min]^-2)
# Sf2: Sensitivity factor for insulin mediated glucose utilization (x2) ([mU.L.min]^-2)
# Sf3: Sensitivity factor for suppression of endogenous glucose production (x3) ([mU.L.min]^-1)
# Weight of the virtual subject (kg)
def simulate_scenarios(item):

    sub,subjectID,vp_data,modParamsOUT, body_weight,TDIR, basalTDIRRatio,this_sim_results_dir,late_bolus_min,cf = item
    #['S_042010','S_062003','S_037010','S_042002','S_042001','S_037011',
    # 'S_037009','S_037013','S_037005','S_037012','S_062006','S_042008',
    # 'S_042009','S_064006','S_062001','S_037005','S_062004','S_042001']
    subjects_train = ['001003','001004','001008','001012','001016','001017','001026','001032','001033','001034',
                '037001','037002','037003','037004','037006','037007','037008','037014']
    subjects_test = ['042006','042007','042011','042012','042013']
    subjects_validation = ['062005','064003','064005','064008','064011']
    subjects = subjects_train+subjects_test+subjects_validation

    if subjectID in subjects:
        print(sub)
        print("INFO: Processing virtual subject {}".format(subjectID))
        df_sub_sim = None

        # Get model params
        age = vp_data['subject'][sub]['age']

        # Get simulation params
        simDays = SIM_options.SIMULATION_N_DAYS_WEEKS * SIM_options.SIMULATION_N_WEEKS

        # Get mealScenario
        meal_data_days = SIM_load_virtualpop_data.get_meal_scenario_days(vp_data, sub, arms=SIM_options.SIMULATION_SCENARIOS_ARMS)

        if len(meal_data_days)!=0:
            n_meal_scenarios = 3
            mealScenarioOUT_array = SIM_load_virtualpop_data.get_meal_scenario_simulation(meal_data_days, n_meal_scenarios*simDays, random_sample=SIM_options.SIMULATION_SHUFFLE_MEALS)

            mealScenarioOUT_array = mealScenarioOUT_array.reshape(n_meal_scenarios,-1)

            for nsim,mealScenarioOUT in enumerate(mealScenarioOUT_array):
                # Get initCGM
                initCGM_array,mu = SIM_load_virtualpop_data.get_initCGM(vp_data,sub)

                for ii,initCGM in enumerate(initCGM_array):

                    # Simulation of no dosing problems
                    if SIM_options.SIMULATION_NO_DOSING_PROBLEMS:
                        ICR_problem = np.ones(simDays * ((24 * 60)//SIM_options.SIMULATION_TS_MINUTES))
                        CF_problem = ICR_problem.copy()
                        BASAL_problem = 1

                        # Run simulation with no dosing issues
                        # TODO: Implement saving data for post-hoc analysis
                        df, _ = OHSU_SIM_simulate_subject(simDays, initCGM, modParamsOUT, body_weight,
                                                                        TDIR, basalTDIRRatio, mealScenarioOUT,
                                                                        ICR_problem, BASAL_problem, CF_problem,sub,late_bolus_min,cf,0,
                                                                        age, 'UNK')
                        df['dosing_problem_id'] = 0
                        df_sub_sim = df.copy()

                    # Run simulations for dosing problems

                    for problem in SIM_options.SIMULATION_DOSING_PROBLEMS_DIC.keys():
                        if not problem in SIM_options.SIMULATION_DOSING_PROBLEMS_SELECTED:
                            continue
                        # print(problem)
                        # Get dosing problem vector for the entire simulation
                        probmatrix = SIM_generate_dosing_problems.generate_problems_matrix(
                            SIM_options.SIMULATION_DOSING_PROBLEMS_DIC)
                        BASAL_problem, ICR_problem_day, CF_problem_day = SIM_generate_dosing_problems.generate_sim_day_problems(
                            problem, probmatrix)
                        ICR_problem = np.tile(ICR_problem_day,simDays)
                        CF_problem = np.tile(CF_problem_day, simDays)

                        # Run simulation with dosing issues
                        df, _ = OHSU_SIM_simulate_subject(simDays, initCGM, modParamsOUT, body_weight,
                                                                    TDIR, basalTDIRRatio, mealScenarioOUT,
                                                                    ICR_problem, BASAL_problem, CF_problem, sub,late_bolus_min,cf,problem,
                                                                    age, 'UNK')
                        df['dosing_problem_id'] = problem
                        df_sub_sim = pd.concat([df_sub_sim, df.copy()], ignore_index=True, axis=0)

                        #sub_dict = {'initCGM':initCGM,
                        #    'df':df_sub_sim}

                    if not df_sub_sim is None:
                        #with open(this_sim_results_dir + subjectID + '_initCGM_'+str(int(initCGM))+'.pkl', 'wb') as f:
                        #    pickle.dump(sub_dict, f)
                        #print(df_sub_sim.groupby('dosing_problem_id').sum()['correction_bolus'])
                        #exit()
                        df_sub_sim.to_pickle(this_sim_results_dir + subjectID + 'mealScenario_'+str(nsim)+'_initCGM_'+str(ii)+"_"+str(int(initCGM))+'.pkl')
                        print(subjectID,initCGM)
        return mu

def main():
    now = datetime.now()
    # Directory for results
    results_dir = '/data/roquemen/datasets/JDRFProject/OHSUPOPsimulations/'
    for late_bolus_min in [0]:#range(5,50,5):
        for cf in [35*0.8,35,35*1.2]:
            this_sim_results_dir = results_dir + SIM_options.SIMULATION_ID + '/' + SIM_options.SIMULATION_TYPES[SIM_options.SIMULATION_TYPE_SELECTED] + '/bolusCorrectionFactor_'+str(int(cf))+'/' #+ now.strftime("%m%d%Y_%H%M%S")

            #this_sim_results_dir = results_dir + SIM_options.SIMULATION_ID + '/' + SIM_options.SIMULATION_TYPES[SIM_options.SIMULATION_TYPE_SELECTED] + '/boluslate_'+str(late_bolus_min)+'min/' #+ now.strftime("%m%d%Y_%H%M%S")
            print(this_sim_results_dir)
            # creating a new directory for simulation results
            Path(this_sim_results_dir).mkdir(parents=True, exist_ok=True)

            # Setup simulation

            # Load virtual subjects data
            vp_data = SIM_load_virtualpop_data.load_virtual_subjects("/data/roquemen/datasets/T1D/features/all_subject_t1dexi_pilot_06222023.pkl")

            # Get model params
            TDIR = SIM_options.SIMULATION_TDIR
            basalTDIRRatio = SIM_options.SIMULATION_BASAL_TDIR_RATIO
            modParamsOUT = SIM_options.SIMULATION_POPULATION_PARAMS[:15]
            body_weight = SIM_options.SIMULATION_POPULATION_PARAMS[-1]


            if SIM_options.SIMULATION_SERIAL:
                cgm_mu = []
                for sub in np.arange(len(vp_data['subject'])): # Change  in here
                    subjectID =  str(vp_data['subject'][sub]['subjectID'])
                    print(subjectID)
                    mu = simulate_scenarios([sub,subjectID,vp_data,modParamsOUT, body_weight,TDIR, basalTDIRRatio,this_sim_results_dir,late_bolus_min,cf])
                    if mu!=None:
                        cgm_mu.append(float(mu))
                print(np.nanmean(cgm_mu))
                print(np.nanstd(cgm_mu))

                #exit()
            else:
                # Define the number of parallel processes
                num_processes = 16#multiprocessing.cpu_count()

                # Create a pool of workers
                pool = multiprocessing.Pool(processes=num_processes)

                # Define the iterable for the loop
                items = []  # Your list of items to iterate over

                for sub in np.arange(len(vp_data['subject'])): # Change  in here
                    subjectID = str(vp_data['subject'][sub]['subjectID'])
                    items.append([sub,subjectID,vp_data,modParamsOUT, body_weight,TDIR, basalTDIRRatio,this_sim_results_dir,late_bolus_min,cf])
                    #simulate_scenarios(sub,subjectID,vp_data,modParamsOUT, body_weight,TDIR, basalTDIRRatio,this_sim_results_dir)

                # Apply the function to each item in parallel
                pool.map(simulate_scenarios, items)

                # Close the pool of workers
                pool.close()
                pool.join()

if __name__ == '__main__':
    main()