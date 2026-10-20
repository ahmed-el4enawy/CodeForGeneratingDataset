from OHSUPOPsimulations import SIM_options

import pickle
import numpy as np
import pandas as pd


def load_virtual_subjects(pkl_file):
    vpop_data = pd.read_pickle(pkl_file)#pickle.load(file)

    for ss in np.arange(len(vpop_data['subject'])):
        for arm in SIM_options.SIMULATION_SCENARIOS_ARMS:
            vpop_data['subject'][ss][arm]['time'] = pd.to_datetime(vpop_data['subject'][ss][arm]['time_local']).dt.round(str(int(SIM_options.SIMULATION_TS_MINUTES)) + 'min')
            vpop_data['subject'][ss][arm]['meals'] = vpop_data['subject'][ss][arm]['carbs_mean']

    print("INFO: Virtual population data loaded")
    return vpop_data

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def get_initCGM(vpop_data, subjectIDX, arm = 'df'):
    sub_cgm_data = vpop_data['subject'][subjectIDX][arm][['time', 'cgm']].copy()
    sub_cgm_data['hour_of_day'] = sub_cgm_data['time'].dt.hour

    # Split at midnight 00:00
    rvalue_midnight, rstart_midnight, _ = find_runs(sub_cgm_data['hour_of_day'].values == 0)
    midnight_indices = rstart_midnight[rvalue_midnight == 1]

    cgm_midnight = sub_cgm_data['cgm'].values[midnight_indices][np.logical_and(sub_cgm_data['cgm'].values[midnight_indices]>40,sub_cgm_data['cgm'].values[midnight_indices]<400)]
    if len(cgm_midnight)<2:
        return np.array([])
    else:
        #print(cgm_midnight)
        # Get average CGM at 00:00
        # print(sub_cgm_data['cgm'].values[midnight_indices])
        if SIM_options.SIMULATION_N_CGM_INITIALIZATION==1:
            return np.nanmean(cgm_midnight)#sub_cgm_data['cgm'].values[midnight_indices])
        else:

            mu = np.nanmean(cgm_midnight)#sub_cgm_data['cgm'].values[midnight_indices])
            sigma = np.nanstd(cgm_midnight)#sub_cgm_data['cgm'].values[midnight_indices])

            initcgm =[]
            i = 0
            while i<SIM_options.SIMULATION_N_CGM_INITIALIZATION:
                cgm = np.random.normal(mu, sigma, 1)
                 # TODO: mu, sigmaprint()
                if cgm>=40 and cgm<=400:
                    if cgm[0] in initcgm:
                        pass
                    else:
                        initcgm.append(cgm[0])
                        i+=1

            return np.array(initcgm),mu

def get_meal_scenario_days(vpop_data, subjectIDX, arms = SIM_options.SIMULATION_SCENARIOS_ARMS):
    #TODO: Load scenarios, not meals
    sub_meal_data_days = []
    for arm in arms:
        # Load scenario meal data
        sub_meal_data = vpop_data['subject'][subjectIDX][arm][['time','meals']].copy()
        sub_meal_data['hour_of_day'] = sub_meal_data['time'].dt.hour

        # Split at midnight 00:00
        rvalue_midnight, rstart_midnight, _ = find_runs(sub_meal_data['hour_of_day'].values == 0)
        midnight_indices = rstart_midnight[rvalue_midnight == 1]
        sub_meal_data_by_arm_and_day = np.split(sub_meal_data['meals'].values, midnight_indices)

        # Pad first and last day with zeros if needed

        for ii in np.arange(len(sub_meal_data_by_arm_and_day)):
            pad_width = ((24 * 60) // SIM_options.SIMULATION_TS_MINUTES) - len(sub_meal_data_by_arm_and_day[ii])
            if pad_width < 0:
                sub_meal_data_by_arm_and_day[ii] = sub_meal_data_by_arm_and_day[ii][
                                                   -((24 * 60) // SIM_options.SIMULATION_TS_MINUTES):]
                continue
            if (ii == 0) and (
                    len(sub_meal_data_by_arm_and_day[ii]) < (24 * 60) // SIM_options.SIMULATION_TS_MINUTES):  # First
                sub_meal_data_by_arm_and_day[ii] = np.pad(sub_meal_data_by_arm_and_day[ii], pad_width=(pad_width, 0),
                                                          constant_values=0)
            elif (ii == len(sub_meal_data_by_arm_and_day) - 1) and (
                    len(sub_meal_data_by_arm_and_day[ii]) < (24 * 60) // SIM_options.SIMULATION_TS_MINUTES):
                sub_meal_data_by_arm_and_day[ii] = np.pad(sub_meal_data_by_arm_and_day[ii], pad_width=(0, pad_width),
                                                          constant_values=0)
            else:
                sub_meal_data_by_arm_and_day[ii] = np.pad(sub_meal_data_by_arm_and_day[ii],
                                                          pad_width=(pad_width // 2, pad_width - (pad_width // 2)),
                                                          constant_values=0)
            # Filter real- meal scenarios
            sub_meal_data_by_arm_and_day[ii] = np.nan_to_num(sub_meal_data_by_arm_and_day[ii],nan=0)

            count = len(sub_meal_data_by_arm_and_day[ii][sub_meal_data_by_arm_and_day[ii]!=0])
            amount = sub_meal_data_by_arm_and_day[ii].sum()
            if count>=SIM_options.SIMULATION_count_min and count<=SIM_options.SIMULATION_count_max:
                if amount>=SIM_options.SIMULATION_amount_min and amount<=SIM_options.SIMULATION_amount_max:
                    #if not(((count<SIM_options.DS_count_min | count>SIM_options.DS_count_max) or
                    #(amount<SIM_options.DS_amount_min |amount>SIM_options.DS_amount_max))):
                    sub_meal_data_days.append(sub_meal_data_by_arm_and_day[ii])

        #sub_meal_data_days += sub_meal_data_by_arm_and_day

    return sub_meal_data_days

def get_meal_scenario_simulation(meal_data_days, total_days, random_sample = True):

    meal_data_days_concat = np.stack(meal_data_days, axis=0)
    if len(meal_data_days) >= total_days:
        if random_sample:
            meals_simulation = meal_data_days_concat[np.random.choice(np.arange(len(meal_data_days)), size=total_days, replace=False),:]
        else:
            meals_simulation = meal_data_days_concat[:total_days,:]
    else:
        meal_data_days_resized = np.resize(np.asarray(meal_data_days_concat),new_shape=(total_days,meal_data_days_concat.shape[1]))
        if random_sample: # Sample with replacement
            meals_simulation = meal_data_days_resized[
                np.random.choice(np.arange(len(meal_data_days_resized)), size=total_days, replace=True),:]
        else: # Repeat until size
            meals_simulation = meal_data_days_resized

    return meals_simulation.flatten()