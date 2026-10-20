# ---------------------------------------------
# Authors: Nichole Tyler (Original in Matlab), Valentina Roquemen (Translate to python),
# Date: February 23, 2024
# ---------------------------------------------

import numpy as np

def IOB_Linear_Calculation(IIR_mat):

    Ts = 5
    IIR_mat = IIR_mat/12 # Convert to units / sample
    bolusind = np.where(IIR_mat!=0)[0]
    li = len(IIR_mat)

    # Calculate Linear Component
    if li <= 48:
        if  len(bolusind)>0:
            decayvals = np.zeros((1,li))
            timeval = li-bolusind # time since last bolus, up to 4 hrs.
            decayval = (48-timeval)/48
            decayvals[bolusind] = decayval
            iobvals = decayvals*IIR_mat
            IOB = np.sum(iobvals)
        else:
            IOB = 0

    else:
        IOB = np.zeros((li,1))
        for i in range(li):
            tempiir = IIR_mat[(i-48):i]
            ll = len(tempiir)
            bolusind =  np.where(tempiir!=0)[0]

            if len(bolusind)>0:
                decayvals = np.zeros(ll,1)
                timeval = ll-bolusind # time since last bolus, up to 4 hrs.
                decayval = (48-timeval)/48
                decayvals[bolusind] = decayval
                iobvals = decayvals*tempiir
                tempiob = np.sum(iobvals)
            else:
                tempiob = 0

            IOB[i] = tempiob

    return IOB

# This code checks for hyperglycemia and adds a correction bolus
# is your BG > 280, are you 1 hr out from a meal, and have you not given a
# correction in the past 4 hrs?

# Output= CGM in mg/dL
# kk: index time
# thour: Hour of the day
# corr_switch: 0,1
# CF = 35, 35*0.8, 35*1.2
def Correction_Ins_Alg(Output,meal_time,kk,high_target,time_corr,Weight,corr_switch,CF, time_resc,thour,u_bolus,cgm):
    if kk>0: # -1 matlab
        # Correction 4 hours after a meal
        if thour > 21 or thour < 7:
            thr_corr = 280
        else:
            thr_corr = 200

        # Test 1
        if (corr_switch == 0) and (Output[kk-1] > thr_corr) and len(meal_time)>0 and ((kk-meal_time[-1])>=48):

            uiob = u_bolus[kk-48:kk]
            IOB_vals= IOB_Linear_Calculation(uiob*Weight*60/1000) # feed in U/hr calculation

            DEval =np.sum(np.diff(cgm[kk-14:kk-1]))/(5*12)

            # CALCULATE INSULIN BOLUS using smart bolus calculator
            if (DEval > 3):
                CFval = (Output[kk-1] + 100 - high_target)/(CF) - IOB_vals
            elif (DEval > 2) and (DEval <= 3):
                CFval = (Output[kk-1] + 75 - high_target)/(CF) - IOB_vals
            elif (DEval > 1) and (DEval <= 2):
                CFval = (Output[kk-1] + 50 - high_target)/(CF) - IOB_vals
            elif (DEval < -1) and (DEval >= -2):
                CFval = (Output[kk-1] - 50 - high_target)/(CF) - IOB_vals
            elif (DEval < -2) and (DEval >= -3):
                CFval = (Output[kk-1] - 75 - high_target)/(CF) - IOB_vals
            elif (DEval < -3):
                CFval = (Output[kk-1] - 100 - high_target)/(CF) - IOB_vals
            else:
                CFval = (Output[kk-1] - high_target)/(CF) - IOB_vals

            if CFval <0:
                CFval = 0

            Bolus_C = CFval*1000/(Weight*5) #(mU/kg/min)
            # dont correct if the person had a rescue in the last hour.
            if len(time_resc)>0 and ((kk-time_resc[-1])<12):
                Bolus_C = 0

            # If the subject is sleeping they probably won't take the bolus. Remove the condition because we want to maximize the examples
            #if thour > 21 or thour < 7:
            #    if np.random.randint(2) < 0.50:
            #        Bolus_C = 0

            if Bolus_C > 0:
                time_corr.append(kk)
                corr_switch = 1

        else:
            Bolus_C = 0

        # Test 2
        # 3 Hours between correction boluses
        if  (len(time_corr)>0) and ((kk-time_corr[-1]) > 36):
           corr_switch = 0

    else:
        Bolus_C = 0
        corr_switch = 0

    return Bolus_C, corr_switch, time_corr
