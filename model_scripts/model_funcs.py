import pandas as pd
import numpy as np
import numpy.matlib as mb
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
import glob as glob
import seaborn as sns
import pickle
import os
import re


def gen_model_plots(c_tState, c_tTask, c_tProp, c_state, c_task, c_color, plot_filename, tResp):
    err_colors = ['grey','orange','red','blue']
    err_ys = [-0.1, -0.125, -0.15, -0.175]
    # make MODEL ESTIMATE plots
    plt.rcParams["figure.figsize"] = (15,6)
    fig, axs = plt.subplots(3,1) # rows, columns
    
    axs[0].plot(c_tState, 'black', alpha=0.5)
    axs[0].plot(c_state, 'blue', alpha=0.5)
    axs[0].plot(0.5*np.ones(200), 'k--', alpha=0.25)
    for x_coord in range(len(c_tState)):
        axs[0].plot(x_coord, err_ys[tResp[x_coord]], color=err_colors[tResp[x_coord]], marker='o', alpha=0.25)
    axs[0].set_ylabel("STATE", rotation=0, fontsize=10, labelpad=20)
    
    axs[1].plot(c_tTask, 'black', alpha=0.5)
    axs[1].plot(c_task, 'green', alpha=0.5)
    axs[1].plot(0.5*np.ones(200), 'k--', alpha=0.25)
    for x_coord in range(len(c_tTask)):
        axs[1].plot(x_coord, err_ys[tResp[x_coord]], color=err_colors[tResp[x_coord]], marker='o', alpha=0.25)
    axs[1].set_ylabel("TASK", rotation=0, fontsize=10, labelpad=20)
    
    axs[2].plot(c_tProp, 'black', alpha=0.5)
    axs[2].plot(c_color, 'gold', alpha=0.5)
    axs[2].plot(0.5*np.ones(200), 'k--', alpha=0.25)
    for x_coord in range(len(c_tProp)):
        axs[2].plot(x_coord, err_ys[tResp[x_coord]], color=err_colors[tResp[x_coord]], marker='o', alpha=0.25)
    axs[2].set_ylabel("COLOR", rotation=0, fontsize=10, labelpad=20)
    
    plt.show()
    plt.savefig(plot_filename)
    plt.close()

def calc_entropy(cur_model_sE, cE):
    jd = np.array([cE * cur_model_sE[:,1],  (1 - cE) * cur_model_sE[:,1], 
                   cE * (1 - cur_model_sE[:,1]),  (1 - cE) * (1 - cur_model_sE[:,1])])
    t_entropy = [] #entorpy of dist
    t_entropy_change = [0.0] #signed change of entropy from previous trial
    for t in range(len(cur_model_sE)):
        t_entropy.append(entropy(jd[:,t].flatten()))
        if t>0:
            t_entropy_change.append(entropy(jd[:,t].flatten())-entropy(jd[:,t-1].flatten()))
    t_entropy = np.array(t_entropy)
    t_entropy_change = np.array(t_entropy_change)
    return t_entropy, t_entropy_change




# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -            Computational Model Code           - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

"""
MPE MODEL
maximum posterior estimator of joint distribution of state and hyperparameters

input:
tState: trial wise task-set (0: color 0 = face, color 1 = scene; 1: color 0 = scene, color 1 = face)
tProp: trial-wise proportion of dots (0 - 1, for the first color)
tResp: trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task

thetas:
lfsp: logistic function slope parameter (color uncertainty)
lfip: logistic function intercept parameter (color bias)
fter: face task error rate
ster: scene task error rate
diff_at: diffusion across trials

output:
jd: joint distribution of parameters at the end of experiment
sE: trial-wise estimate of state, encoding probability of state 0
tE: trial-wise estimate of task, encoding probability of task 0
mDist: marginal distribution for the 4 thetas,
rRange: values of the thetas
"""

#function [jd, sE, tE, mDist, pRange] = MPEModel(tState, tProp, tResp)
def MPEModel(tState, tProp, tResp):
    
    sE = []
    tE = []
    cE = []

    #start with uniform prior
    pRange = {'lfsp':np.linspace(1.5, 4.5, 61, endpoint=True), 
            'lfip':np.array([-3, -2.85, -2.7, -2.55, -2.4, -2.25, -2.1, -1.95, -1.8, -1.65, -1.5, -1.4, -1.3, -1.2, -1.10, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05,
                             0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 1.95, 2.10, 2.25, 2.4, 2.55, 2.7, 2.85, 3]),
            'fter':np.linspace(0, 0.25, 26, endpoint=True), 
            'ster':np.linspace(0, 0.25, 26, endpoint=True), 
            'diff_at':np.linspace(0, 0.6, 31, endpoint=True)}
    dim = np.array( [2, len(pRange['lfsp']),  len(pRange['lfip']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )
    
    total = 1
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total # the division of total is to normalize values into probability
    
    #likelihood of getting correct response
    ll = np.zeros(jd.shape)

    tProp = np.log(tProp / (1 - tProp))

    for it in range(len(tState)):
        if (it % 20) == 0:
            print(str(it), ' trials have been simulated.')
        
        #diffusion of joint distribution over trials
        for i4 in range(dim[5]):
            x = (jd[0, :, :, :, :, i4] + jd[1, :, :, :, :, i4]) / 2 # average of 2task state

            jd[0, :, :, :, :, i4] = jd[0, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, :, i4] = jd[1, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        
        sE.append(np.sum(jd[0, :, :, :, :, :]))
        
        # first color logit greater than 0 means it is dominant, less than 0 means other color dominant
        # below is the mapping between color ratio, task state, and task
        # tState = 0, tProp<0 : tTask = 1 ... state-0 and Yellow
        # tState = 0, tProp>0 : tTask = 0 ... state-0 and Red
        # tState = 1, tProp<0 : tTask = 0 ... state-1 and Yellow
        # tState = 1, tProp>0 : tTask = 1 ... state-1 and Red
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0 # state-1 and Yellow  OR  state-0 and Red
        else:
            tTask = 1
        # here marginalize theta1
        S_Theta1 = np.sum(jd, axis=(5,4,3)) # will be state x lsfp x lifp
        
        tE.append(0)
        cE.append(0)
        for i0 in range(dim[0]):
            for i1 in range(dim[1]):
                for i1b in range(dim[2]):
                    theta1 = np.exp(pRange['lfsp'][i1])
                    theta1b = pRange['lfip'][i1b] # make intercept proportional to slope
                    pColor = 1 / (1 + np.exp((-1*theta1*tProp[it])+theta1b))
                    cE[-1] = pColor
                    if i0 == 0:
                        pTask0 = pColor
                    else:
                        pTask0 = 1 - pColor
                    
                    tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1, i1b] #IDK... test it I guess
                    for i2 in range(dim[3]):
                        for i3 in range(dim[4]):
                            # pP is the likelihood, change this if there is only one type of error
                            # this is for separated response error and task error
                            if tTask == 0:
                                pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0), 1]
                            else:
                                pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0, 1]
                            
                            #posterior, jd now is prior
                            if tResp[it] < 3:
                                ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] * pP[int(tResp[it])]
                            else:
                                ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] 

                            # #this is for only one type of error
                            # if tTask == 0:
                            #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                            # else:
                            #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                            
                            # #posterior, jd now is prior
                            # if tResp[it] == 0:
                            #     ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] * pP[0]
                            # else:
                            #     ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] * pP[1]
                            
        #normalize
        jd = ll / np.sum(ll)
    
        mDist={}
        mDist['lfsp'] = np.sum(jd, axis=(0, 2, 3, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 2), 0))    # 4 3 2 0
        mDist['lfip'] = np.sum(jd, axis=(0, 1, 3, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 1), 0))    # 4 3 2 0
        mDist['fter'] = np.sum(jd, axis=(0, 1, 2, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 2), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.sum(jd, axis=(0, 1, 2, 3, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 3), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.sum(jd, axis=(0, 1, 2, 3, 4)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 1), 0)) # 3 2 1 0
    
    return jd, sE, tE, mDist, pRange, cE


"""
MODEL BELIEF GENERATION using optimal theta parameters

input:
thetas: [ lfsp  lfip  fter  ster  diff ] (see list below)
tState: trial wise task-set (0: color 0 = face, color 1 = scene; 1: color 0 = scene, color 1 = face)
tProp: trial-wise proportion of dots (0 - 1, for the first color)
tResp: trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task
use_pState: whether we should use a probabilistic state or dichotomous state belief
use_pPercept: whether we should use a probabilistic color or dichotomous color belief

thetas:
lfsp: logistic function slope parameter (color uncertainty)
lfip: logistic function intercept parameter (color bias)
fter: face task error rate
ster: scene task error rate
diff_at: diffusion across trials

output:
sE: trial-wise estimate of state, encoding probability of state 0
tE: trial-wise estimate of task, encoding probability of task 0
cE: trial-wise estimate of color, encoding probability of color 0
"""

# function [sE, tE, cE] = ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
def ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept):
        
    tProp = np.log(tProp / (1 - tProp))
    tE = np.zeros(tState.shape)
    sE = np.zeros((tState.shape[0], 2))
    cE = np.zeros(tState.shape)

    pS = np.array([0.5, 0.5])
    tmpPS = np.array([0.5, 0.5])

    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        
        pS[0] = pS[0] * (1 - thetas[4]) + thetas[4] * 0.5
        pS[1] = pS[1] * (1 - thetas[4]) + thetas[4] * 0.5
        
        if not(use_pState):
            if pS[0] > pS[1]:
                pS[0] = 1
                pS[1] = 0
            else:
                pS[0] = 0
                pS[1] = 1
        
        #first color logit greater than 0 means it is dominant, less than 0
        #means other color dominant
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1

        theta1 = np.exp(thetas[0])
        theta1b = thetas[1]
        #pColor = 1 / (1 + np.exp((-1*theta1) * tProp[it]))
        pColor = 1 / (1 + np.exp((-1*theta1*tProp[it])+theta1b))
        
        if not(use_pPercept):
            # Making color perception deterministic
            if pColor > 0.5:
                pColor = 1
            else:
                pColor = 0
                
        cE[it] = pColor
        
        tE[it] = pColor * pS[0] + (1 - pColor) * pS[1]
        #  1 =     0        0          1           1 
        #  0 =     1        0          0           1  
        #  0 =     0        1          1           0 
        #  1 =     1        1          0           0  

        # -- add code to check if no response made
        if tResp[it] == 3:
            if it > 0:
                sE[it, 0] = sE[it-1, 0]
                sE[it, 1] = sE[it-1, 0]
            else:
                sE[it, 0] = 0.5
                sE[it, 1] = 0.5
            continue # no response... skip updating

        for i0 in range(2):        
            if i0 == 0:
                pTask0 = pColor
            else:
                pTask0 = 1 - pColor
        
            if tTask == 0:
                pP = [pTask0 * (1 - thetas[2]), pTask0 * thetas[2], (1 - pTask0), 0.5]
            else:
                pP = [(1 - pTask0) * (1 - thetas[3]), (1 - pTask0) * thetas[3], pTask0, 0.5]
            
            #update posterior P(S|R) = P(R|C = 0, S) * P(C = 0) * P(S) + P(R|C = 1, S) * P(C = 1) * P(S)
            if pS[i0] == 0:
                tmpPS[i0] = 0.001 * pP[int(tResp[it])]
            elif pS[i0] == 1:
                tmpPS[i0] = 0.999 * pP[int(tResp[it])]
            else:
                tmpPS[i0] = pS[i0] * pP[int(tResp[it])]
        
        #if both likelihood drops to 0, randomly initialize state estimate
        if np.abs(np.sum(tmpPS)) < 1e-9:
            tmpPS[0] = np.random.rand()
            tmpPS[1] = 1 - tmpPS[0]
        else:
            tmpPS = tmpPS / np.sum(tmpPS)
        pS = tmpPS
        sE[it, 0] = tmpPS[0]
        sE[it, 1] = tmpPS[1]
    return sE, tE, cE



# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -        Control Computational Model Code       - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

"""
CONTROL MODEL(S)
maximum posterior estimator of joint distribution of state and hyperparameters

input:
tState: trial wise task-set (0: color 0 = face, color 1 = scene; 1: color 0 = scene, color 1 = face)
tProp: trial-wise proportion of dots (0 - 1, for the first color)
tResp: trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task
use_pState: whether we should use a probabilistic state or dichotomous state belief
use_pPercept: whether we should use a probabilistic color or dichotomous color belief

thetas:
lfsp: logistic function slope parameter (color uncertainty)
lfip: logistic function intercept parameter (color bias)
fter: face task error rate
ster: scene task error rate
diff_at: diffusion across trials

output:
jd: joint distribution of parameters at the end of experiment
sE: trial-wise estimate of state, encoding probability of state 0
tE: trial-wise estimate of task, encoding probability of task 0
mDist: marginal distribution for the 4 thetas,
rRange: values of the thetas
"""

#function [jd, sE, tE, mDist, pRange] = ControlModel(tState, tProp, tResp, use_pState, use_pPercept)
def MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept):
    
    sE = []
    tE = []
    cE = []

    #start with uniform prior
    pRange = {'lfsp':np.linspace(1.5, 5, 71, endpoint=True), 
            'lfip':np.array([-3, -2.85, -2.7, -2.55, -2.4, -2.25, -2.1, -1.95, -1.8, -1.65, -1.5, -1.4, -1.3, -1.2, -1.10, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05,
                             0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 1.95, 2.10, 2.25, 2.4, 2.55, 2.7, 2.85, 3]),
            'fter':np.linspace(0, 0.3, 31, endpoint=True), 
            'ster':np.linspace(0, 0.3, 31, endpoint=True), 
            'diff_at':np.linspace(0, 0.9, 46, endpoint=True)}
    dim = np.array( [2, len(pRange['lfsp']),  len(pRange['lfip']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )
    
    total = 1
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total # the division of total is to normalize values into probability
    
    #likelihood of getting correct response
    ll = np.zeros(jd.shape)

    tProp = np.log(tProp / (1 - tProp))

    for it in range(len(tState)):
        if (it % 20) == 0:
            print(str(it), ' trials have been simulated.')
        
        #diffusion of joint distribution over trials
        for i4 in range(dim[5]):
            x = (jd[0, :, :, :, :, i4] + jd[1, :, :, :, :, i4]) / 2 # average of 2task state

            jd[0, :, :, :, :, i4] = jd[0, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, :, i4] = jd[1, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        
        sE.append(np.sum(jd[0, :, :, :, :, :]))
        
        # first color logit greater than 0 means it is dominant, less than 0 means other color dominant
        # below is the mapping between color ratio, task state, and task
        # tState = 0, tProp<0 : tTask = 1 ... state-0 and Yellow
        # tState = 0, tProp>0 : tTask = 0 ... state-0 and Red
        # tState = 1, tProp<0 : tTask = 0 ... state-1 and Yellow
        # tState = 1, tProp>0 : tTask = 1 ... state-1 and Red
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0 # state-1 and Yellow  OR  state-0 and Red
        else:
            tTask = 1
        # here marginalize theta1 and theta1b
        S_Theta1 = np.sum(jd, axis=(5,4,3)) # will be state x lsfp x lifp
        if not(use_pState):
            # Making task-set decisions deterministic
            for i1 in range(S_Theta1.shape[1]):
                for i1b in range(S_Theta1.shape[2]):
                    if S_Theta1[0, i1, i1b] > S_Theta1[1, i1, i1b]:
                        S_Theta1[0, i1, i1b] += S_Theta1[1, i1, i1b]
                        S_Theta1[1, i1, i1b] = 0
                    else:
                        S_Theta1[0, i1, i1b] = 0
                        S_Theta1[1, i1, i1b] += S_Theta1[0, i1, i1b]
        
        tE.append(0)
        cE.append(0)
        for i0 in range(dim[0]):
            for i1 in range(dim[1]):
                for i1b in range(dim[2]):
                    theta1 = np.exp(pRange['lfsp'][i1])
                    theta1b = pRange['lfip'][i1b] # make intercept proportional to slope
                    pColor = 1 / (1 + np.exp((-1*theta1*tProp[it])+theta1b))
                    if not(use_pPercept):
                        # Making color perception deterministic
                        if pColor > 0.5:
                            pColor = 1
                        else:
                            pColor = 0
                    cE[-1] = pColor
                    if i0 == 0:
                        pTask0 = pColor
                    else:
                        pTask0 = 1 - pColor
                        
                    tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1, i1b] #IDK... test it I guess
                    for i2 in range(dim[3]):
                        for i3 in range(dim[4]):
                            #pP is the likelihood, change this if there is only one type of error
                            # # this is for separated response error and task error
                            # if tTask == 0:
                            #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0)]
                            # else:
                            #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0]
                            
                            # #posterior, jd now is prior
                            # ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[int(tResp[it])]

                            #this is for only one type of error
                            if tTask == 0:
                                pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                            else:
                                pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                            
                            #posterior, jd now is prior
                            if tResp[it] == 0:
                                ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] * pP[0]
                            else:
                                ll[i0, i1, i1b, i2, i3, :] = jd[i0, i1, i1b, i2, i3, :] * pP[1]
                            
        #normalize
        jd = ll / np.sum(ll)
        #all_jd[it,:,:] = np.sum(jd, axis=(2,3,4))
    
        mDist={}
        mDist['lfsp'] = np.sum(jd, axis=(0, 2, 3, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 2), 0))    # 4 3 2 0
        mDist['lfip'] = np.sum(jd, axis=(0, 1, 3, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 1), 0))    # 4 3 2 0
        mDist['fter'] = np.sum(jd, axis=(0, 1, 2, 4, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 2), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.sum(jd, axis=(0, 1, 2, 3, 5)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 3), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.sum(jd, axis=(0, 1, 2, 3, 4)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 1), 0)) # 3 2 1 0
    
    return jd, sE, tE, mDist, pRange, cE


"""
Jan MODEL
maximum posterior estimator of joint distribution of state and hyperparameters
* this model assumes a random color guess when the color proportion is very ambiguous & no state updating when color proportion is very ambiguous

input:
tState: trial wise task-set (0: color 0 = face, color 1 = scene; 1: color 0 = scene, color 1 = face)
tProp: trial-wise proportion of dots (0 - 1, for the first color)
tResp: trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task

thetas:
cub: upper bound of color guessing range
clb: lower bound of color guessing range
fter: face task error rate
ster: scene task error rate
diff_at: diffusion across trials

output:
jd: joint distribution of parameters at the end of experiment
sE: trial-wise estimate of state, encoding probability of state 0
tE: trial-wise estimate of task, encoding probability of task 0
mDist: marginal distribution for the 4 thetas,
rRange: values of the thetas
"""

#function [jd, sE, tE, mDist, pRange] = MPEModel(tState, tProp, tResp)
def JanModel(tState, tProp, tResp):
    
    sE = []
    tE = []
    cE = []

    #start with uniform prior
    pRange = {'cub':np.linspace(-0.1, 0.21, 32, endpoint=True), 
              'clb':np.linspace(-0.21, 0.1, 32, endpoint=True), 
              'fter':np.linspace(0, 0.2, 21, endpoint=True), 
              'ster':np.linspace(0, 0.2, 21, endpoint=True), 
              'diff_at':np.linspace(0, 0.9, 46, endpoint=True)}
    dim = np.array( [2, len(pRange['cub']), len(pRange['clb']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )
    
    total = 1
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total # the division of total is to normalize values into probability
    
    #likelihood of getting correct response
    ll = np.zeros(jd.shape)

    tProp = np.log(tProp / (1 - tProp))

    for it in range(200): #len(tState)):
        if (it % 10) == 0:
            print(str(it), ' trials have been simulated.')
        
        #diffusion of joint distribution over trials
        for i4 in range(dim[5]):
            x = (jd[0, :, :, :, :, i4] + jd[1, :, :, :, :, i4]) / 2 # average of 2task state

            jd[0, :, :, :, :, i4] = jd[0, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, :, i4] = jd[1, :, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        sE.append(np.sum(jd[0, :, :, :, :, :]))

        # first color logit greater than 0 means it is dominant, less than 0 means other color dominant
        # below is the mapping between color ratio, task state, and task
        # tState = 0, tProp<0 : tTask = 1 ... state-0 and Yellow
        # tState = 0, tProp>0 : tTask = 0 ... state-0 and Red
        # tState = 1, tProp<0 : tTask = 0 ... state-1 and Yellow
        # tState = 1, tProp>0 : tTask = 1 ... state-1 and Red
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0 # state-1 and Yellow  OR  state-0 and Red
        else:
            tTask = 1
        # here marginalize theta1
        S_Theta1 = np.sum(jd, axis=(5,4,3)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 2)) # double check that axes are correct
        # Making task-set decisions deterministic
        for i1a in range(S_Theta1.shape[1]):
            for i1b in range(S_Theta1.shape[2]):
                if S_Theta1[0, i1a, i1b] > S_Theta1[1, i1a, i1b]:
                    S_Theta1[0, i1a, i1b] += S_Theta1[1, i1a, i1b]
                    S_Theta1[1, i1a, i1b] = 0
                else:
                    S_Theta1[0, i1a, i1b] = 0
                    S_Theta1[1, i1a, i1b] += S_Theta1[0, i1a, i1b]
        
        tE.append(0)
        cE.append(0)
        for i0 in range(dim[0]):
            for i1a in range(dim[1]):
                theta1 = pRange['cub'][i1a] # color upper bound (for random guessing)
                for i1b in range(dim[2]):
                    theta2 = pRange['clb'][i1b] # color lower bound (for random guessing)
                    if ((tProp[it] < theta2) | (tProp[it] > theta1)):
                        if tProp[it] > 0:
                            pColor =  1 # 1 / (1 + np.exp(-1 * (20) * tProp[it])) # push to 0 or 1
                        else: 
                            pColor = 0
                    else:
                        pColor = np.random.choice([1, 0]) # 0.5 + np.random.choice([-0.01, 0.01]) # random guess with minor jitter to push above or below 0.5
                    cE[-1] = pColor
                    if i0 == 0:
                        pTask0 = pColor
                    else:
                        pTask0 = 1 - pColor
                    
                    #print(tE[-1] + pTask0 * S_Theta1[i0, i1])
                    tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1a, i1b] # * S_Theta2[i0, i1b] #IDK... test it I guess
                    for i2 in range(dim[3]):
                        for i3 in range(dim[4]):
                            # #pP is the likelihood, change this if there is only one type of error
                            # # this is for separated response error and task error
                            # if tTask == 0:
                            #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0)]
                            # else:
                            #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0]
                            
                            # #posterior, jd now is prior
                            # ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] * pP[int(tResp[it])]

                            #this is for only one type of error
                            if tTask == 0:
                                pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                            else:
                                pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                            
                            #posterior, jd now is prior
                            if it==0:
                                # have to update first trial so that it's not 0.5 the whole time
                                if tResp[it] == 0:
                                    ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] * pP[0]
                                else:
                                    ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] * pP[1]
                            else:
                                if ((tProp[it] < theta2) | (tProp[it] > theta1)):
                                    if np.sum(jd[1, i1a, i1b, i2, i3, :]) > np.sum(jd[0, i1a, i1b, i2, i3, :]):
                                        jd[0, i1a, i1b, i2, i3, :] += jd[1, i1a, i1b, i2, i3, :]
                                        jd[1, i1a, i1b, i2, i3, :] = 0
                                    else:
                                        jd[0, i1a, i1b, i2, i3, :] = 0
                                        jd[1, i1a, i1b, i2, i3, :] += jd[0, i1a, i1b, i2, i3, :]
                                else:
                                        ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] # no updating?
                            # if tResp[it] == 0:
                            #     ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] * pP[0]
                            # else:
                            #     ll[i0, i1a, i1b, i2, i3, :] = jd[i0, i1a, i1b, i2, i3, :] * pP[1]
                        
        #normalize
        jd = ll / np.sum(ll)
        #all_jd[it,:,:] = np.sum(jd, axis=(2,3,4))
    
        mDist={}
        #mDist['lfsp'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 0))    # 4 3 2 0
        mDist['cub'] = np.sum(jd, axis=(5,4,3,2,0)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 2), 0))
        mDist['clb'] = np.sum(jd, axis=(5,4,3,1,0)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 3), 1), 0))
        mDist['fter'] = np.sum(jd, axis=(5,4,2,1,0)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 4), 2), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.sum(jd, axis=(5,3,2,1,0)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 5), 3), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.sum(jd, axis=(4,3,2,1,0)) # np.squeeze(np.sum(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 1), 0)) # 3 2 1 0
    
    return jd, sE, tE, mDist, pRange, cE


# thetas = [ cub  clb  fter  ster  diff ]
def Jan_ModelBeliefGeneration(thetas, tState, tProp, tResp):
        
    tProp = np.log(tProp / (1 - tProp))
    tE = np.zeros(tState.shape)
    sE = np.zeros((tState.shape[0], 2))
    cE = np.zeros(tState.shape)

    pS = np.array([0.5, 0.5])
    tmpPS = np.array([0.5, 0.5])

    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        
        pS[0] = pS[0] * (1 - thetas[4]) + thetas[4] * 0.5
        pS[1] = pS[1] * (1 - thetas[4]) + thetas[4] * 0.5
        
        if pS[0] > pS[1]:
            pS[0] = 1
            pS[1] = 0
        else:
            pS[0] = 0
            pS[1] = 1

        #first color logit greater than 0 means it is dominant, less than 0 means other color dominant
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1
        
        theta1_upp = thetas[0]
        theta1_low = thetas[1]
        if ((tProp[it] < theta1_low) | (tProp[it] > theta1_upp)):
            if tProp[it] > 0:
                pColor =  1 # 1 / (1 + np.exp(-1 * (20) * tProp[it])) # push to 0 or 1
            else: 
                pColor = 0
        else:
            pColor = np.random.choice([0, 1]) # 0.5 + np.random.choice([-0.01, 0.01]) # random guess
        
        cE[it] = pColor
        
        tE[it] = pColor * pS[0] + (1 - pColor) * pS[1]
        #  1 =     0        0          1           1 
        #  0 =     1        0          0           1  
        #  0 =     0        1          1           0 
        #  1 =     1        1          0           0  

        # -- add code to check if no response made
        if tResp[it] == 3:
            if it > 0:
                sE[it, 0] = sE[it-1, 0]
                sE[it, 1] = sE[it-1, 0]
            else:
                sE[it, 0] = 0.5
                sE[it, 1] = 0.5
            continue # no response... skip updating

        for i0 in range(2):        
            if i0 == 0:
                pTask0 = pColor
            else:
                pTask0 = 1 - pColor

            if tTask == 0:
                pP = [pTask0 * (1 - thetas[2]), pTask0 * thetas[2], (1 - pTask0), 1] # 2 = face error
            else:
                pP = [(1 - pTask0) * (1 - thetas[3]), (1 - pTask0) * thetas[3], pTask0, 1] # 3 = scene error
              
            #update posterior P(S|R) = P(R|C = 0, S) * P(C = 0) * P(S) + P(R|C = 1, S) * P(C = 1) * P(S)
            if ((tProp[it] < theta1_upp) & (tProp[it] > theta1_low)):
                tmpPS[i0] = pS[i0] # no updating?
            else:
                if tResp[it] != 0:
                    if pS[i0] == 0:
                        tmpPS[i0] = 1 # 0.001 * pP[int(tResp[it])]
                    elif pS[i0] == 1:
                        tmpPS[i0] = 0 # 0.999 * pP[int(tResp[it])]
                else:
                    tmpPS[i0] = pS[i0]
                # else:
                #     tmpPS[i0] = pS[i0] * pP[int(tResp[it])]

        #if both likelihood drops to 0, randomly initialize state estimate
        if np.abs(np.sum(tmpPS)) < 1e-9:
            tmpPS[0] = np.random.rand()
            tmpPS[1] = 1 - tmpPS[0]
        else:
            tmpPS = tmpPS / np.sum(tmpPS)
        pS = tmpPS
        sE[it, 0] = tmpPS[0]
        sE[it, 1] = tmpPS[1]
    return sE, tE, cE





# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -            Format Inputs for Models           - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

"""
need to re-format data for computational model ... MODIFIED 4/21/2024 to flip tTask
tState
  tState = np.where(cur_df['state']==-1, 1, 0)  # recode state (1 -> 0 AND -1 -> 1 ... this flip matches model better)
tTask
  tTask   # face=0, scene=1
tProp
  tProp = np.array(cur_df['amb_r']).flatten()  # set as proportion of red
tResp
  0=correct , 1=right task but wrong answer , 2=wrong task, 3=no response
"""

# function ... format_subj_data_for_model_input(cur_sub, dataset_dir, output_dir)
def format_subj_data_for_model_input(cur_sub, dataset_dir, output_dir):
    print("\n\n\nConverting subject ", cur_sub, " into model input format...")
    # Load ouptut csv files for the two tasks and put each in their own respective master files (all subjects in long format)
    DM_list = sorted(glob.glob(os.path.join(dataset_dir,("sub-"+cur_sub+"_task-Quantum_*.csv"))))
    #DM_list = sorted(glob.glob(os.path.join(dataset_dir, "CSVs", ("sub-"+cur_sub+"_task-Cued*_*.csv"))))
    DM_dfs_list = []
    for cur_DM in sorted(DM_list):
        temp_df = pd.read_csv(cur_DM) # load file
        DM_dfs_list.append(temp_df) # add to df list
    cur_df = pd.concat(DM_dfs_list, ignore_index=True) # merge block level dfs that were stored in a list

    tState = np.where(cur_df['state']==-1, 1, 0) # recode state (1 now 0 ... -1 now 1 ... this flip matches model better than just changing -1)
    tTask = np.ones(len(tState)) # face=0, scene=1
    TaskPerformed = np.zeros(len(tState)) # face=1, scene=0

    cue_color = np.array(cur_df['cue']).flatten() # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
    cue_color = np.where(cue_color==-1, 0, cue_color) # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
    # now change amb to match tProp setup
    try:
        tProp = np.array(cur_df['amb_r']).flatten() # set as proportion of red
    except:
        amb_array = np.asarray(cur_df['amb'])
        color_array = list(cur_df['cue_predominant_color'])
        amb_r_array = []
        for idx, color in enumerate(color_array):
            if color == 'red':
                amb_r_array.append(amb_array[idx])
            else:
                amb_r_array.append( (1-amb_array[idx]) )
        cur_df['amb_r'] = amb_r_array
        tProp = np.array(cur_df['amb_r']).flatten() # set as proportion of red

    tResp=[]
    target = list(cur_df['target'])
    correct = np.array(cur_df['correct']).flatten()
    subj_resp = np.array(cur_df['subj_resp']).flatten()
    for trl, trl_target in enumerate(target):
        if trl_target == "face":
            tTask[trl] = 0 # change task to 0
        if correct[trl] == 1:
            tResp.append(0)
        else:
            if subj_resp[trl]==-1:
                # no response... cound as wrong task for now
                tResp.append(3)
            elif trl_target == "face":
                # resp should be 0 or 1
                if subj_resp[trl] > 1:
                    # wrong task
                    tResp.append(2)
                else:
                    # right task but wrong answer
                    tResp.append(1)
            elif trl_target == "scene":
                # resp should be 2 or 3
                if subj_resp[trl] < 2:
                    # wrong task
                    tResp.append(2)
                else:
                    # right task but wrong answer
                    tResp.append(1)
        if subj_resp[trl] < 2:
            # they think they should do the scene task
            TaskPerformed[trl]=1
    tResp = np.array(tResp).flatten()

    print("\ntState:\n", tState)
    print("\ntTask:\n", tTask)
    print("\ntProp:\n", tProp)
    print("\ntResp:\n", tResp)

    with open(os.path.join(output_dir, 'sub-'+cur_sub+'_tState.npy'), 'wb') as f:
        np.save(f, tState) # what state they were in
    with open(os.path.join(output_dir, 'sub-'+cur_sub+'_tTask.npy'), 'wb') as f:
        np.save(f, tTask) # what task they were doing
    with open(os.path.join(output_dir, 'sub-'+cur_sub+'_tProp.npy'), 'wb') as f:
        np.save(f, tProp) # what cue color proportion they saw
    with open(os.path.join(output_dir, 'sub-'+cur_sub+'_tResp.npy'), 'wb') as f:
        np.save(f, tResp) # what response they made
    cur_df.to_csv(os.path.join(output_dir, 'sub-'+cur_sub+'_dataframe.csv'))
    # NO RETURN ... end of func





# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -             Model Simulation Code             - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

"""
inputs:
  nTrials:     number of trials
  switchRange: block length, simulated trials will uniformly sampled from the range
  thetas:      thetas of subject: logistic slope parameter, face error rate and scene error rate

output:
  tState: simulated state
  tProp:  simulated color proportion
  tResp:  simulated response
  tTask:  simulated task
"""

#function [tState, tProp, tResp, tTask] = SequenceSimulation(nTrials, switchRange, thetas)
def SequenceSimulation(nTrials, switchRange, thetas):

  #rng; # NEED TO FIND PYTHON EQUIVALENT

  tState = np.array([])
  cState = 0
  l = 0
  #r = np.array(range(switchRange[0],(switchRange[1]+1),1)) 

  # set up state based on switch range
  vol_list = []
  while l < nTrials:
    # randomly decide if state is volitile or not
    vol_cond = np.random.randint([1,4]) # 1=low  2=med  ... 3=high
    # append state changes
    if len(vol_list) > 1:
        if np.sum(vol_list[-2:]) > 3:
            vol_cond = 1 # force low volitility if the last two random choices were both 2
        elif np.sum(vol_list[-2:]) == 2:
            vol_cond = 2 # force high volitility if the last two random choices were both 1
    else:
        vol_cond = 2
    
    r = np.random.randint([switchRange[0],switchRange[1]+1])
    vol_list.append(vol_cond)
    
    for ii in range(vol_cond):
        tState = np.concatenate( (tState, mb.repmat(cState, 1, int(r/vol_cond)).flatten()), axis=None ) 
        cState = 1 - cState
        l = l + int(r/vol_cond)

  tState = tState[:nTrials] # make sure tState is only as long as the number of trials
  
  tProp = np.asarray([np.random.randint(low=32, high=49, size=int(nTrials/2)), np.random.randint(low=51, high=68, size=int(nTrials/2))]).flatten()  # randomly generate proportion (cue)... btw 0 and 1
  tProp = tProp/100
  np.random.shuffle(tProp)
  #possible_prop = [0.11111111,0.33333333,0.44444444,0.48,0.52,0.55555556,0.66666667,0.88888889]
  # avoid extreme values in simulation
  # tProp[tProp == 0.05] = 0.05
  # tProp[tProp == 0.95] = 0.95

  tResp = np.zeros((tState.shape))
  tTask = np.zeros((tState.shape))

  # loop through all trials
  for i in range(nTrials):
    # np.random.shuffle(possible_prop)
    # tProp[i] = possible_prop[0]

    x = np.exp(thetas[0]) * np.log( (tProp[i]/(1 - tProp[i])) )
    p = 1 / (1 + np.exp((-1*x)))

    # agent got confused within 3 trials after switch
    if (i > 4) and ( (tState[i] != tState[(i-1)]) or (tState[i] != tState[(i-2)]) or (tState[i] != tState[(i-3)])):
      # if we are within 3 trials of a switch then assume they're still doing the old task
      cState = 1 - tState[i]
    else:
      # if we are within the first 3 task trials assume they are randomly guessing
      if i < 4:
        if np.random.rand() < 0.5:
          cState = 0
        else:
          cState = 1
      else:
        cState = tState[i]

    if ( (tProp[i] > 0.5) and (tState[i] == 0) ) or ( (tProp[i] < 0.5) and (tState[i] > 0) ):
      tTask[i] = 1 #0
    else:
      tTask[i] = 0 #1

    # rTask is the simulated task the agent thinks it should do
    if np.random.rand() < p:
      #color 0
      if cState == 0:
        rTask = 0
      else:
        rTask = 1
    else:
      #color 1
      if cState == 0:
        rTask = 1
      else:
        rTask = 0

    if tTask[i] != rTask:
      tResp[i] = 2
    else:
      if rTask == 0:
        if np.random.rand() < thetas[1]:
          tResp[i] = 1
        else:
          tResp[i] = 0
      else:
        if np.random.rand() < thetas[2]:
          tResp[i] = 1
        else:
          tResp[i] = 0

  return tState, tProp, tResp, tTask