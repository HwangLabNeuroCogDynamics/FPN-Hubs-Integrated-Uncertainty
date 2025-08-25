import sys
import os
import argparse
import glob
import pandas as pd
import pickle
import logging
import numpy as np
from natsort import natsorted
import numpy.matlib as mb
import matplotlib.pyplot as plt
#from fpnhiu.model_scripts.model_funcs import ModelBeliefGenerateion
from varname import nameof #not usually installed, found it on stackoverflow, make sure you install it.
from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
plt.ioff()

# ------------ Set Up for Parsing Arguments --------------
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data from subject",
        usage="[subject] [output_dir] [OPTIONS] ... ",
    )
    parser.add_argument("subject", help="single subject id... e.g., 10001")
    parser.add_argument("dataset_dir", help="/Shared/lss_kahwang_hpc/data/FPNHIU")
    parser.add_argument("--threshold", help="--threshold 0.5 ... sets fdmean threshold of 0.5", default=0.2)
    return parser

# -------------------  Set Options  ---------------------
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
subj_opt = args.subject
dataset_dir = args.dataset_dir
output_dir = os.path.join(args.dataset_dir,"3dDeconvolve")
model_dir = os.path.join(args.dataset_dir,"model_data")

# ---------- Define other useful functions --------------
# function to write AFNI style stim timing file
def write_stimtime(filepath, inputvec):
	''' short hand function to write AFNI style stimtime'''
	
	f = open(filepath, 'w')
	for val in inputvec[0]:
		if val =='*':
			f.write(val + '\n')
		else:
			# this is to dealt with some weird formating issue
			f.write(np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n')
	f.close()


def load_regressors(regressor_file, cols=None, verbose=False):
    """Loads regressor tsv into df with selected columns and fills in NaN values.

    Args:
        regressor_file (str): filepath to regressor tsv
        cols ([str], optional): List of column names. Defaults to None.
        verbose (bool, optional): If true, will log more info. Defaults to False.

    Returns:
        regressor_df (Dataframe): Dataframe containing regressors from selected columns.
        regressor_names ([str]): List of regressor names.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    df_orig = pd.read_csv(regressor_file, sep="\t")

    regressor_df = pd.DataFrame()
    regressor_names = []
    regressor_df = regressor_df.append(df_orig[cols])
    regressor_names.append(cols)

    for col in regressor_df.columns:
        sum_nan = sum(regressor_df[col].isnull())
        if sum_nan > 0:
            logging.info("Filling in " + str(sum_nan) +
                         " NaN value for " + col)
            regressor_df.loc[np.isnan(regressor_df[col]), col] = np.mean(
                regressor_df[col]
            )
    logging.info("# of Confound Regressors: " + str(len(regressor_df.columns)))

    return regressor_df, regressor_names

def censor(df, threshold=0.2, verbose=False):
    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t")

    censor_vector = np.empty((len(df.index)))
    prev_motion = 0

    for index, row in enumerate(zip(df["framewise_displacement"])):
        # censor first three points
        if index < 3:
            censor_vector[index] = 0
            continue

        if row[0] > threshold:
            censor_vector[index] = 0
            prev_motion = index
        elif prev_motion + 1 == index or prev_motion + 2 == index:
            censor_vector[index] = 0
        else:
            censor_vector[index] = 1

    if verbose:
        percent_censored = round(
            np.count_nonzero(censor_vector == 0) / len(censor_vector) * 100
        )
        print(f"\tCensored {percent_censored}% of points")
    return censor_vector

def load_regressors_and_censor(files, cols=None, threshold=0.2):
    output_df = pd.DataFrame()
    censor_list = []

    for file in files:
        print(f'Parsing: {file.split("/")[-1]}')
        file_df, _ = load_regressors(file, cols=cols, verbose=False)
        output_df = output_df.append(file_df)
        censor_list.extend(censor(file, threshold=threshold))

    return output_df, censor_list

# ----------------------------------------------------------
# ------------           Run Code           ----------------
# ----------------------------------------------------------
# --- create subject folder ... if it does not exist already
if not os.path.isdir(os.path.join(output_dir,("sub-"+subj_opt))):
    print("creating subj directory now")
    os.mkdir(os.path.join(output_dir,("sub-"+subj_opt)))

# --- generate nuisance file
confounds = sorted(glob.glob(os.path.join(dataset_dir,"fmriprep",("sub-"+subj_opt),"func","*confounds_*.tsv")))
print(confounds)
cdf = pd.read_csv(confounds[0], sep='\t') # create data frame from run 1
for c in confounds[1:5]: # only analyzing 5 runs
    cdf =cdf.append(pd.read_csv(c, sep='\t')) # append runs 2 - 5
default_columns = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2', 
                        'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                        'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2', 
                        'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2', 
                        'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
                        'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
noise_mat = cdf.loc[:,default_columns].values
noise_mat[np.isnan(noise_mat)]=0
np.savetxt(os.path.join(output_dir,("sub-"+subj_opt),'noise.1D'), noise_mat)

# --- now generate censor file
regressor_filepath = os.path.join(output_dir, ("sub-"+subj_opt), "nuisance.1D")
censor_filepath = os.path.join(output_dir, ("sub-"+subj_opt), "censor.1D")
print(f"Prepping 3dDeconvolve on subject {subj_opt}")
print(f"Extracting columns {default_columns} from regressor files")
regressor_files = natsorted(glob.glob(os.path.join(dataset_dir,"fmriprep",("sub-"+subj_opt),"func","*confounds_timeseries.tsv")))
regressor_df, output_censor = load_regressors_and_censor(regressor_files, cols=default_columns, threshold=float(args.threshold))
print(f"Writing regressor file to {regressor_filepath}")
regressor_df.to_csv(regressor_filepath, header=False, index=False, sep="\t")
print(f"Writing censor file to {censor_filepath}")
with open(censor_filepath, "w") as file:
    for num in output_censor:
        file.writelines(f"{num}\n")
print(f"\n\nSuccessfully extracted columns {default_columns} from regressor files and censored motion")


# --- load current subject INPUT data files
tState = np.load(os.path.join(model_dir,("sub-"+subj_opt+"_tState.npy")))
tProp = np.load(os.path.join(model_dir,("sub-"+subj_opt+"_tProp.npy")))
tResp = np.load(os.path.join(model_dir,("sub-"+subj_opt+"_tResp.npy")))
tTask = np.load(os.path.join(model_dir,("sub-"+subj_opt+"_tTask.npy")))
tProp_binary = tProp.round()

# --- load model files 
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pState_pPercept_outputs.p")), 'rb') as handle:
#     pSpP_params=pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dState_pPercept_outputs.p")), 'rb') as handle:
#     dSpP_params=pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pState_dPercept_outputs.p")), 'rb') as handle:
#     pSdP_params=pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dState_dPercept_outputs.p")), 'rb') as handle:
#     dSdP_params=pickle.load(handle)

# --- load model beliefs
# # -- sE ... 2D
# # sE[:,0] = probability of state 0
# # sE[:,1] = probability of state 1
with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSpP_belief-state.p")), 'rb') as handle:
    pSpP_sE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSpP_belief-state.p")), 'rb') as handle:
#     dSpP_sE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSdP_belief-state.p")), 'rb') as handle:
#     pSdP_sE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSdP_belief-state.p")), 'rb') as handle:
#     dSdP_sE = pickle.load(handle)
# # -- tE ... 1D
# # tE = probability of task 0 (face?)
with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSpP_belief-task.p")), 'rb') as handle:
    pSpP_tE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSpP_belief-task.p")), 'rb') as handle:
#     dSpP_tE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSdP_belief-task.p")), 'rb') as handle:
#     pSdP_tE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSdP_belief-task.p")), 'rb') as handle:
#     dSdP_tE = pickle.load(handle)
# # -- cE ... 1D
# # cE = probability of color 0 (yellow)
with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSpP_belief-color.p")), 'rb') as handle:
    pSpP_cE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSpP_belief-color.p")), 'rb') as handle:
#     dSpP_cE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_pSdP_belief-color.p")), 'rb') as handle:
#     pSdP_cE = pickle.load(handle)
# with open(os.path.join(model_dir,("sub-"+str(subj_opt)+"_dSdP_belief-color.p")), 'rb') as handle:
#     dSdP_cE = pickle.load(handle)


# --- calculate entropy 
def calc_entropy(cur_model_sE, cE):
    jd = np.array([cE * cur_model_sE[:,1],  (1 - cE) * cur_model_sE[:,1], 
                   cE * (1 - cur_model_sE[:,1]),  (1 - cE) * (1 - cur_model_sE[:,1])])
    t_entropy = [] #entorpy of dist
    for t in range(len(cur_model_sE)):
        t_entropy.append(entropy(jd[:,t].flatten()))
    t_entropy = np.array(t_entropy) # convert to numpy array
    return t_entropy
# entropy for MPE model
pSpP_t_entropy = calc_entropy(pSpP_sE, pSpP_cE)
# # entropy for control models
# dSpP_t_entropy = calc_entropy(dSpP_sE, dSpP_cE)
# pSdP_t_entropy = calc_entropy(pSdP_sE, pSdP_cE)
# dSdP_t_entropy = calc_entropy(dSdP_sE, dSdP_cE)


# --- Set up model based parametric regressor variables
sE = np.array(pSpP_sE[:,0]) 
tE = np.array(pSpP_tE) 
cE = np.array(pSpP_cE)

t_entropy = pSpP_t_entropy

statePE = abs(tState - (1-sE))
taskPE = abs(tTask - (1-tE))
colorPE = abs(tProp_binary - (cE)) # because cE is actually the prob of color-1 (not color 0)

# add state derivative
stateD = np.zeros(len(sE))
for idx, sE_trl in enumerate(pSpP_sE[:,1]):
    if idx == 0:
        stateD[idx] = np.abs(sE_trl - 0.5)
    else:
        stateD[idx] = np.abs(sE_trl - pSpP_sE[(idx-1),1])

# convert state and task estimate into confidence/uncertainty ... closer to 0 is high confence, low uncertainty
state_est = 1 - abs(np.array(sE)-0.5)*2
task_est = 1 - abs(np.array(tE)-0.5)*2
color_est = 1 - abs(np.array(cE)-0.5)*2

# -- z-scored color state and entropy
z_state = (state_est-state_est.mean()) / state_est.std()
z_color = (color_est-color_est.mean()) / color_est.std()
z_entropy = (t_entropy-t_entropy.mean()) / t_entropy.std()

z_statePE = (statePE-statePE.mean()) / statePE.std()
z_colorPE = (colorPE-colorPE.mean()) / colorPE.std()
z_taskPE = (taskPE-taskPE.mean()) / taskPE.std()

###############################################################
### now create regressors for these model params.
###############################################################
df = pd.read_csv(os.path.join(dataset_dir,"model_data",('sub-'+subj_opt+'_dataframe.csv')))
df['Run'] = df['block']+1
num_runs=5 # setting at 5 because some subjects have more but we don't want to use them
# add some model estimates to the dataframe and save it out
df_mes = df.copy()
df_mes['tState'] = tState
df_mes['tTask'] = tTask
df_mes['tProp'] = tProp
df_mes['tResp'] = tResp

df_mes['pSpP_cE'] = cE # add color estimate
df_mes['pSpP_sE'] = sE
df_mes['pSpP_tE'] = tE

df_mes['pSpP_entropy'] = np.array(t_entropy)
#df_mes['pSpP_entropyChange'] = np.array(t_entropy_change)

df_mes['pSpP_cPE'] = colorPE # add color PE estimate
df_mes['pSpP_sPE'] = statePE
df_mes['pSpP_tPE'] = taskPE

df_mes['pSpP_sD'] = stateD
#df_mes['pSpP_tD'] = taskD

df_mes['pSpP_cU'] = color_est # add colorU estimate
df_mes['pSpP_sU'] = state_est
df_mes['pSpP_tU'] = task_est

df_mes.to_csv(os.path.join(dataset_dir,"model_data",('sub-'+subj_opt+'_dataframe_3dDeconvolve_pSpP.csv')))

## create amplitude modulation record for 3dDeconvolve. See: https://afni.nimh.nih.gov/pub/dist/doc/misc/Decon/AMregression.pdf
# belief_dict = {'state':state_est, 'task':task_est, 'color':color_est, 'entropy':t_entropy,
#                 'statePE':statePE, 'taskPE':taskPE, 'stateD':stateD} # prior to reviews, these are the stimfiles we used
belief_dict = {'stateD':stateD,
               'zState':z_state, 'zColor':z_color, 'zEntropy':z_entropy,
               'zColorPE':z_colorPE,'zStatePE':z_statePE,'zTaskPE':z_taskPE}
               
for epch in ['cue','probe','fb']:
    for belief in ['stateD','zState','zColor','zEntropy','zColorPE','zStatePE','zTaskPE']:
        tmp_stimtime = [['*']*num_runs]
        j=0
        for i, run in enumerate(np.arange(1,num_runs+1)): 
            run_df = df[df['Run']==run].reset_index() #"view" of block we are curreint sorting through
            tmp = []
            for tr in np.arange(0, len(run_df)):
                tmp.append(belief_dict[belief][j])
                j = j+1
            tmp_stimtime[0][i] = tmp
        # write out stimtime array to text file
        write_stimtime(os.path.join(output_dir,("sub-"+subj_opt),(epch+"_"+belief+"_am.1D")), tmp_stimtime)

acc_array = np.asarray(df['correct'])
for belief in ['correct','incorrect']:
    tmp_stimtime = [['*']*num_runs]
    j=0
    for i, run in enumerate(np.arange(1,num_runs+1)): 
        run_df = df[df['Run']==run].reset_index() #"view" of block we are curreint sorting through
        tmp = []
        for tr in np.arange(0, len(run_df)):
            if belief=='correct':
                if int(acc_array[j]) == 1:
                    tmp.append(run_df.loc[tr,'Time_Since_Run_Feedback_Prez'])
            else:
                if int(acc_array[j]) == 0:
                    tmp.append(run_df.loc[tr,'Time_Since_Run_Feedback_Prez'])
            j = j+1
        tmp_stimtime[0][i] = tmp
    # write out stimtime array to text file
    write_stimtime(os.path.join(output_dir,("sub-"+subj_opt),("fb_"+belief+".1D")), tmp_stimtime)


##### create cue and probe timing from psychopy output
#create FIR timing for cue and probe, extract trial timing
Cue_stimtime = [['*']*num_runs] #stimtime format, * for runs with no events of this stimulus class
Probe_stimtime = [['*']*num_runs] #create one of this for every condition
Fb_stimtime = [['*']*num_runs] #create one of this for every condition
# new additions ... will be grouped based on model estimate of state belief (but not modulated by that parameter)
jj=0
for i, run in enumerate(np.arange(1,num_runs+1)):  # loop through 5 runs
    run_df = df[df['Run']==run].reset_index() #"view" of block we are curreint sorting through
    run_df['sE'] = sE[(i*len(run_df)):(run*len(run_df))]
    run_df['tE'] = tE[(i*len(run_df)):(run*len(run_df))]
    Cue_times = [] #empty vector to store trial time info for the current block
    Probe_times = []
    Fb_times = []
    color_by_trl = []
    amb_by_trl = []
    state_by_trl = []
    # sE ... state estimate by trial
    for tr in np.arange(0, len(run_df)):  #this is to loop through trials
        Cue_times.append(run_df.loc[tr,'Time_Since_Run_Cue_Prez']) 		
        Probe_times.append(run_df.loc[tr,'Time_Since_Run_Photo_Prez']) 
        Fb_times.append(run_df.loc[tr,'Time_Since_Run_Feedback_Prez']) 
        color_by_trl.append(run_df.loc[tr,'cue_predominant_color'])
        amb_by_trl.append(run_df.loc[tr,'amb'])
        state_by_trl.append(sE[jj])
        jj+=1
    Cue_stimtime[0][i] = Cue_times # put trial timing of each block into the stimtime array
    Probe_stimtime[0][i] = Probe_times
    Fb_stimtime[0][i] = Fb_times
    
#write out stimtime array to text file	
write_stimtime(os.path.join(output_dir,("sub-"+subj_opt),"cue_stimtime.1D"), Cue_stimtime)
write_stimtime(os.path.join(output_dir,("sub-"+subj_opt),"probe_stimtime.1D"), Probe_stimtime)
write_stimtime(os.path.join(output_dir,("sub-"+subj_opt),"fb_stimtime.1D"), Fb_stimtime)