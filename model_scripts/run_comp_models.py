import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
from fpnhiu.model_scripts.model_funcs import gen_model_plots, calc_entropy, MPEModel, JanModel, ModelBeliefGeneration, Jan_ModelBeliefGeneration, MPEModel_Dichotomous, format_subj_data_for_model_input
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
plt.ioff()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data from subject",
        usage="[subject] [output_dir] [OPTIONS] ... ",
    )
    parser.add_argument("subject", help="single subject id... e.g., 10001")
    parser.add_argument("dataset_dir", help="/Shared/lss_kahwang_hpc/data/TRIIMS")
    parser.add_argument("--mpe_model",
                        help="run mpe model, default is false",
                        default=False, action="store_true")
    parser.add_argument("--control_model",
                        help="run control model, default is false",
                        default=False, action="store_true")
    return parser


# ----------------  Set Options  ------------------
#generate_plots = False
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
subj_opt = args.subject
dataset_dir = args.dataset_dir
output_dir = os.path.join(args.dataset_dir,"model_data")
# -------------------------------------------------

# ----- put subject data in model input format
if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_tState.npy"))):
    os.remove(os.path.join(output_dir,("sub-"+subj_opt+"_tState.npy")))
if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_tProp.npy"))):
    os.remove(os.path.join(output_dir,("sub-"+subj_opt+"_tProp.npy")))
if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_tResp.npy"))):
    os.remove(os.path.join(output_dir,("sub-"+subj_opt+"_tResp.npy")))
if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_tTask.npy"))):
    os.remove(os.path.join(output_dir,("sub-"+subj_opt+"_tTask.npy")))
format_subj_data_for_model_input(subj_opt, dataset_dir, output_dir)


# ----- load input data for this subject
print("\nloading files for subject",str(subj_opt))
# load current subject input DATA files
c_tState = np.load(os.path.join(output_dir,("sub-"+subj_opt+"_tState.npy")))
c_tProp = np.load(os.path.join(output_dir,("sub-"+subj_opt+"_tProp.npy")))
c_tResp = np.load(os.path.join(output_dir,("sub-"+subj_opt+"_tResp.npy")))
c_tTask = np.load(os.path.join(output_dir,("sub-"+subj_opt+"_tTask.npy")))
cur_df = pd.read_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe.csv'))
cur_df['Run'] = cur_df['block']+1
c_tProp_bin = c_tProp.copy()
c_tProp_bin = np.where(c_tProp_bin>0.5, 1, 0)


# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -         MPE MODEL        - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------
# ----- run mpe model (if requested)
if args.mpe_model:
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_pState_pPercept_outputs_v4.p"))):
        print("\tMPE Model already generated for this subject... loading existing output and moving on")
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_MPEmodel_outputs.p")), 'rb') as handle:
            c_mpe_output=pickle.load(handle)
    else:
        print("\tRunning MPE (pSpP) Model now...")
        jd, sE, tE, mDist, pRange, cE = MPEModel(c_tState, c_tProp, c_tResp)
        c_mpe_output={'jd':jd, 'sE':sE, 'tE':tE, 'mDist':mDist, 'pRange':pRange, 'cE':cE} # save outputs in a dictionary
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_pPercept_outputs_v4.p")), 'wb') as handle:
            pickle.dump(c_mpe_output, handle, protocol=4) #pickle.HIGHEST_PROTOCOL)
    
    # -- generate model beliefs
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-state.p"))):
        print('model belief already generated for this subject... loading existing output and moving on')
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-state_v4.p")), 'rb') as handle:
            sE_belief=pickle.load(handle)
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-task_v4.p")), 'rb') as handle:
            tE_belief=pickle.load(handle)
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-color_v4.p")), 'rb') as handle:
            cE_belief=pickle.load(handle)
    else:
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_pPercept_outputs_v4.p")), 'rb') as handle:
            mpe_model_dict=pickle.load(handle)
        
        s_jd = np.sum(mpe_model_dict['jd'], axis=0)
        #s_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
        s_pRange = mpe_model_dict['pRange']
        s_theta = [s_pRange['lfsp'][np.argmax(np.sum(s_jd,axis=(1,2,3,4)))], 
                    s_pRange['lfip'][np.argmax(np.sum(s_jd,axis=(0,2,3,4)))], 
                    s_pRange['fter'][np.argmax(np.sum(s_jd,axis=(0,1,3,4)))], 
                    s_pRange['ster'][np.argmax(np.sum(s_jd,axis=(0,1,2,4)))], 
                    s_pRange['diff_at'][np.argmax(np.sum(s_jd,axis=(0,1,2,3,)))]]
        #                      ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
        sE_belief, tE_belief, cE_belief = ModelBeliefGeneration(s_theta, c_tState, c_tProp, c_tResp, True, True)
        cur_df['pSpP_sE'] = sE_belief[:,1] # add probability it is state 0
        cur_df['pSpP_tE'] = tE_belief # probability task 0 (face)
        cur_df['pSpP_cE'] = cE_belief # probability color 0 (red)
        
        #print("new tE:", tE_belief[:10], "\n\nold tE:", mpe_model_dict['tE'][:10])
        #coeff_pval = stats.pearsonr(np.asarray(mpe_model_dict['tE']), np.asarray(tE_belief))
        #print("correlation coeff: ", coeff_pval[0], "\tpval:", coeff_pval[1])
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-state_v4.p")), 'wb') as handle:
            pickle.dump(sE_belief, handle, protocol=4)
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-task_v4.p")), 'wb') as handle:
            pickle.dump(tE_belief, handle, protocol=4)
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSpP_belief-color_v4.p")), 'wb') as handle:
            pickle.dump(cE_belief, handle, protocol=4)
        
        # entropy for MPE model
        pSpP_t_entropy, pSpP_t_entropy_change = calc_entropy(sE_belief, tE_belief)
        cur_df['pSpP_entropy'] = pSpP_t_entropy 
        #cur_df['pSpP_entropyChange'] = pSpP_t_entropy_change 
        
        # save df with pSpP model estimates
        cur_df.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_model-pSpP_v4.csv'))

    # - - - - - - - - - - - - - - - -
    # ----- make plots
    # -- model estimates
    c_sE = sE_belief[:200,1] # [:,1] = probability state 1
    c_tE = 1 - tE_belief[:200] # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = cE_belief[:200]
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-ModelEstimates_model-pSpP_v4.png")), c_tResp[:200])
    # -- prediction error
    c_sE = np.abs(c_tState[:200] - sE_belief[:200,1]) # [:,1] = probability state 1
    c_tE = np.abs(c_tTask[:200] - (1 - tE_belief[:200]))  # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = np.abs(c_tProp_bin[:200] - cE_belief[:200])
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-PredictionError_model-pSpP_v4.png")), c_tResp[:200])
    # -- theta parameters
    pRange_keys = ['lfsp', 'lfip', 'fter', 'ster', 'diff_at']
    pRange_titles = {'lfsp':'slope param', 'lfip':'intercept param', 'fter':'face error param', 'ster':'scene error param', 'diff_at':'diffusion param'}
    x_ranges = [[1.5, 5], [-3.0, 3.0], [0.0, 0.35], [0.0, 0.35], [0.0, 0.9]]
    pRange_colors = ['deepskyblue','steelblue','orange','green','red'] # set colors so I know what is what
    pRange = mpe_model_dict['pRange']
    mDist = mpe_model_dict['mDist']
    plt.rcParams["figure.figsize"] = (15,3)
    fig_thetas, ax_thetas = plt.subplots(1,5)
    for ind, cur_key in enumerate(pRange_keys):
        ax_thetas[ind].set_xlim(x_ranges[ind][0],x_ranges[ind][1])
        ax_thetas[ind].set_yticklabels([])
        ax_thetas[ind].plot(pRange[cur_key], mDist[cur_key], color=pRange_colors[ind])
        ax_thetas[ind].set_title(pRange_titles[cur_key])
    plt.show()
    plt.savefig(os.path.join(output_dir,("sub-"+subj_opt+"_model-pSpP__thetas_v4.png")))
    plt.close()



# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -         CONTROL MODELS        - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

# ControlModel(tState, tProp, tResp, use_pState, use_pPercept)
# MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept):
if args.control_model:
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ----- run JAN control model (if requested) where ...
    cur_df_cJ = cur_df.copy()
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_pState_jPercept_outputs.p"))):
        print("\tstate==probabalistic & color==probabalistic/Dichotomous Model already generated for this subject... moving on")
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_jPercept_outputs.p")), 'rb') as handle:
            c_jan_output=pickle.load(handle)
    else:
        print("\tRunning state==Probabalistic & color==Dichotomous Model now...")
        #                                     MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept)
        jd_j, sE_j, tE_j, mDist_j, pRange_j, cE_j = JanModel(c_tState, c_tProp, c_tResp, False)
        c_jan_output={'jd':jd_j, 'sE':sE_j, 'tE':tE_j, 'mDist':mDist_j, 'pRange':pRange_j, 'cE':cE_j} # save outputs in a dictionary
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_jPercept_outputs.p")), 'wb') as handle:
            pickle.dump(c_jan_output, handle, protocol=4)
    pSjP_jd = np.sum(c_jan_output['jd'], axis=0)
    #pSjP_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    pSjP_pRange = c_jan_output['pRange']
    pSjP_theta = [pSjP_pRange['cub'][np.argmax(np.sum(pSjP_jd,axis=(1,2,3,4)))], 
                    pSjP_pRange['clb'][np.argmax(np.sum(pSjP_jd,axis=(0,2,3,4)))],
                    pSjP_pRange['fter'][np.argmax(np.sum(pSjP_jd,axis=(0,1,3,4)))], 
                    pSjP_pRange['ster'][np.argmax(np.sum(pSjP_jd,axis=(0,1,2,4)))], 
                    pSjP_pRange['diff_at'][np.argmax(np.sum(pSjP_jd,axis=(0,1,2,3)))]]
    print(pSjP_theta)
    #                          ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
    pSjP_sE_new, pSjP_tE_new, pSjP_cE_new = Jan_ModelBeliefGeneration(pSjP_theta, c_tState, c_tProp, c_tResp, False)
    cur_df_cJ['pSjP_sE'] = pSjP_sE_new[:,1] # add probability it is state 1
    cur_df_cJ['pSjP_tE'] = pSjP_tE_new
    cur_df_cJ['pSjP_cE'] = pSjP_cE_new
    
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSjP_belief-state.p")), 'wb') as handle:
        pickle.dump(pSjP_sE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSjP_belief-task.p")), 'wb') as handle:
        pickle.dump(pSjP_tE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSjP_belief-color.p")), 'wb') as handle:
        pickle.dump(pSjP_cE_new, handle, protocol=4)
        
    # entropy for Jan model
    pSjP_t_entropy, pSjP_t_entropy_change = calc_entropy(pSjP_sE_new, pSjP_cE_new)
    cur_df_cJ['pSjP_entropy'] = pSjP_t_entropy 
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSjP_entropy.p")), 'wb') as handle:
        pickle.dump(pSjP_t_entropy, handle, protocol=4) 
    
    # save df with pSpP model estimates
    cur_df_cJ.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_model-pSjP.csv'))
    
    # ----- make plots
    # -- model estimates
    c_sE = pSjP_sE_new[:,1] # [:,1] = probability state 1
    c_tE = 1 - pSjP_tE_new # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = pSjP_cE_new
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-ModelEstimates_model-pSjP.png")), c_tResp[:200])
    # -- prediction error
    c_sE = np.abs(c_tState - pSjP_sE_new[:,1]) # [:,1] = probability state 1
    c_tE = np.abs(c_tTask - (1 - pSjP_tE_new))  # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = np.abs(c_tProp_bin - pSjP_cE_new)
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-PredictionError_model-pSjP.png")), c_tResp[:200])
    # -- theta parameters
    pRange_keys = ['cub', 'clb', 'fter', 'ster', 'diff_at']
    pRange_titles = {'cub':'upper bound param', 'clb':'lower bound param', 'fter':'face error param', 'ster':'scene error param', 'diff_at':'diffusion param'}
    x_ranges = [[-0.15, 0.25], [-0.25, 0.15], [0.0, 0.3], [0.0, 0.3], [0.0, 0.9]]
    pRange_colors = ['deepskyblue', 'steelblue','orange','green','red'] # set colors so I know what is what
    pRange = c_jan_output['pRange']
    mDist = c_jan_output['mDist']
    plt.rcParams["figure.figsize"] = (15,3)
    fig_thetas, ax_thetas = plt.subplots(1,5)
    for ind, cur_key in enumerate(pRange_keys):
        ax_thetas[ind].set_xlim(x_ranges[ind][0],x_ranges[ind][1])
        ax_thetas[ind].set_yticklabels([])
        ax_thetas[ind].plot(pRange[cur_key], mDist[cur_key], color=pRange_colors[ind])
        ax_thetas[ind].set_title(pRange_titles[cur_key])
    plt.show()
    plt.savefig(os.path.join(output_dir,("sub-"+subj_opt+"_model-pSjP__thetas.png")))
    plt.close()
    
    #print(mDist['cub'])


    
    
    # -- load pSpP model ... have to use lfsp from pP model for dP models (or else code doesn't work)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_pPercept_outputs.p")), 'rb') as handle:
        mpe_model_dict=pickle.load(handle)
    s_jd = np.sum(mpe_model_dict['jd'], axis=0)
    #s_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    s_pRange = mpe_model_dict['pRange']
    s_theta = [s_pRange['lfsp'][np.argmax(np.sum(s_jd,axis=(1,2,3,4)))], 
                s_pRange['lfip'][np.argmax(np.sum(s_jd,axis=(0,2,3,4)))], 
                s_pRange['fter'][np.argmax(np.sum(s_jd,axis=(0,1,3,4)))], 
                s_pRange['ster'][np.argmax(np.sum(s_jd,axis=(0,1,2,4)))], 
                s_pRange['diff_at'][np.argmax(np.sum(s_jd,axis=(0,1,2,3,)))]]
    
    
    #for ext in ['.csv', '_v2.csv', '_v3.csv', '_v4.csv']:
    ext1 = '.p'
    ext2 = '.csv'
    ext3 = '.png'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ----- run control model (if requested) where STATE & PERCEPTUAL are DICHOTOMOUS (original control)
    cur_df_c1 = cur_df.copy()
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_dState_dPercept_outputsAAA.p"))):
        print("\tstate==Dichotomous & color==Dichotomous Model already generated for this subject... moving on")
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_dState_dPercept_outputs.p")), 'rb') as handle:
            c_det_output=pickle.load(handle)
    else:
        print("\tRunning state==Dichotomous & color==Dichotomous Model now...")
        #                                     MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept)
        jd_d, sE_d, sT_d, mDist_d, pRange_d, cE_d = MPEModel_Dichotomous(c_tState, c_tProp, c_tResp, False, False)
        c_det_output={'jd':jd_d, 'sE':sE_d, 'tE':sT_d, 'mDist':mDist_d, 'pRange':pRange_d, 'cE':cE_d} # save outputs in a dictionary
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_dState_dPercept_outputs"+ext1)), 'wb') as handle:
            pickle.dump(c_det_output, handle, protocol=4)
    dSdP_jd = np.sum(c_det_output['jd'], axis=0)
    #dSdP_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    dSdP_pRange = c_det_output['pRange']
    dSdP_theta = [dSdP_pRange['lfsp'][np.argmax(np.sum(dSdP_jd,axis=(1,2,3,4)))], 
                    dSdP_pRange['lfip'][np.argmax(np.sum(dSdP_jd,axis=(0,2,3,4)))], 
                    dSdP_pRange['fter'][np.argmax(np.sum(dSdP_jd,axis=(0,1,3,4)))], 
                    dSdP_pRange['ster'][np.argmax(np.sum(dSdP_jd,axis=(0,1,2,4)))], 
                    dSdP_pRange['diff_at'][np.argmax(np.sum(dSdP_jd,axis=(0,1,2,3,)))]]
    
    #                          ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
    dSdP_sE_new, dSdP_tE_new, dSdP_cE_new = ModelBeliefGeneration(dSdP_theta, c_tState, c_tProp, c_tResp, False, False)
    cur_df_c1['dSdP_sE'] = dSdP_sE_new[:,1] # add probability it is state 1
    cur_df_c1['dSdP_tE'] = dSdP_tE_new
    cur_df_c1['dSdP_cE'] = dSdP_cE_new
    
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSdP_belief-state"+ext1)), 'wb') as handle:
        pickle.dump(dSdP_sE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSdP_belief-task"+ext1)), 'wb') as handle:
        pickle.dump(dSdP_tE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSdP_belief-color"+ext1)), 'wb') as handle:
        pickle.dump(dSdP_cE_new, handle, protocol=4)
        
    # entropy for dSdP model
    dSdP_t_entropy, dSdP_t_entropy_change = calc_entropy(dSdP_sE_new, dSdP_cE_new)
    cur_df_c1['dSdP_entropy'] = dSdP_t_entropy 
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSdP_entropy"+ext1)), 'wb') as handle:
        pickle.dump(dSdP_t_entropy, handle, protocol=4) 
    
    # save df with pSpP model estimates
    cur_df_c1.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_model-dSdP.csv'))
    

    # ----- make plots
    # -- model estimates
    c_sE = dSdP_sE_new[:,1] # [:,1] = probability state 1
    c_tE = 1 - dSdP_tE_new # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = dSdP_cE_new
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-ModelEstimates_model-dSdP"+ext3)), c_tResp[:200])
    # -- prediction error
    c_sE = np.abs(c_tState - dSdP_sE_new[:,1]) # [:,1] = probability state 1
    c_tE = np.abs(c_tTask - (1 - dSdP_tE_new))  # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = np.abs(c_tProp_bin - dSdP_cE_new)
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-PredictionError_model-dSdP"+ext3)), c_tResp[:200])
    # -- theta parameters
    pRange_keys = ['lfsp', 'lfip', 'fter', 'ster', 'diff_at']
    pRange_titles = {'lfsp':'slope param', 'lfip':'intercept param', 'fter':'face error param', 'ster':'scene error param', 'diff_at':'diffusion param'}
    x_ranges = [[1.5, 5], [-3.0, 3.0], [0.0, 0.35], [0.0, 0.35], [0.0, 0.9]]
    pRange_colors = ['deepskyblue','steelblue','orange','green','red'] # set colors so I know what is what
    pRange = c_det_output['pRange']
    mDist = c_det_output['mDist']
    plt.rcParams["figure.figsize"] = (15,3)
    fig_thetas, ax_thetas = plt.subplots(1,5)
    for ind, cur_key in enumerate(pRange_keys):
        ax_thetas[ind].set_xlim(x_ranges[ind][0],x_ranges[ind][1])
        ax_thetas[ind].set_yticklabels([])
        ax_thetas[ind].plot(pRange[cur_key], mDist[cur_key], color=pRange_colors[ind])
        ax_thetas[ind].set_title(pRange_titles[cur_key])
    plt.show()
    plt.savefig(os.path.join(output_dir,("sub-"+subj_opt+"_model-dSdP__thetas"+ext3)))
    plt.close()



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ----- run control model (if requested) where STATE is PROBABALISTIC & PERCEPTUAL is DICHOTOMOUS
    cur_df_c2 = cur_df.copy()
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_pState_dPercept_outputsAAA.p"))):
        print("\tstate==probabalistic & color==Dichotomous Model already generated for this subject... moving on") 
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_dPercept_outputs.p")), 'rb') as handle:
            c_det_output2=pickle.load(handle)
    else:
        print("\tRunning state==probabalistic & color==Dichotomous Model now...") 
        #                                     MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept)
        jd_d, sE_d, sT_d, mDist_d, pRange_d, cE_d = MPEModel_Dichotomous(c_tState, c_tProp, c_tResp, True, False)
        c_det_output2={'jd':jd_d, 'sE':sE_d, 'tE':sT_d, 'mDist':mDist_d, 'pRange':pRange_d, 'cE':cE_d} # save outputs in a dictionary
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_pState_dPercept_outputs"+ext1)), 'wb') as handle:
            pickle.dump(c_det_output2, handle, protocol=4) 
    pSdP_jd = np.sum(c_det_output2['jd'], axis=0)
    #pSdP_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    pSdP_pRange = c_det_output2['pRange']
    pSdP_theta = [pSdP_pRange['lfsp'][np.argmax(np.sum(pSdP_jd,axis=(1,2,3,4)))], 
                    pSdP_pRange['lfip'][np.argmax(np.sum(pSdP_jd,axis=(0,2,3,4)))], 
                    pSdP_pRange['fter'][np.argmax(np.sum(pSdP_jd,axis=(0,1,3,4)))], 
                    pSdP_pRange['ster'][np.argmax(np.sum(pSdP_jd,axis=(0,1,2,4)))], 
                    pSdP_pRange['diff_at'][np.argmax(np.sum(pSdP_jd,axis=(0,1,2,3,)))]]
    
    #                          ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
    pSdP_sE_new, pSdP_tE_new, pSdP_cE_new  = ModelBeliefGeneration(pSdP_theta, c_tState, c_tProp, c_tResp, True, False)
    cur_df_c2['pSdP_sE'] = pSdP_sE_new[:,1] # add probability it is state 1
    cur_df_c2['pSdP_tE'] = pSdP_tE_new
    cur_df_c2['pSdP_cE'] = pSdP_cE_new
    
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSdP_belief-state"+ext1)), 'wb') as handle:
        pickle.dump(pSdP_sE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSdP_belief-task"+ext1)), 'wb') as handle:
        pickle.dump(pSdP_tE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSdP_belief-color"+ext1)), 'wb') as handle:
        pickle.dump(pSdP_cE_new, handle, protocol=4) 
        
    # entropy for pSdP model
    pSdP_t_entropy, pSdP_t_entropy_change = calc_entropy(pSdP_sE_new, pSdP_cE_new)
    cur_df_c2['pSdP_entropy'] = pSdP_t_entropy 
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_pSdP_entropy"+ext1)), 'wb') as handle:
        pickle.dump(pSdP_t_entropy, handle, protocol=4) 
    
    # save df with pSpP model estimates
    cur_df_c2.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_model-pSdP.csv'))
    
    
    # ----- make plots
    # -- model estimates
    c_sE = pSdP_sE_new[:,1] # [:,1] = probability state 1
    c_tE = 1 - pSdP_tE_new # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = pSdP_cE_new
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-ModelEstimates_model-pSdP"+ext3)), c_tResp[:200])
    # -- prediction error
    c_sE = np.abs(c_tState - pSdP_sE_new[:,1]) # [:,1] = probability state 1
    c_tE = np.abs(c_tTask - (1 - pSdP_tE_new)) # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = np.abs(c_tProp_bin - pSdP_cE_new)
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-PredictionError_model-pSdP"+ext3)), c_tResp[:200])
    # -- theta parameters
    pRange_keys = ['lfsp', 'lfip', 'fter', 'ster', 'diff_at']
    pRange_titles = {'lfsp':'slope param', 'lfip':'intercept param', 'fter':'face error param', 'ster':'scene error param', 'diff_at':'diffusion param'}
    x_ranges = [[1.5, 5], [-3.0, 3.0], [0.0, 0.35], [0.0, 0.35], [0.0, 0.9]]
    pRange_colors = ['deepskyblue','steelblue','orange','green','red'] # set colors so I know what is what
    pRange = c_det_output2['pRange']
    mDist = c_det_output2['mDist']
    plt.rcParams["figure.figsize"] = (15,3)
    fig_thetas, ax_thetas = plt.subplots(1,5)
    for ind, cur_key in enumerate(pRange_keys):
        ax_thetas[ind].set_xlim(x_ranges[ind][0],x_ranges[ind][1])
        ax_thetas[ind].set_yticklabels([])
        ax_thetas[ind].plot(pRange[cur_key], mDist[cur_key], color=pRange_colors[ind])
        ax_thetas[ind].set_title(pRange_titles[cur_key])
    plt.show()
    plt.savefig(os.path.join(output_dir,("sub-"+subj_opt+"_model-pSdP__thetas"+ext3)))
    plt.close()



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ----- run control model (if requested) where STATE is DICHOTOMOUS & PERCEPTUAL is PROBABALISTIC
    cur_df_c3 = cur_df.copy()
    if os.path.exists(os.path.join(output_dir,("sub-"+subj_opt+"_dState_pPercept_outputsAAA.p"))):
        print("\tstate==Dichotomous & color==probabalistic Model already generated for this subject... moving on")
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_dState_pPercept_outputs.p")), 'rb') as handle:
            c_det_output3=pickle.load(handle)
    else:
        print("\tRunning state==Dichotomous & color==probabalistic Model now...")
        #                                     MPEModel_Dichotomous(tState, tProp, tResp, use_pState, use_pPercept)
        jd_d, sE_d, sT_d, mDist_d, pRange_d, cE_d = MPEModel_Dichotomous(c_tState, c_tProp, c_tResp, False, True)
        c_det_output3={'jd':jd_d, 'sE':sE_d, 'tE':sT_d, 'mDist':mDist_d, 'pRange':pRange_d, 'cE':cE_d} # save outputs in a dictionary
        with open(os.path.join(output_dir,("sub-"+subj_opt+"_dState_pPercept_outputs"+ext1)), 'wb') as handle:
            pickle.dump(c_det_output3, handle, protocol=4) 
    dSpP_jd = np.sum(c_det_output3['jd'], axis=0)
    #dSpP_pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    dSpP_pRange = c_det_output3['pRange']
    dSpP_theta = [dSpP_pRange['lfsp'][np.argmax(np.sum(dSpP_jd,axis=(1,2,3,4)))], 
                    dSpP_pRange['lfip'][np.argmax(np.sum(dSpP_jd,axis=(0,2,3,4)))], 
                    dSpP_pRange['fter'][np.argmax(np.sum(dSpP_jd,axis=(0,1,3,4)))], 
                    dSpP_pRange['ster'][np.argmax(np.sum(dSpP_jd,axis=(0,1,2,4)))], 
                    dSpP_pRange['diff_at'][np.argmax(np.sum(dSpP_jd,axis=(0,1,2,3,)))]]
    
    #                          ModelBeliefGeneration(thetas, tState, tProp, tResp, use_pState, use_pPercept)
    dSpP_sE_new, dSpP_tE_new, dSpP_cE_new  = ModelBeliefGeneration(dSpP_theta, c_tState, c_tProp, c_tResp, False, True)
    cur_df_c3['dSpP_sE'] = dSpP_sE_new[:,1] # add probability it is state 1
    cur_df_c3['dSpP_tE'] = dSpP_tE_new
    cur_df_c3['dSpP_cE'] = dSpP_cE_new
    
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSpP_belief-state"+ext1)), 'wb') as handle:
        pickle.dump(dSpP_sE_new, handle, protocol=4)
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSpP_belief-task"+ext1)), 'wb') as handle:
        pickle.dump(dSpP_tE_new, handle, protocol=4) 
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSpP_belief-color"+ext1)), 'wb') as handle:
        pickle.dump(dSpP_cE_new, handle, protocol=4) 
        
    # entropy for dSpP model
    dSpP_t_entropy, dSpP_t_entropy_change = calc_entropy(dSpP_sE_new, dSpP_cE_new)
    cur_df_c3['dSpP_entropy'] = dSpP_t_entropy 
    with open(os.path.join(output_dir,("sub-"+subj_opt+"_dSpP_entropy"+ext1)), 'wb') as handle:
        pickle.dump(dSpP_t_entropy, handle, protocol=4) 
    
    # save df with pSpP model estimates
    cur_df_c3.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_model-dSpP.csv'))
    
    
    # ----- make plots
    # -- model estimates
    c_sE = dSpP_sE_new[:,1] # [:,1] = probability state 1
    c_tE = 1 - dSpP_tE_new # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = dSpP_cE_new
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-ModelEstimates_model-dSpP"+ext3)), c_tResp[:200])
    # -- prediction error
    c_sE = np.abs(c_tState - dSpP_sE_new[:,1]) # [:,1] = probability state 1
    c_tE = np.abs(c_tTask - (1 - dSpP_tE_new))  # 1 - tE ... because tE = probability task 0 ... we want probability task 1
    c_cE = np.abs(c_tProp_bin - dSpP_cE_new)
    gen_model_plots(c_tState[:200], c_tTask[:200], c_tProp[:200], c_sE[:200], c_tE[:200], c_cE[:200], os.path.join(output_dir,(subj_opt+"_param-PredictionError_model-dSpP"+ext3)), c_tResp[:200])
    # -- theta parameters
    pRange_keys = ['lfsp', 'lfip', 'fter', 'ster', 'diff_at']
    pRange_titles = {'lfsp':'slope param', 'lfip':'intercept param', 'fter':'face error param', 'ster':'scene error param', 'diff_at':'diffusion param'}
    x_ranges = [[1.5, 5], [-3.0, 3.0], [0.0, 0.35], [0.0, 0.35], [0.0, 0.9]]
    pRange_colors = ['deepskyblue','steelblue','orange','green','red'] # set colors so I know what is what
    pRange = c_det_output3['pRange']
    mDist = c_det_output3['mDist']
    plt.rcParams["figure.figsize"] = (15,3)
    fig_thetas, ax_thetas = plt.subplots(1,5)
    for ind, cur_key in enumerate(pRange_keys):
        ax_thetas[ind].set_xlim(x_ranges[ind][0],x_ranges[ind][1])
        ax_thetas[ind].set_yticklabels([])
        ax_thetas[ind].plot(pRange[cur_key], mDist[cur_key], color=pRange_colors[ind])
        ax_thetas[ind].set_title(pRange_titles[cur_key])
    plt.show()
    plt.savefig(os.path.join(output_dir,("sub-"+subj_opt+"_model-dSpP__thetas"+ext3)))
    plt.close()

    #cur_df.to_csv(os.path.join(output_dir, 'sub-'+subj_opt+'_dataframe_withModelEstimates.csv'))