import os
import sys
import argparse
import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import nilearn
from nilearn import datasets
from nilearn.image import new_img_like, load_img, get_data, index_img
from nilearn.input_data import NiftiMasker
import nilearn.decoding
import nilearn.masking
from nilearn.image import resample_to_img, index_img
from nilearn import masking
from nilearn.image.resampling import coord_transform
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn._utils import check_niimg_4d
import nibabel as nib
import sklearn
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR, SVC, LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression, LinearRegression
import scipy.stats as stats
import random
from joblib import Parallel, delayed
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
Check based on reviewer comments
Making the following comparisons:
 state and entropy
 color and entropy
"""

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run different reviewer requested checks and extra comparisons",
        usage="[dataset_dir] [OPTIONS] ... ",
    )
    parser.add_argument("dataset_dir", help="/Shared/lss_kahwang_hpc/data/FPNHIU")
    parser.add_argument("--compare_state_color_entropy", help="compares parametric modulation R^2 values for state, color, and entropy",
                        default=False, action="store_true")
    parser.add_argument("--compare_statePE_colorPE_taskPE", help="compares parametric modulation R^2 values for state, color, and entropy",
                        default=False, action="store_true")
    parser.add_argument("--compare_state_and_color_decoding", help="compares decoding performance for state and color",
                        default=False, action="store_true")
    parser.add_argument("--compare_BIC", help="compares bic scores for probabilistic and binary decoding",
                        default=False, action="store_true")
    parser.add_argument("--correlation_checks", help="checks correlations between model estimated beliefs and uncertainties",
                        default=False, action="store_true")
    return parser

# ------------ Set Options --------------  compare_statePE_colorPE_taskPE
#generate_plots = False
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
dataset_dir = args.dataset_dir
func_path = os.path.join(args.dataset_dir,"3dDeconvolve/")
model_output_path = os.path.join(args.dataset_dir,"model_data/")
bids_path = os.path.join(args.dataset_dir,"BIDS/")
out_path = os.path.join(args.dataset_dir,"Decoding/")
if os.path.exists("/mnt/nfs/lss/lss_kahwang_hpc/"):
    mask_dir = "/mnt/nfs/lss/lss_kahwang_hpc/ROIs/"
else:
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs/"

def create_nii(stats_mat, cortical_mask):
    cortical_masker = NiftiMasker(cortical_mask)
    cortical_masker.fit()
    stat_nii = cortical_masker.inverse_transform(stats_mat)
    return stat_nii

# -- get subject lsit
unusable_subs = ['10118', '10218', '10275', '10282', '10296', '10318', '10319', '10321', '10322', '10351', '10358'] # '10313',  should be usable, but fb keeps crashing for some reason
subjects = sorted(os.listdir(bids_path)) # this folder only contains usable subjects
print(subjects)
subject_list = []
for ii, subj in enumerate(subjects):
    if os.path.isdir(os.path.join(func_path,subj)):
        cur_subj = subj.split('-')[1]
        if cur_subj not in unusable_subs:
                subject_list.append(cur_subj)
print("a total of ", len(subject_list), " subjects will be included")

# -- load mask
mask = nib.load(mask_dir + "CorticalMask_RSA_task-Quantum.nii.gz")
# -- load random func run to get dimensions
img = nib.load( func_path + ("sub-"+subject_list[0]) + "/cue__entropyR2_SPMGmodel_stats_REML.nii.gz" )
dims=img.get_fdata().shape
# -- apply mask
masked_data=nilearn.masking.apply_mask(img, mask) 
masked_data_dims = masked_data.shape
print("masked data dimensions: ", masked_data_dims[0])




if args.compare_statePE_colorPE_taskPE:
    # -- okay now loop through
    for event_idx, event_type in enumerate(["cue","probe"]):
        state_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        entropy_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        color_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        print("size of beta maps are ", state_maps.shape)
        
        # ---- loop through subjects to load decoded maps
        print("loading parametric modulation beta maps")
        for subj_idx, cur_subj in enumerate(subject_list):
            print("loading files for subject ", cur_subj)
            
            # -- load nii files
            entropy_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__taskPER2_SPMGmodel_stats_REML.nii.gz")) )
            state_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__statePER2_SPMGmodel_stats_REML.nii.gz")) )
            color_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__colorPER2_SPMGmodel_stats_REML.nii.gz")) )
            
            # -- apply mask now to get vector
            state_masked = nilearn.masking.apply_mask(state_nii, mask) 
            entropy_masked = nilearn.masking.apply_mask(entropy_nii, mask) 
            color_masked = nilearn.masking.apply_mask(color_nii, mask) 
            
            # -- save masked data in larger matrix
            state_maps[:,subj_idx] = state_masked
            entropy_maps[:,subj_idx] = entropy_masked
            color_maps[:,subj_idx] = color_masked
        
        state_entropy_group_stats = np.zeros((state_maps.shape[0],2)) # dim is tval and pval in that order
        color_entropy_group_stats = np.zeros((color_maps.shape[0],2)) # dim is tval and pval in that order
        color_state_group_stats = np.zeros((color_maps.shape[0],2)) # dim is tval and pval in that order
        print("looping through voxels and running t-tests")
        # ---- loop through voxels to run t-tests
        for v_idx in range( state_maps.shape[0] ):
            #state_group_stats[v_idx,0], state_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(state_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            state_entropy_group_stats[v_idx,0], state_entropy_group_stats[v_idx,1] = stats.ttest_ind(entropy_maps[v_idx,:], state_maps[v_idx,:], equal_var=False, alternative='two-sided')
        for v_idx in range( entropy_maps.shape[0] ):
            #task_group_stats[v_idx,0], task_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(task_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            color_entropy_group_stats[v_idx,0], color_entropy_group_stats[v_idx,1] = stats.ttest_ind(entropy_maps[v_idx,:], color_maps[v_idx,:], equal_var=False, alternative='two-sided')
        for v_idx in range( color_maps.shape[0] ):
            #color_group_stats[v_idx,0], color_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(color_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            color_state_group_stats[v_idx,0], color_state_group_stats[v_idx,1] = stats.ttest_ind(state_maps[v_idx,:], color_maps[v_idx,:], equal_var=False, alternative='two-sided')
        
        print("saving out group averaged files now")
        # ---- convert decoding ttests to nii file
        state_entropy_tnii = create_nii(state_entropy_group_stats[:,0], mask)
        state_entropy_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_taskPE-statePE__tval.nii")))
        
        color_entropy_tnii = create_nii(color_entropy_group_stats[:,0], mask)
        color_entropy_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_taskPE-colorPE__tval.nii")))
        
        color_state_tnii = create_nii(color_state_group_stats[:,0], mask)
        color_state_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_statePE-colorPE__tval.nii")))



if args.compare_state_color_entropy:
    # -- okay now loop through
    for event_idx, event_type in enumerate(["cue"]): #,"probe"
        state_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        entropy_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        color_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
        print("size of beta maps are ", state_maps.shape)
        
        # ---- loop through subjects to load decoded maps
        print("loading parametric modulation beta maps")
        for subj_idx, cur_subj in enumerate(subject_list):
            print("loading files for subject ", cur_subj)
            
            # -- load nii files
            entropy_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__entropyR2_SPMGmodel_stats_REML.nii.gz")) )
            state_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__stateR2_SPMGmodel_stats_REML.nii.gz")) )
            color_nii = nib.load( os.path.join(func_path, ("sub-"+cur_subj), (event_type+"__colorR2_SPMGmodel_stats_REML.nii.gz")) )
            
            # -- apply mask now to get vector
            state_masked = nilearn.masking.apply_mask(state_nii, mask) 
            entropy_masked = nilearn.masking.apply_mask(entropy_nii, mask) 
            color_masked = nilearn.masking.apply_mask(color_nii, mask) 
            
            # -- save masked data in larger matrix
            state_maps[:,subj_idx] = state_masked
            entropy_maps[:,subj_idx] = entropy_masked
            color_maps[:,subj_idx] = color_masked
        
        state_entropy_group_stats = np.zeros((state_maps.shape[0],2)) # dim is tval and pval in that order
        color_entropy_group_stats = np.zeros((color_maps.shape[0],2)) # dim is tval and pval in that order
        color_state_group_stats = np.zeros((color_maps.shape[0],2)) # dim is tval and pval in that order
        print("looping through voxels and running t-tests")
        # ---- loop through voxels to run t-tests
        for v_idx in range( state_maps.shape[0] ):
            #state_group_stats[v_idx,0], state_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(state_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            state_entropy_group_stats[v_idx,0], state_entropy_group_stats[v_idx,1] = stats.ttest_ind(entropy_maps[v_idx,:], state_maps[v_idx,:], equal_var=False, alternative='two-sided')
        for v_idx in range( entropy_maps.shape[0] ):
            #task_group_stats[v_idx,0], task_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(task_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            color_entropy_group_stats[v_idx,0], color_entropy_group_stats[v_idx,1] = stats.ttest_ind(entropy_maps[v_idx,:], color_maps[v_idx,:], equal_var=False, alternative='two-sided')
        for v_idx in range( color_maps.shape[0] ):
            #color_group_stats[v_idx,0], color_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(color_maps[v_idx,:]-0.5), popmean=0, nan_policy='omit')
            color_state_group_stats[v_idx,0], color_state_group_stats[v_idx,1] = stats.ttest_ind(state_maps[v_idx,:], color_maps[v_idx,:], equal_var=False, alternative='two-sided')
        
        print("saving out group averaged files now")
        # ---- convert decoding ttests to nii file
        state_entropy_tnii = create_nii(state_entropy_group_stats[:,0], mask)
        state_entropy_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_entropy-state__tval.nii")))
        
        color_entropy_tnii = create_nii(color_entropy_group_stats[:,0], mask)
        color_entropy_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_entropy-color__tval.nii")))
        
        color_state_tnii = create_nii(color_state_group_stats[:,0], mask)
        color_state_tnii.to_filename(os.path.join(func_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs_"+event_type+"_state-color__tval.nii")))

        # ---- get average t-value for sig. entropy clusters where R-squared values were compared between entropy and state/color
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(dataset_dir, "3dMEMA", "nii3D", "cue_zEntropy_masked.BRIK")) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        roi_data = np.where(np.squeeze(brik_data)>0,1,0) # binarize just in case its not already    
        roi_data = np.where(np.squeeze(mask.get_fdata())>0,roi_data,0) # apply cortical mask too
        mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
        # -- mask state-entropy t-values
        entropy_state_masked = nilearn.masking.apply_mask(state_entropy_tnii, mask_img)
        print("average t-value for entropy-state is ", entropy_state_masked.mean())
        # -- mask color-entropy t-values
        entropy_color_masked = nilearn.masking.apply_mask(color_entropy_tnii, mask_img)
        print("average t-value for entropy-color is ", entropy_color_masked.mean())
        # -- mask color-entropy t-values
        state_color_masked = nilearn.masking.apply_mask(color_state_tnii, mask_img)
        print("average t-value for color-state is ", state_color_masked.mean())
    
    

       
if args.correlation_checks:
    
    c_s_correlations = []
    cU_sU_correlations = []

    s_E_correlations = []
    c_E_correlations = []
    cU_E_correlations = []
    sU_E_correlations = []

    cPE_sPE_correlations = []
    tPE_sPE_correlations = []
    cPE_tPE_correlations = []

    # ------
    high_sU_9 = []
    high_sU_8 = []
    high_sU_7 = []
    high_sU_6 = []
    high_sU_5 = []
    high_sU_4 = []
    high_sU_3 = []
    high_sU_2 = []
    low_sU = []
    # ------

    for cur_sub in subject_list:
        df = pd.read_csv(model_output_path + "sub-"+cur_sub+"_dataframe_model-pSpP.csv")

        tProp = np.load(os.path.join(model_output_path,("sub-"+cur_sub+"_tProp.npy")))
        tProp = tProp[0:200]
        tProp_binary = tProp.round()
        #tProp = np.log(tProp / (1 - tProp))
        tState = np.load(os.path.join(model_output_path,("sub-"+cur_sub+"_tState.npy")))
        tState = tState[0:200]
        tTask = np.load(os.path.join(model_output_path,("sub-"+cur_sub+"_tTask.npy")))
        tTask = tTask[0:200]

        # pull out 3 latent variables + entropy
        tE = np.array(df['pSpP_tE'])[0:200] ## looks like 10283 did 6 runs? so there are 240 trials, cut it to 200
        sE = np.array(df['pSpP_sE'])[0:200]
        cE = np.array(df['pSpP_cE'])[0:200]
        entropy = np.array(df['pSpP_entropy'])[0:200]
        #e_log = np.log(entropy) # log transform entropy
        #runs = np.repeat([1,2,3,4,5], 40)

        sU = 2*(0.5 - np.abs(sE - 0.5)) # lessU 0 - 1 moreU
        cU = 2*(0.5 - np.abs(cE - 0.5)) # lessU 0 - 1 moreU
        
        correct = df['correct'][0:200]
        task_performed = np.zeros(200)
        for t_idx, tt in enumerate(tTask):
            if correct[t_idx]:
                task_performed[t_idx] = tTask[t_idx]
            else:
                task_performed[t_idx] = 1-tTask[t_idx]
            
        sPE = np.abs(tState - (np.array(df['pSpP_sE'])[0:200]))
        tPE = np.abs(tTask - (1 - np.array(df['pSpP_tE'])[0:200]))
        cPE = np.abs(tProp_binary - (np.array(df['pSpP_cE'])[0:200]))

        # ---- check correlations between state and color UNCERTAINTY
        c_s_correlations.append(np.corrcoef(sE, cE)[0,1])
        
        # ---- check correlations between state and color UNCERTAINTY
        cU_sU_correlations.append(np.corrcoef(sU, cU)[0,1])
        
        # ---- check correlations between state and ENTROPY
        s_E_correlations.append(np.corrcoef(sE, entropy)[0,1]) 
        
        # ---- check correlations between color and ENTROPY
        c_E_correlations.append(np.corrcoef(entropy, cE)[0,1])
        
        # ---- check correlations between state UNCERTAINTY and ENTROPY
        sU_E_correlations.append(np.corrcoef(sU, entropy)[0,1]) 
        
        # ---- check correlations between color UNCERTAINTY and ENTROPY
        cU_E_correlations.append(np.corrcoef(entropy, cU)[0,1])
        
        # ---- check correlation between state and color PEs
        cPE_sPE_correlations.append(np.corrcoef(sPE, cPE)[0,1])
        # ---- check correlation between state and task PEs
        tPE_sPE_correlations.append(np.corrcoef(sPE, tPE)[0,1])
        # ---- check correlation between task and color PEs
        cPE_tPE_correlations.append(np.corrcoef(tPE, cPE)[0,1])
        
        # ------
        high_sU_9.append(len(sU[sU>.9]))
        high_sU_8.append(len(sU[sU>.8]))
        high_sU_7.append(len(sU[sU>.7]))
        high_sU_6.append(len(sU[sU>.6]))
        high_sU_5.append(len(sU[sU>.5]))
        high_sU_4.append(len(sU[sU>.4]))
        high_sU_3.append(len(sU[sU>.3]))
        high_sU_2.append(len(sU[sU>.2]))
        low_sU.append(len(sU[sU<.1]))
        # ------

    df_corr = pd.DataFrame({'color_state':c_s_correlations,
                            'colorUncertainty_stateUncertainty':cU_sU_correlations, 
                            'Entropy_state':s_E_correlations,
                            'Entropy_color':c_E_correlations,
                            'Entropy_stateUncertainty':sU_E_correlations, 
                            'Entropy_colorUncertainty':cU_E_correlations, 
                            'colorPE_statePE':cPE_sPE_correlations, 
                            'taskPE_statePE':tPE_sPE_correlations, 
                            'colorPE_taskPE':cPE_tPE_correlations, 
                            'sub':subject_list})

    df_corr.to_csv(os.path.join(model_output_path, "correlation_checks.csv"))

    df_sU = pd.DataFrame({'sub':subject_list, 'sU_gt-9_trial_num':high_sU_9, 'sU_gt-8_trial_num':high_sU_8, 'sU_gt-7_trial_num':high_sU_7, 'sU_gt-6_trial_num':high_sU_6, 
                        'sU_gt-5_trial_num':high_sU_5, 'sU_gt-4_trial_num':high_sU_4, 'sU_gt-3_trial_num':high_sU_3, 'sU_gt-2_trial_num':high_sU_2, 'sU_lt-1_trial_num':low_sU})
    df_sU.to_csv(os.path.join(model_output_path, "state_uncertainty_trial_counts.csv"))




def create_nii_v2(ckey, results, mask):
    cur_nii_data = np.zeros(len(results))
    for i in np.arange(len(results)):
        cur_nii_data[i] = results[i][ckey]
    cur_nii = masking.unmask(cur_nii_data, mask)
    return cur_nii

if args.compare_BIC:
    # load schaefer 400 roi mask so we can grab region V4 as requested
    # -- load mask
    schaefer_mask = nib.load(mask_dir + "Schaefer400_RSA_2mm.nii.gz") # 7 network
    schaefer_masked = nilearn.masking.apply_mask(schaefer_mask, mask)
    schaefer_rois = {'v4_RH':[204, 216, 226], 
                   'v4_LH':[4, 16, 26],
                   'v5_RH_1':[205], 'v5_LH_1':[5],
                   'dlPFC_RH_1': [345], 'dlPFC_LH_1': [139],
                   'OFC_RH_1': [319, 320, 321, 322, 323, 324],
                   'OFC_LH_1': [114, 115, 116, 117, 118]}
    
    roi_mask_dict={'state':'State', 'color':'Color'}
    for epoch in ["cue"]:
        for predicted in ["color","state"]:
            OLS_bin_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            OLS_prob_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            Logit_bin_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            Logit_prob_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            bmc_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            binary_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            probabilistic_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            binary_HC_maps = np.zeros( (schaefer_masked.shape[0],len(subject_list)) ) # voxels by subjects
            
            print("loading decoding maps with BIC scores") # state_cue_searchlight_decoding-probabilistic__OLS_bic.nii.gz
            for subj_idx, cur_subj in enumerate(subject_list):
                if int(cur_subj)!=int(10306):
                    # -- load nii
                    OLS_bin_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__OLS_bic.nii.gz")) )
                    OLS_prob_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-probabilistic__OLS_bic.nii.gz")) )
                    Logit_bin_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__Logit_bic.nii.gz")) )
                    Logit_prob_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-probabilistic__Logit_bic.nii.gz")) )
                    bmc_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__bimodality_coefficient_relative_to_fiveninths.nii.gz")) )
                    binary_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__score.nii.gz")) )
                    probabilistic_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-probabilistic__r.nii.gz")) )
                    binary_HC_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary-HC__score.nii.gz")) )

                    # -- apply mask
                    OLS_bin_masked = nilearn.masking.apply_mask(OLS_bin_nii, mask)
                    OLS_prob_masked = nilearn.masking.apply_mask(OLS_prob_nii, mask)
                    Logit_bin_masked = nilearn.masking.apply_mask(Logit_bin_nii, mask)
                    Logit_prob_masked = nilearn.masking.apply_mask(Logit_prob_nii, mask)
                    bmc_masked = nilearn.masking.apply_mask(bmc_nii, mask) # nilearn.masking.apply_mask(bmc_nii, mask_nif)
                    binary_nii_masked = nilearn.masking.apply_mask(binary_nii, mask)
                    probabilistic_nii_masked = nilearn.masking.apply_mask(probabilistic_nii, mask)
                    binary_HC_nii_masked = nilearn.masking.apply_mask(binary_HC_nii, mask)
                    
                    # -- save masked data in larger matrix
                    OLS_bin_maps[:,subj_idx] = OLS_bin_masked
                    OLS_prob_maps[:,subj_idx] = OLS_prob_masked
                    Logit_bin_maps[:,subj_idx] = Logit_bin_masked
                    Logit_prob_maps[:,subj_idx] = Logit_prob_masked
                    bmc_maps[:,subj_idx] = bmc_masked
                    binary_maps[:,subj_idx] = binary_nii_masked
                    probabilistic_maps[:,subj_idx] = probabilistic_nii_masked
                    binary_HC_maps[:,subj_idx] = binary_HC_nii_masked
                
                    # results = pickle.load(open(os.path.join(out_path, "sub-"+str(cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary_results.p")), "rb"))
                    # for ckey in ['bimodality_coefficient_relative_to_fiveninths']:
                    #     cur_nii = create_nii_v2(ckey, results, mask)
                    #     cur_nii.to_filename(os.path.join(out_path, "sub-"+str(cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__"+ckey+".nii.gz")))
                    #     bmc_masked = nilearn.masking.apply_mask(cur_nii, mask)
                    #     bmc_maps[:,subj_idx] = bmc_masked
                
            # # run t-tests for high certainty binary decoding
            # group_stats = np.zeros((binary_HC_maps.shape[0],2)) # dim is tval and pval in that order
            # print("looping through voxels and running t-tests")
            # # ---- loop through voxels to run t-tests
            # for v_idx in range( binary_HC_maps.shape[0] ):
            #     group_stats[v_idx,0], group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(binary_HC_maps[v_idx,:]), popmean=0.5, nan_policy='omit')
            # print("saving out group averaged files for ", predicted," and ", epoch," now")
            # # ---- convert decoding ttests to nii file
            # group_tnii = create_nii(group_stats[:,0], mask)
            # group_tnii.to_filename(os.path.join(out_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__score-HC__tval.nii")))
            # # ---- get average coeff (avg. beta) from within subject searchlight method
            # state_avg_coeff=np.mean(binary_HC_maps, axis=1) # 2nd dim is subjects
            # state_group_avg_nii = create_nii(state_avg_coeff, mask)
            # state_group_avg_nii.to_filename(os.path.join(out_path, ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__score-HC__avg_coeff.nii")))
            
            results = {'roi':[], 'regression_type':[], 'bic':[]}
            results_bmc = {'roi':[], 'bmc':[], 'score':[], 'score_tval':[], 'score_pval':[], 'r':[], 'r_tval':[], 'r_pval':[], 'HC_score':[], 'HC_score_tval':[], 'HC_score_pval':[], 'binary_comparison_HC-AT_tval':[], 'binary_comparison_HC-AT_pval':[]}
            for s_roi in schaefer_rois.keys():
                OLS_binary_bic = 0
                OLS_prob_bic = 0
                Logit_binary_bic = 0
                Logit_prob_bic = 0
                bimod_coef = []
                binary_score = []
                prob_r = []
                binary_HC_score = []
                binary_score_ttest = []
                prob_r_ttest = []
                
                binary_tval_list = []
                binary_pval_list = []
                prob_tval_list = []
                prob_pval_list = []
                binary_HC_tval_list = []
                binary_HC_pval_list = []
                binary2_tval_list = []
                binary2_pval_list = []
                
                for c_roi in schaefer_rois[s_roi]:
                    OLS_binary_bic += np.sum(np.mean(OLS_bin_maps[(schaefer_masked==c_roi),:], axis=0))
                    OLS_prob_bic += np.sum(np.mean(OLS_prob_maps[(schaefer_masked==c_roi),:], axis=0))
                    Logit_binary_bic += np.sum(np.mean(Logit_bin_maps[(schaefer_masked==c_roi),:], axis=0))
                    Logit_prob_bic += np.sum(np.mean(Logit_prob_maps[(schaefer_masked==c_roi),:], axis=0))
                    
                    bimod_coef.append(np.mean(bmc_maps[(schaefer_masked==c_roi),:]))
                    #print(bimod_coef)
                    
                    binary_score.append(np.mean(np.mean(binary_maps[(schaefer_masked==c_roi),:], axis=0)))
                    prob_r.append(np.mean(np.mean(probabilistic_maps[(schaefer_masked==c_roi),:], axis=0)))
                    binary_HC_score.append(np.mean(np.mean(binary_maps[(schaefer_masked==c_roi),:], axis=0)))
                    
                    binary_score_ttest = binary_maps[(schaefer_masked==c_roi),:] 
                    binary_tvals = np.zeros(binary_score_ttest.shape[0])
                    binary_pvals = np.zeros(binary_score_ttest.shape[0])
                    for vox in range(binary_score_ttest.shape[0]):
                        binary_tvals[vox], binary_pvals[vox] = stats.ttest_1samp(np.asarray(binary_score_ttest)[vox,:], popmean=0.5, nan_policy='omit')
                    binary_tval_list.append(binary_tvals.mean())
                    binary_pval_list.append(binary_pvals.mean())
                    
                    prob_r_ttest = probabilistic_maps[(schaefer_masked==c_roi),:] 
                    prob_tvals = np.zeros(prob_r_ttest.shape[0])
                    prob_pvals = np.zeros(prob_r_ttest.shape[0])
                    for vox in range(prob_r_ttest.shape[0]):
                        prob_tvals[vox], prob_pvals[vox] = stats.ttest_1samp(np.asarray(prob_r_ttest)[vox,:], popmean=0.0, nan_policy='omit')
                    prob_tval_list.append(prob_tvals.mean())
                    prob_pval_list.append(prob_pvals.mean())
                    
                    binary_HC_score_ttest = binary_HC_maps[(schaefer_masked==c_roi),:] 
                    binary_HC_tvals = np.zeros(binary_HC_score_ttest.shape[0])
                    binary_HC_pvals = np.zeros(binary_HC_score_ttest.shape[0])
                    for vox in range(binary_HC_score_ttest.shape[0]):
                        binary_HC_tvals[vox], binary_HC_pvals[vox] = stats.ttest_1samp(np.asarray(binary_HC_score_ttest)[vox,:], popmean=0.5, nan_policy='omit')
                    binary_HC_tval_list.append(binary_HC_tvals.mean())
                    binary_HC_pval_list.append(binary_HC_pvals.mean())
                    
                    # add direct comparison between binary with all trials and binary with only high certainty
                    t_statistics = np.zeros(binary_HC_score_ttest.shape[0])
                    p_values = np.zeros(binary_HC_score_ttest.shape[0])
                    for vox in range(binary_HC_score_ttest.shape[0]):
                        t_statistics[vox], p_values[vox] = stats.ttest_ind(np.asarray(binary_HC_score_ttest)[vox,:], np.asarray(binary_score_ttest)[vox,:], equal_var=False, alternative='two-sided')
                    binary2_tval_list.append(t_statistics.mean())
                    binary2_pval_list.append(p_values.mean())
                    
                results['roi'].append(s_roi)
                results['regression_type'].append('OLS_binary')
                results['bic'].append(OLS_binary_bic)
                results['roi'].append(s_roi)
                results['regression_type'].append('OLS_probabilistic')
                results['bic'].append(OLS_prob_bic)
                results['roi'].append(s_roi)
                results['regression_type'].append('Logit_binary')
                results['bic'].append(Logit_binary_bic)
                results['roi'].append(s_roi)
                results['regression_type'].append('Logit_probabilistic')
                results['bic'].append(Logit_prob_bic)
                
                results_bmc['roi'].append(s_roi)
                results_bmc['bmc'].append(np.asarray(bimod_coef).mean())
                results_bmc['score'].append(np.asarray(binary_score).mean())
                results_bmc['r'].append(np.asarray(prob_r).mean())
                results_bmc['HC_score'].append(np.asarray(binary_HC_score).mean())
                
                # run t-tests
                results_bmc['score_tval'].append(np.asarray(binary_tval_list).mean())
                results_bmc['score_pval'].append(np.asarray(binary_pval_list).mean())
                
                results_bmc['r_tval'].append(np.asarray(prob_tval_list).mean())
                results_bmc['r_pval'].append(np.asarray(prob_pval_list).mean())
                
                results_bmc['HC_score_tval'].append(np.asarray(binary_HC_tval_list).mean())
                results_bmc['HC_score_pval'].append(np.asarray(binary_HC_pval_list).mean())
                
                results_bmc['binary_comparison_HC-AT_tval'].append(np.asarray(binary2_tval_list).mean())
                results_bmc['binary_comparison_HC-AT_pval'].append(np.asarray(binary2_pval_list).mean())
            
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(out_path, "bic_comparisons_"+predicted+"_"+epoch+".csv"))
            
            df_bmc = pd.DataFrame(results_bmc)
            df_bmc.to_csv(os.path.join(out_path, "bmc_comparisons_"+predicted+"_"+epoch+".csv"))    
                    

    
    # roi_mask_dict={'state':'State', 'color':'Color'}
    # for epoch in ["cue"]:
    #     for predicted in ["state", "color"]: # ["state", "task", "color", "entropy"]:
    #         OLS_bin_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
    #         OLS_prob_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
    #         Logit_bin_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
    #         Logit_prob_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
    #         bmc_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
            
    #         # load sig state or color decoding regions
    #         # -- load current roi mask
    #         brik_mask = nib.load(os.path.join(out_path, "GroupStats", "sig_roi_masks", epoch, (cur_reg + "_mask+orig.BRIK"))) # load sig rois for state or color
    #         brik_data = brik_mask.get_fdata()
    #         print("max roi value for current brik", brik_data.max())
    #         brik_data = np.where(brik_data>1,1,0) # binarize to grab all rois
    #         #brik_data = np.squeeze(brik_data)
    #         mask_img = nib.Nifti1Image(brik_data, brik_mask.affine, brik_mask.header)
    #         bmc_maps = np.zeros( (brik_data.sum(), len(subject_list)) )
            
            
    #         print("loading decoding maps with BIC scores") # state_cue_searchlight_decoding-probabilistic__OLS_bic.nii.gz
    #         for subj_idx, cur_subj in enumerate(subject_list):
    #             # -- load nii
    #             OLS_bin_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__OLS_bic.nii.gz")) )
    #             OLS_prob_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-probabilistic__OLS_bic.nii.gz")) )
    #             Logit_bin_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-binary__Logit_bic.nii.gz")) )
    #             Logit_prob_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_decoding-probabilistic__Logit_bic.nii.gz")) )
    #             bmc_nii = nib.load( os.path.join(out_path, "Binary_Decoding", ("sub-"+cur_subj), (epoch+"_"+predicted+"__svc_bimodcoeff.nii.gz")) )

    #             # -- apply mask
    #             OLS_bin_masked = nilearn.masking.apply_mask(OLS_bin_nii, mask_nif)
    #             OLS_prob_masked = nilearn.masking.apply_mask(OLS_prob_nii, mask_nif)
    #             Logit_bin_masked = nilearn.masking.apply_mask(Logit_bin_nii, mask_nif)
    #             Logit_prob_masked = nilearn.masking.apply_mask(Logit_prob_nii, mask_nif)
    #             bmc_masked = nilearn.masking.apply_mask(bmc_nii, mask_img) # nilearn.masking.apply_mask(bmc_nii, mask_nif)

    #             # -- save masked data in larger matrix
    #             OLS_bin_maps[:,subj_idx] = OLS_bin_masked
    #             OLS_prob_maps[:,subj_idx] = OLS_prob_masked
    #             Logit_bin_maps[:,subj_idx] = Logit_bin_masked
    #             Logit_prob_maps[:,subj_idx] = Logit_prob_masked
    #             bmc_maps[:,subj_idx] = bmc_masked
                
    #             # make histogram plots for this subject
    #             plt.close()
    #             plt.hist(bmc_masked)
    #             plt.title(cur_subj)
    #             plt.savefig(os.path.join(out_path, "Binary_Decoding", "histograms", ("sub-"+cur_subj+"__"+epoch+"_"+predicted+"__onlySigClusters__decision_scores_hist.png")))
                
    #         # make histogram plots for this subject
    #         plt.hist(bmc_maps.flatten())
    #         plt.title("All 38 Subjects")
    #         plt.savefig(os.path.join(out_path, "Binary_Decoding", "histograms", "all38subjects__"+epoch+"_"+predicted+"__onlySigClusters__decision_scores_hist.png"))
            
    #         # -- set up sum matrices
    #         bin_OLS_sum = np.sum(OLS_bin_maps, axis=1)
    #         prob_OLS_sum = np.sum(OLS_prob_maps, axis=1)
    #         bin_Logit_sum = np.sum(Logit_bin_maps, axis=1)
    #         prob_Logit_sum = np.sum(Logit_prob_maps, axis=1)
            
    #         # ---- convert to nii file
    #         print("saving out summed OLS BIC files for ", predicted," and ", epoch," now")
    #         OLS_sum_nii = create_nii(bin_OLS_sum, mask)
    #         OLS_sum_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-bic__binary_sum.nii")))
    #         OLS_sum_nii = create_nii(prob_OLS_sum, mask)
    #         OLS_sum_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-bic__probabilistic_sum.nii")))
    #         print("saving out summed Logit BIC files for ", predicted," and ", epoch," now")
    #         Logit_sum_nii = create_nii(bin_Logit_sum, mask)
    #         Logit_sum_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-bic__binary_sum.nii")))
    #         Logit_sum_nii = create_nii(prob_Logit_sum, mask)
    #         Logit_sum_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-bic__probabilistic_sum.nii")))
            
    #         # -- set up group stats matrix
    #         OLS_group_stats = np.zeros((OLS_prob_maps.shape[0],3)) # dim is t-val and p-val and (prob bic - bin bic) in that order
    #         Logit_group_stats = np.zeros((OLS_prob_maps.shape[0],3)) # dim is t-val and p-val and (prob bic - bin bic) in that order
    #         Lp_Ob_group_stats = np.zeros((OLS_prob_maps.shape[0],3)) # dim is t-val and p-val and (prob bic - bin bic) in that order
    #         Op_Lb_group_stats = np.zeros((OLS_prob_maps.shape[0],3)) # dim is t-val and p-val and (prob bic - bin bic) in that order
    #         bmc_group_stats = np.zeros((bmc_maps.shape[0],2))
            
    #         print("looping through voxels and running t-tests")
    #         # ---- loop through voxels to run t-tests
    #         for v_idx in range( masked_data_dims[0] ):
    #             # - - - -   OLS
    #             OLS_group_stats[v_idx,0], OLS_group_stats[v_idx,1] = stats.ttest_ind(OLS_prob_maps[v_idx,:], OLS_bin_maps[v_idx,:], equal_var=False, alternative='two-sided')
    #             OLS_group_stats[v_idx,2] = np.sum(OLS_prob_maps[v_idx,:]) - np.sum(OLS_bin_maps[v_idx,:]) # difference between summed bic across 38 participants
                
    #             # - - - -   Logit
    #             Logit_group_stats[v_idx,0], Logit_group_stats[v_idx,1] = stats.ttest_ind(Logit_prob_maps[v_idx,:], Logit_bin_maps[v_idx,:], equal_var=False, alternative='two-sided')
    #             Logit_group_stats[v_idx,2] = np.sum(Logit_prob_maps[v_idx,:]) - np.sum(Logit_bin_maps[v_idx,:]) # difference between summed bic across 38 participants
                
    #             # - - - -   Mixture
    #             #Logit_group_stats[v_idx,2] = np.sum(Logit_prob_maps[v_idx,:]) - np.sum(Logit_bin_maps[v_idx,:]) # difference between summed bic across 38 participants
    #             Lp_Ob_group_stats[v_idx,2] = np.sum(Logit_prob_maps[v_idx,:]) - np.sum(OLS_bin_maps[v_idx,:]) # difference between summed bic across 38 participants
    #             Op_Lb_group_stats[v_idx,2] = np.sum(OLS_prob_maps[v_idx,:]) - np.sum(Logit_bin_maps[v_idx,:]) # difference between summed bic across 38 participants
                
    #             # - - - -   Bimodal Coefficient
    #             bmc_group_stats[v_idx,0], bmc_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(bmc_maps[v_idx,:]), popmean=0, nan_policy='omit')
            
    #         # - - - -   OLS
    #         print("saving out OLS BIC files for ", predicted," and ", epoch," now")
    #         # ---- convert decoding ttests to nii file
    #         group_tnii = create_nii(OLS_group_stats[:,0], mask)
    #         group_tnii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-bic__tval.nii")))
    #         # ---- convert decoding ttests to nii file
    #         group_pnii = create_nii(OLS_group_stats[:,1], mask)
    #         group_pnii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-bic__pval.nii")))
    #         # ---- get average coeff (avg. beta) from within subject searchlight method
    #         group_diff_nii = create_nii(OLS_group_stats[:,2], mask)
    #         group_diff_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-bic__diff_P-B.nii")))
            
    #         # - - - -   Logit
    #         print("saving out Logit BIC files for ", predicted," and ", epoch," now")
    #         # ---- convert decoding ttests to nii file
    #         group_tnii = create_nii(Logit_group_stats[:,0], mask)
    #         group_tnii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-bic__tval.nii")))
    #         # ---- convert decoding ttests to nii file
    #         group_pnii = create_nii(Logit_group_stats[:,1], mask)
    #         group_pnii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-bic__pval.nii")))
    #         # ---- get average coeff (avg. beta) from within subject searchlight method
    #         group_diff_nii = create_nii(Logit_group_stats[:,2], mask)
    #         group_diff_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-bic__diff_P-B.nii")))
            
    #         # - - - -   Mixture
    #         print("saving out mixture BIC files for ", predicted," and ", epoch," now")
    #         # ---- get average coeff (avg. beta) from within subject searchlight method
    #         group_diff_nii = create_nii(Lp_Ob_group_stats[:,2], mask)
    #         group_diff_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__Logit-prob_OLS-bin__bic_diff.nii")))
    #         # ---- get average coeff (avg. beta) from within subject searchlight method
    #         group_diff_nii = create_nii(Op_Lb_group_stats[:,2], mask)
    #         group_diff_nii.to_filename(os.path.join(final_output_path, "GroupStats", "non-smoothed", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__OLS-prob_Logit-bin__bic_diff.nii")))
            
    #         # # - - - -   Bimodal Coefficient
    #         print("saving out bimodal coefficient files for ", predicted," and ", epoch," now")
    #         # ---- convert decoding ttests to nii file
    #         group_tnii = create_nii(bmc_group_stats[:,0], mask)
    #         group_tnii.to_filename(os.path.join(final_output_path, "Binary_Decoding", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__bimodcoeff__tval.nii")))
    #         # ---- convert decoding ttests to nii file
    #         group_pnii = create_nii(bmc_group_stats[:,1], mask)
    #         group_pnii.to_filename(os.path.join(final_output_path, "Binary_Decoding", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__bimodcoeff__pval.nii")))
            
    




if args.compare_state_and_color_decoding:
    import glob
    results_dict = {'epoch':[], 'sig_clust_from':[], 'clust_num':[], 'num_voxels':[], 'state_average':[], 'color_average':[], 't-value':[], 'p-value':[]}
    
    for epoch in ['cue','probe']:
        print("working on ", epoch)
        # load state and color decoding maps for current epoch   GroupAnalysis_38subjs__color_cue__r__avg_coeff.nii
        color_nii = nib.load( os.path.join(out_path, "GroupStats", ("GroupAnalysis_38subjs__color_"+epoch+"__r__avg_coeff.nii")) )
        state_nii = nib.load( os.path.join(out_path, "GroupStats", ("GroupAnalysis_38subjs__state_"+epoch+"__r__avg_coeff.nii")) )
        
        mask_list = ["State", "Color"]
        for cur_reg in mask_list:
            print("\tgrabbing masks for ", cur_reg)
            
            # -- load current roi mask
            brik_mask = nib.load(os.path.join(out_path, "GroupStats", "sig_roi_masks", epoch, (cur_reg + "_mask+orig.BRIK"))) # load sig rois for state or color
            brik_data = brik_mask.get_fdata()
            max_roi = int(brik_data.max())
            print("max roi value for current brik", max_roi)
            
            for idx in range(max_roi):
                if idx==0:
                    # for the first roi just use the whole brain
                    roi_data = np.where(brik_data>1,1,0) # binarize just in case its not already
                else:
                    # use only current roi
                    roi_data = np.where(brik_data==(idx+1),1,0) # binarize just in case its not already
                roi_data = np.squeeze(roi_data)
                mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
                
                print("\t\tworking on mask num ", str(idx+1), " out of ", max_roi)
                results_dict['epoch'].append(epoch)
                results_dict['sig_clust_from'].append(cur_reg)
                results_dict['clust_num'].append(idx+1)
                
                # -- apply mask to state
                masked_state = nilearn.masking.apply_mask(state_nii, mask_img) # should be vector of voxels in current mask
                masked_state = np.abs(masked_state) # we only care about magnitude, not sign
                # -- apply mask to color
                masked_color = nilearn.masking.apply_mask(color_nii, mask_img) # should be vector of voxels in current mask
                masked_color = np.abs(masked_color) # we only care about magnitude, not sign
                
                if len(masked_state) != len(masked_color):
                    print("ERROR!!!! state and color should match... issue with code")
                else:
                    results_dict['num_voxels'].append(len(masked_color)) # since they're the same just use one to calculate
                
                # -- also save average value for state and color
                results_dict['state_average'].append(masked_state.mean())
                results_dict['color_average'].append(masked_color.mean())
                
                # -- run t-test between t-values from state and color
                t_statistic, p_value = stats.ttest_ind(masked_state, masked_color, equal_var=False, alternative='two-sided')
                results_dict['t-value'].append(t_statistic) # since they're the same just use one to calculate
                results_dict['p-value'].append(p_value) # since they're the same just use one to calculate
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv( os.path.join(out_path, "GroupStats", "sig_roi_masks", "state_color_decoding_ttest_results.csv") )