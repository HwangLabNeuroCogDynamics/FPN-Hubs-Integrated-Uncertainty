# try searchlight decoding to classify state and task info.
# based on https://nilearn.github.io/dev/auto_examples/02_decoding/plot_haxby_searchlight.html#sphx-glr-auto-examples-02-decoding-plot-haxby-searchlight-py
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
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn._utils import check_niimg_4d

from nilearn import masking, image
#from nilearn._utils.niimg_conversions import ( check_niimg_3d, safe_get_data, )
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.image.resampling import coord_transform

import nibabel as nib
import sklearn
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR, SVC, LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression, LinearRegression

import scipy.stats as stats
from scipy.stats import skew, kurtosis
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

import statsmodels.api as sm
import random
from joblib import Parallel, delayed
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode subject",
        usage="[Subject] [dataset_dir] [OPTIONS] ... ",
    )
    parser.add_argument("--subject", help="id of subject (i.e., 10001)")
    parser.add_argument("--dataset_dir", help="path of dataset")
    parser.add_argument("--ttest", help="compare subject decoding estimates to 0",
                        default=False, action="store_true")
    parser.add_argument("--binary", help="compare subject decoding estimates to 0",
                        default=False, action="store_true")
    return parser

parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
subject = args.subject
dataset_dir = args.dataset_dir
#cur_sub = "10283"

bids_path = os.path.join(dataset_dir, "BIDS/")
func_path = os.path.join(dataset_dir, "3dDeconvolve/")
model_output_path = os.path.join(dataset_dir, "model_data/")
out_path = os.path.join(dataset_dir, "Decoding/")
if os.path.exists("/mnt/nfs/lss/lss_kahwang_hpc/"):
    mask_dir = "/mnt/nfs/lss/lss_kahwang_hpc/ROIs/"
elif os.path.exists("/Shared/lss_kahwang_hpc/"):
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs/"


                     
# ------------------------------------------------------------------------
# Fast affinity function adapted from nilearn
# ------------------------------------------------------------------------
def new_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
    """
    Build a sparse sphere-to-voxel affinity matrix.

    seeds      : array-like of shape (n_seeds, 3) in world (mm) coordinates
    niimg      : Nifti1Image of 4D data (for affine) or None
    radius     : float, searchlight radius in mm
    allow_overlap: bool, if False, raises on overlapping spheres
    mask_img   : Nifti1Image or filename of 3D binary mask

    Returns:
      X         : data matrix from mask_img & niimg (unused here)
      A_sparse  : lil_matrix (n_seeds x n_voxels) boolean
    """
    seeds = np.asarray(seeds)
    if seeds.ndim == 1:
        seeds = seeds[np.newaxis, :]

    # Load or derive mask coordinates and optional data
    if niimg is None:
        mask, affine = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask))
        X = None
    else:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation="nearest",
            force_resample=False,
        )
        mask, _ = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask != 0))
        X = masking.apply_mask_fmri(niimg, mask_img)

    # Convert mask voxel indices to world space
    mask_coords_world = coord_transform(
        mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], affine
    )
    mask_coords_world = np.column_stack(mask_coords_world)

    # Map seeds from world → voxel indices
    inv_affine = np.linalg.inv(affine)
    seeds_voxel = (
        np.column_stack([seeds, np.ones(len(seeds))]) @ inv_affine.T
    )[:, :3]
    nearest_voxels = np.round(seeds_voxel).astype(int)
    coord_dict = {tuple(c): i for i, c in enumerate(mask_coords)}
    nearests = [coord_dict.get(tuple(coord), None) for coord in nearest_voxels]

    # KDTree for radius queries
    tree = neighbors.KDTree(mask_coords_world)
    neighbors_list = tree.query_radius(seeds, r=radius)

    n_seeds = len(seeds)
    n_voxels = len(mask_coords)
    A_sparse = sparse.lil_matrix((n_seeds, n_voxels), dtype=bool)

    # Fill boolean matrix
    for i, (nbrs, nearest) in enumerate(zip(neighbors_list, nearests)):
        if nbrs.size > 0:
            A_sparse[i, nbrs] = True
        if nearest is not None:
            A_sparse[i, nearest] = True

    # Ensure seed itself is present
    seed_world = coord_transform(
        seeds_voxel[:, 0], seeds_voxel[:, 1], seeds_voxel[:, 2], affine
    )
    seed_world = np.column_stack(seed_world)
    _, seed_idx = tree.query(seed_world, k=1)
    for i, idx in enumerate(seed_idx):
        A_sparse[i, idx] = True

    # Sanity checks
    sphere_sizes = np.asarray(A_sparse.sum(axis=1)).ravel()
    empty = np.nonzero(sphere_sizes == 0)[0]
    if len(empty):
        raise ValueError(f"Empty spheres: {empty}")
    if not allow_overlap and np.any(A_sparse.sum(axis=0) >= 2):
        raise ValueError("Overlap detected between spheres")

    return X, A_sparse.tocsr()

#  searchlight_prediction_job_BINARY(sphere_idx, A, X, Y, certain_trials, belief, use_all_trials)
def searchlight_prediction_job_BINARY(sphere_idx, A, X, Y, certain_trials, y_actual, use_all_trials):
    """
    For a given sphere (defined by voxel indices), run searchlight classification.
    Uses a pipeline (StandardScaler + LogisticRegression) with a fixed cross-validation
    scheme (assumes 200 trials split into 5 groups).

    Parameters:
      sphere_voxel_inds: Indices of voxels in the sphere.
      sphere_idx: Index of the sphere.
      X: fMRI data matrix (n_samples x n_voxels).
      Y: Hard labels for classification.
      belief_subset: The subset of trials to use with decoding (only using high certainty/low uncertainty trials)
      per_group_num: number of trials per group to use for decoding
      max_Y: max trial value to use to keep groups even

    Returns:
      A dictionary containing decoding score and the sphere index.
    """
    results = {}
    #use_all_trials = True
    
    # get sphere_voxel_inds
    sphere_voxel_inds = A[sphere_idx].indices
    
    # Extract the searchlight data (all beta estimates for the voxels in this sphere)
    searchlight_data = X[:, sphere_voxel_inds]
    
    if use_all_trials:
        y_train = Y
        c_searchlight_data = searchlight_data
        per_group_num = 40 # reset to the full number of trials per run
    else:
        # reduce Y and X to only subset of trials to use
        y_train = Y[certain_trials]
        y_actual = y_actual[certain_trials]
        c_searchlight_data = searchlight_data[certain_trials]
        per_group_num = [np.sum(1.0*certain_trials[:40]),
                        np.sum(1.0*certain_trials[40:80]),
                        np.sum(1.0*certain_trials[80:120]),
                        np.sum(1.0*certain_trials[120:160]),
                        np.sum(1.0*certain_trials[160:])]
    
    del searchlight_data, Y, X
    
    #pipeline = make_pipeline(StandardScaler(), LinearRegression()) # Standardize features by removing the mean and scaling to unit variance
    pipeline = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ~~ StandardScaler ~~ pipeline steps according to stackoveflow:
    #   Step 0: The data are split into TRAINING data and TEST data according to the cv parameter that you specified.
    #   Step 1: the scaler is fitted on the TRAINING data
    #   Step 2: the scaler transforms TRAINING data
    #   Step 3: the models are fitted/trained using the transformed TRAINING data
    #   Step 4: the scaler is used to transform the TEST data
    #   Step 5: the trained models predict using the transformed TEST data
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # y_predict = cross_val_predict(pipeline, c_searchlight_data, y_train, 
    #                                 groups = np.repeat([1,2,3,4,5], per_group_num), 
    #                                 cv = LeaveOneGroupOut())  #method ='predict_proba'
    # results['score'] = y_predict.mean() #.scores_
    run_groups = np.repeat([1,2,3,4,5], per_group_num) # per_group_num will be a single integer when using all trials, 5 element vector when using a subset of trials
    full_decision_scores = np.zeros(len(y_train))
    # -- loop through runs for decoding training and testing
    for c_test_run in range(1,6):
        training_inds = run_groups!=c_test_run # will grab other 4 runs for training
        test_inds = run_groups==c_test_run
        
        pipeline.fit(c_searchlight_data[training_inds,:], y_train[training_inds])
        decision_scores = pipeline.decision_function(c_searchlight_data[test_inds,:]) # get ...
        #print(decision_scores)
        
        full_decision_scores[test_inds] = decision_scores # add to full vector
        
    results['decision_scores'] = full_decision_scores # won't be a single value... will be an array of size 40 if using all trials per run
    
    # ---- get final score
    predictions = np.sign(full_decision_scores)
    pred_1 = predictions.astype(int)==1
    y_1 = y_train.astype(int)==1
    classifications = np.where(y_1==pred_1,1,0) # 1 for correct, 0 for incorrect classification
    results['score'] = classifications.mean()
    # ---- calculate bimodality coeff
    n = len(full_decision_scores)
    skewness = skew(full_decision_scores)
    kurt = kurtosis(full_decision_scores)
    bimodality_coefficient = (skewness**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    #print(bimodality_coefficient)
    results['bimodality_coefficient'] = bimodality_coefficient # a value > 5/9 suggests bi-modal distribution
    results['bimodality_coefficient_relative_to_fiveninths']  = bimodality_coefficient - (5/9) # I just want a version that gives me a simple (+) for probably bi-modal and (-) for probably not bi-modal
    # ---- run BIC tests
    full_decision_scores = sm.add_constant(full_decision_scores)
    # -- logistic regression for bic score
    model = sm.Logit(y_train, full_decision_scores).fit(disp=0) # want the binary belief for logit
    results['Logit_bic'] = model.bic
    # -- linear regression for bic score
    model2 = sm.OLS(y_actual, full_decision_scores).fit() # want the probabilistitc belief for regresison
    results['beta'] = model2.params[1] # constant would be index 0, y_predict beta would be index 1
    results['OLS_bic'] = model2.bic

    results['sphere_idx'] = sphere_idx
    
    return results


def searchlight_prediction_job_PROBABILISTIC(sphere_idx, A, X, y_train):
    """
    For a given sphere (defined by voxel indices), run searchlight classification.
    Uses a pipeline (StandardScaler + LogisticRegression) with a fixed cross-validation
    scheme (assumes 200 trials split into 5 groups).

    Parameters:
      sphere_idx: Index of the sphere.
      A: list of spheres and voxels in each sphere.
      X: fMRI data matrix (n_samples x n_voxels).
      Y: continuous values for classification.

    Returns:
      A dictionary containing decoding score and the sphere index.
    """
    results = {}
    results['sphere_idx'] = sphere_idx
    
    # get sphere_voxel_inds
    sphere_voxel_inds = A[sphere_idx].indices
    
    # Extract the searchlight data (all beta estimates for the voxels in this sphere)
    searchlight_data = X[:, sphere_voxel_inds]
    
    pipeline = make_pipeline(StandardScaler(), LinearRegression()) # Standardize features by removing the mean and scaling to unit variance
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ~~ StandardScaler ~~ pipeline steps according to stackoveflow:
    #   Step 0: The data are split into TRAINING data and TEST data according to the cv parameter that you specified.
    #   Step 1: the scaler is fitted on the TRAINING data
    #   Step 2: the scaler transforms TRAINING data
    #   Step 3: the models are fitted/trained using the transformed TRAINING data
    #   Step 4: the scaler is used to transform the TEST data
    #   Step 5: the trained models predict using the transformed TEST data
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # y_train will have been logit transformed outside this function
    y_predict = cross_val_predict(pipeline, searchlight_data, y_train, 
                                  groups = np.repeat([1,2,3,4,5],40), cv = LeaveOneGroupOut())  #method ='predict_proba'
    
    #np.exp(y_predict)/(1+np.exp(y_predict)) # inverse logit transform
    results['r2'] = sklearn.metrics.r2_score(y_train, y_predict)  # this is r2 score, not variacne explained. See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    results['r'] = np.corrcoef(y_train, y_predict)[0,1] # this is correlation coefficient
    results['predicted_y'] = y_predict # this is the model probablistic prediction, trial by trial
    
    y_predict = sm.add_constant(y_predict) 
    # -- logistic regression for bic score
    if any(y_train<0):
        y_actual = np.where(y_train>0,1,0)
    else:
        y_actual = np.where(y_train>y_train.mean(),1,0)
    try:
        model = sm.Logit(y_actual, y_predict).fit(disp=0) # or just y_predict
        results['Logit_bic'] = model.bic
    except:
        results['Logit_bic'] = np.nan
    
    
    # -- linear regression for bic score
    if any(y_train<0):
        model = sm.OLS(np.exp(y_train)/(1+np.exp(y_train)), y_predict).fit(disp=0) # or just y_train, y_predict
    else:
        model = sm.OLS(y_train, y_predict).fit(disp=0) # or just y_train, y_predict
    results['beta'] = model.params[1] # constant would be index 0, y_predict beta would be index 1
    results['OLS_bic'] = model.bic # OLS just has bic
    
    return results


def searchlight_prediction_job(sphere_idx, A, X, Y, jointP):
    """
    For a given sphere (defined by voxel indices), run searchlight classification.
    Uses a pipeline (StandardScaler + LogisticRegression) with a fixed cross-validation
    scheme (assumes 200 trials split into 5 groups).

    Parameters:
      sphere_voxel_inds: Indices of voxels in the sphere.
      sphere_idx: Index of the sphere.
      X: fMRI data matrix (n_samples x n_voxels).
      Y: Hard labels for classification.
      jointP: The original joint probability distribution array of shape (n_samples, 4)
              (used as the true distribution in KL divergence calculation).

    Returns:
      A dictionary containing accuracy, log loss, KL divergence, predicted probabilities,
      predicted classes, and the sphere index.
    """
    # get sphere_voxel_inds
    sphere_voxel_inds = A[sphere_idx].indices
    
    # Extract the searchlight data (all beta estimates for the voxels in this sphere)
    searchlight_data = X[:, sphere_voxel_inds]
    
    # Create a classification pipeline with multinomial logistic regression
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    )
    
    # Use cross_val_predict to get predicted probabilities
    y_pred_proba = cross_val_predict(pipeline, searchlight_data, Y, groups=np.repeat([1, 2, 3, 4, 5], 40), cv=LeaveOneGroupOut(), method='predict_proba')
    
    # Derive predicted classes by taking the argmax of probabilities
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # Compute classification metrics: accuracy and log loss
    accuracy = sklearn.metrics.accuracy_score(Y, y_pred_class)
    loss = sklearn.metrics.log_loss(Y, y_pred_proba)
    
    # Calculate the KL divergence for each sample and then take the mean.
    # KL divergence: sum_i [ jointP * log(jointP / y_pred_proba) ]
    # Adding a small epsilon to y_pred_proba to avoid log(0) issues.
    epsilon = 1e-12
    kl_divergence = np.mean(np.sum(jointP * np.log(jointP / (y_pred_proba + epsilon)), axis=1))
    
    return {
        'accuracy': accuracy,
        'log_loss': loss,
        'kl_divergence': kl_divergence,
        'predicted_probabilities': y_pred_proba,
        'predicted_classes': y_pred_class,
        'sphere_idx': sphere_idx}

def run_searchlight_decoding(subject, epoch, to_predict, func_path, model_output_path, out_path, binary_mask_img, radius=8, num_cores=8):
    """
    Run subject-level searchlight decoding for a given subject.
    
    Parameters:
      subject : Subject ID as a string (e.g., "10283").
      epoch : String indicating the epoch (e.g., "cue", "probe", or "fb").
      to_predict : Variable to decode ("joint", "state", "task", "color", or "entropy").
                   If "joint", the target is assumed to be a categorical variable with 4 classes.
      func_path : Path to the fMRI images (expects folders like "sub-<subject>").
      model_output_path : Path to subject's behavioral/model data (expects file "sub-<subject>_dataframe_model-pSpP.csv").
      out_path : Directory where output (e.g., results pickle) will be saved.
      binary_mask_img: cortical mask after being converted to binary data
      radius : Radius of the searchlight sphere (default is 8).
      num_cores : Number of parallel jobs (default is 8).
      
    Returns:
      results : A list of dictionaries containing decoding results for each sphere.
    """
    # --- Load behavioral data ---
    df = pd.read_csv(os.path.join(model_output_path, f"sub-{subject}_dataframe_3dDeconvolve_pSpP.csv"))
    
    # need to recreate joint p from the state and color
    if to_predict == "jointP":
        p_s = df["pSpP_sE"].values  
        p_c = df["pSpP_cE"].values 
        jointP = np.stack([(1 - p_s) * (1 - p_c), (1 - p_s) * p_c, p_s * (1 - p_c), p_s * p_c ], axis=1)
        Y = np.argmax(jointP, axis=1)[:200]  #take the max
        Y = Y.astype(int)
        jointP = jointP[:200]
    else:
        # -- now prepare Y for prediction
        if to_predict == 'state':
            Y = df["pSpP_sE"].values # predict state
        elif to_predict == 'task':
            Y = df["pSpP_tE"].values # predict task
        elif to_predict == 'color':
            Y = df["pSpP_cE"].values # predict color
        elif to_predict == 'entropy':
            Y = df["pSpP_entropy"].values   #predict entropy
        Y = Y[:200]
        if to_predict != "entropy":
            Y = np.where(Y<.05, (Y + ((1/len(Y))/2)), Y) # avoid 0
            Y = np.where(Y>.95, (Y - ((1/len(Y))/2)), Y) # avoid 1
            # now log transform
            Y = np.log(Y/(1-Y))
    
    # --- Load fMRI image for the subject and epoch ---
    fmri_file = os.path.join(func_path, f"sub-{subject}", f"{epoch}.LSS.nii.gz")
    fmri_img = nib.load(fmri_file)
    # Assume the image contains both amplitude and derivative; select every other volume
    tr_mask = np.tile([True, False], 200)
    fmri_img = index_img(fmri_img, tr_mask)
    
    # -------------  NEW faster method to get searchlight spheres set up  ------------- #
    print("\nsetup spheres for ", epoch," epoch ... start time:-", datetime.datetime.now())
    # -- get seeds
    binary=binary_mask_img.get_fdata()
    ijk = np.column_stack(np.where(binary == 1))
    seeds = np.column_stack( coord_transform(ijk[:,0], ijk[:,1], ijk[:,2], binary_mask_img.affine) )
    # -- get X and A
    X, A = new_apply_mask_and_get_affinity(seeds, fmri_img, radius, allow_overlap=True, mask_img=binary_mask_img)
    num_spheres = A.shape[0]
    print("\nspheres for ", epoch," epoch finish setup time:-", datetime.datetime.now())
    print(f"  # spheres: {num_spheres}")

    # - - - - - - - - - - - - - - - - - #
    # - -   now do parallel jobs    - - #
    # - - - - - - - - - - - - - - - - - #
    ct = datetime.datetime.now()
    print("\nstart time:-", ct)
    # --- Run searchlight decoding in parallel using Joblib ---
    if to_predict == "jointP":
        results = Parallel(n_jobs=num_cores, verbose=2)( delayed(searchlight_prediction_job)(i, A, X, Y, jointP) for i in range(num_spheres) )
    else:
        results = Parallel(n_jobs=num_cores, verbose=2)( delayed(searchlight_prediction_job_PROBABILISTIC)(i, A, X, Y) for i in range(num_spheres) )
    ct = datetime.datetime.now()
    print("finish time:-", ct)
    
    return results


def create_nii(stats_mat, cortical_mask):
    cortical_masker = NiftiMasker(cortical_mask)
    cortical_masker.fit()
    stat_nii = cortical_masker.inverse_transform(stats_mat)
    return stat_nii

def create_nii_v2(ckey, results, mask):
    cur_nii_data = np.zeros(len(results))
    for i in np.arange(len(results)):
        cur_nii_data[i] = results[i][ckey]
    if ckey=="kl_divergence":
        # Transform the KL divergence into a similarity measure using the exponential transformation.
        # This yields a value in (0, 1] where 1 indicates perfect correspondence.
        cur_nii = masking.unmask(np.exp(-cur_nii_data), mask)    
    else:
        cur_nii = masking.unmask(cur_nii_data, mask)
    
    return cur_nii
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# -- load mask
mask = nib.load(os.path.join(mask_dir, "CorticalMask_RSA_task-Quantum.nii.gz"))
mask_data = mask.get_fdata()
binary_mask = (mask_data > 0).astype(np.int8)
binary_mask_img = nilearn.image.new_img_like(mask, binary_mask) # I don't think I need to do this, but doing it just in case the int8 call reduces file size to increase speed

# -- get subject lsit
unusable_subs = ['10118', '10218', '10275', '10282', '10296', '10318', '10319', '10321', '10322', '10351', '10358'] # list of subjects to exclude
subjects = sorted(os.listdir(bids_path)) # this folder only contains usable subjects
print(subjects)
subject_list = []
for ii, subj in enumerate(subjects):
    if os.path.isdir(os.path.join(func_path,subj)):
        cur_subj = subj.split('-')[1]
        if cur_subj not in unusable_subs:
            subject_list.append(cur_subj)
print("a total of ", len(subject_list), " subjects will be included")

if args.ttest:
    # -- get subject lsit
    unusable_subs = ['10118', '10218', '10275', '10282', '10296', '10318', '10319', '10321', '10322', '10351', '10358'] # list of subjects to exclude
    subjects = sorted(os.listdir(bids_path)) # this folder only contains usable subjects
    print(subjects)
    subject_list = []
    for ii, subj in enumerate(subjects):
        if os.path.isdir(os.path.join(func_path,subj)):
            cur_subj = subj.split('-')[1]
            if cur_subj not in unusable_subs:
                subject_list.append(cur_subj)
    print("a total of ", len(subject_list), " subjects will be included")
    # -- load random func run to get dimensions
    img = nib.load( out_path + ("sub-"+subject_list[0]) + "/task_cue_searchlight_decoding-probabilistic__r.nii.gz" )
    dims=img.get_fdata().shape
    # -- apply mask
    masked_data=nilearn.masking.apply_mask(img, binary_mask_img) 
    masked_data_dims = masked_data.shape
    print("masked data dimensions: ", masked_data_dims[0])
        
    if not(args.binary):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - -     Probabalistic Decoding ... t-test   - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # , "statePE", "taskPE", "stateD"
        # - - JointP decoding different from state, task, color, and entropy
        predicted = "jointP"
        for epoch in ["cue", "probe"]:
            r_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects (to_predict + "_" + epoch + "_searchlight_decoding-probabilistic__"+ckey+
            print("loading decoding maps")
            for subj_idx, cur_subj in enumerate(subject_list):
                print("loading files for subject ", cur_subj)
                # -- load nii files
                r_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted + "_" + epoch + "_searchlight_decoding__jointP__accuracy.nii.gz")) ) # nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted+"_"+epoch+"_searchlight_prediction_r.nii.gz")) ) 
                # -- apply mask
                r_masked = nilearn.masking.apply_mask(r_nii, binary_mask_img)
                # -- save masked data in larger matrix
                r_maps[:,subj_idx] = r_masked
            
            r_group_stats = np.zeros((r_maps.shape[0],2)) # dim is tval and pval in that order
            print("looping through voxels and running t-tests")
            # ---- loop through voxels to run t-tests
            for v_idx in range( r_maps.shape[0] ):
                r_group_stats[v_idx,0], r_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(r_maps[v_idx,:]), popmean=0.25, nan_policy='omit')
            print("saving out group averaged files for ", predicted," and ", epoch," now")
            # ---- convert decoding ttests to nii file
            state_group_tnii = create_nii(r_group_stats[:,0], mask)
            state_group_tnii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__r__tval.nii")))
            # ---- get average coeff (avg. beta) from within subject searchlight method
            state_avg_coeff=np.mean(r_maps, axis=1) # 2nd dim is subjects
            state_group_avg_nii = create_nii(state_avg_coeff, mask)
            state_group_avg_nii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__r__avg_coeff.nii")))
        
        for predicted in [ "entropy", "state", "task", "color"]:
            for epoch in ["fb","cue", "probe"]:
                r_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
                # ---- loop through subjects to load decoded maps
                print("loading decoding maps")
                for subj_idx, cur_subj in enumerate(subject_list):
                    print("loading files for subject ", cur_subj)
                    # -- load nii files
                    r_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted + "_" + epoch + "_searchlight_decoding-probabilistic__r.nii.gz")) )
                    # -- apply mask
                    r_masked = nilearn.masking.apply_mask(r_nii, binary_mask_img)
                    # -- save masked data in larger matrix
                    r_maps[:,subj_idx] = r_masked
                
                r_group_stats = np.zeros((r_maps.shape[0],2)) # dim is tval and pval in that order
                print("looping through voxels and running t-tests")
                # ---- loop through voxels to run t-tests
                for v_idx in range( r_maps.shape[0] ):
                    r_group_stats[v_idx,0], r_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(r_maps[v_idx,:]), popmean=0.0, nan_policy='omit')
                print("saving out group averaged files for ", predicted," and ", epoch," now")
                # ---- convert decoding ttests to nii file
                state_group_tnii = create_nii(r_group_stats[:,0], mask)
                state_group_tnii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__r__tval.nii")))
                # ---- get average coeff (avg. beta) from within subject searchlight method
                state_avg_coeff=np.mean(r_maps, axis=1) # 2nd dim is subjects
                state_group_avg_nii = create_nii(state_avg_coeff, mask)
                state_group_avg_nii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__r__avg_coeff.nii")))
    else:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - -     Binary Decoding ... t-test   - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        for predicted in ["state", "color"]:
            for epoch in ["cue", "probe"]:
                r_maps = np.zeros( (masked_data_dims[0],len(subject_list)) ) # voxels by subjects
                # ---- loop through subjects to load decoded maps
                print("loading decoding maps")
                for subj_idx, cur_subj in enumerate(subject_list):
                    print("loading files for subject ", cur_subj)
                    # -- load nii files
                    r_nii = nib.load( os.path.join(out_path, ("sub-"+cur_subj), (predicted + "_" + epoch + "_searchlight_decoding-binary__score.nii.gz")) )
                    # -- apply mask
                    r_masked = nilearn.masking.apply_mask(r_nii, binary_mask_img)
                    # -- save masked data in larger matrix
                    r_maps[:,subj_idx] = r_masked
                
                r_group_stats = np.zeros((r_maps.shape[0],2)) # dim is tval and pval in that order
                print("looping through voxels and running t-tests")
                # ---- loop through voxels to run t-tests
                for v_idx in range( r_maps.shape[0] ):
                    r_group_stats[v_idx,0], r_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(r_maps[v_idx,:]), popmean=0.0, nan_policy='omit')
                print("saving out group averaged files for ", predicted," and ", epoch," now")
                # ---- convert decoding ttests to nii file
                state_group_tnii = create_nii(r_group_stats[:,0], mask)
                state_group_tnii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__score__tval.nii")))
                # ---- get average coeff (avg. beta) from within subject searchlight method
                state_avg_coeff=np.mean(r_maps, axis=1) # 2nd dim is subjects
                state_group_avg_nii = create_nii(state_avg_coeff, mask)
                state_group_avg_nii.to_filename(os.path.join(out_path, "GroupStats", ("GroupAnalysis_"+str(len(subject_list))+"subjs__"+predicted+"_"+epoch+"__score__avg_coeff.nii")))
        
        
else:
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    num_cores=15
    radius=8
    # for subj_idx, subject in enumerate(subject_list):
    #     if int(subject)==10280:
    #         continue
    # -- make subject folder for decoding results if necessary
    if not os.path.isdir(os.path.join(out_path, "sub-"+str(subject))):
        os.mkdir(os.path.join(out_path, "sub-"+str(subject)))
    
    if not(args.binary):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # - - - - - -        Run Probabalistic Decoding       - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # - - JointP decoding different from state, task, color, and entropy
        #for subj_idx, subject in enumerate(subject_list):
        for to_predict in ["state", "task", "color", "entropy"]: # "jointP", "state", "task", "color", "entropy"
            for epoch in ["fb"]:
                # - - decoding for color, state, task, and entropy
                print("running probabalistic decoding on subject ", subject, " with ", to_predict)
                subject = int(subject) #10008
                
                # - - - - - - - - - - - - - - 
                # - - Run searchlight decoding for the subject
                if not(os.path.exists(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-probabilistic_results.p")))):
                    results = run_searchlight_decoding(subject, epoch, to_predict, func_path, model_output_path, out_path, binary_mask_img, radius=radius, num_cores=num_cores)
                    print("Searchlight decoding completed. Processed", len(results), "spheres.")

                    # - - - - - - - - - - - - - - 
                    # - - - -   save out results
                    print("saving out results...")
                    pickle.dump(results, open(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-probabilistic_results.p")), "wb"))
                
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    # - - - - - - -      Create .nii files      - - - - - - - 
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    num_of_spheres = len(results)
                    if to_predict != "jointP":
                        for ckey in ['OLS_bic','Logit_bic','r','beta']:
                            cur_nii = create_nii_v2(ckey, results, mask)
                            cur_nii.to_filename(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-probabilistic__"+ckey+".nii.gz")))
                    else:
                        for ckey in ['accuracy','kl_divergence']:
                            cur_nii = create_nii_v2(ckey, results, mask)
                            cur_nii.to_filename(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding__jointP__"+ckey+".nii.gz")))
    else:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # - - - - - -        Run Binary Decoding       - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        use_all_trials = False

        for to_predict in ["state", "color"]:
            for epoch in ["cue", "probe"]:
                # - - - - - - - - - - - - - - 
                # - - Run searchlight decoding for the subject
                # # - - decoding for color, state, task
                print("running binary decoding with ", to_predict)
                subject = int(subject) #10008
                
                # --- Load behavioral data ---
                df = pd.read_csv(os.path.join(model_output_path, f"sub-{subject}_dataframe_3dDeconvolve_pSpP.csv"))
                
                # -- now prepare Y for prediction
                if to_predict == 'state':
                    Y_original = df["pSpP_sE"].values # predict state
                elif to_predict == 'task':
                    Y_original = df["pSpP_tE"].values # predict task
                elif to_predict == 'color':
                    Y_original = df["pSpP_cE"].values # predict color
                Y_original = Y_original[:200]
                Y = Y_original.copy().round()*1.0
                # --- Load fMRI image for the subject and epoch ---
                fmri_file = os.path.join(func_path, f"sub-{subject}", f"{epoch}.LSS.nii.gz")
                fmri_img = nib.load(fmri_file)
                # Assume the image contains both amplitude and derivative; select every other volume
                tr_mask = np.tile([True, False], 200)
                fmri_img = index_img(fmri_img, tr_mask)
                
                # -------------  NEW faster method to get searchlight spheres set up  ------------- #
                print("\nsetup spheres for ", epoch," epoch ... start time:-", datetime.datetime.now())
                # -- get seeds
                ijk = np.column_stack(np.where(binary_mask == 1))
                seeds = np.column_stack( coord_transform(ijk[:,0], ijk[:,1], ijk[:,2], binary_mask_img.affine) )
                # -- get X and A
                X, A = new_apply_mask_and_get_affinity(seeds, fmri_img, radius, allow_overlap=True, mask_img=binary_mask_img)
                num_spheres = A.shape[0]
                print("\nspheres for ", epoch," epoch finish setup time:-", datetime.datetime.now())
                print(f"  # spheres: {num_spheres}")

                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # - - - - - - -      Additional Reviwer Requested Decoding Version      - - - - - - - 
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # Also REVIEWER asked us to only use high certainty trials !!! Have to calculate subset of trials for each belief
                certain_trials = np.abs(Y_original-0.5) > .4 # subtract center point and take abs value so that we can easily exclude values between .1 and .9
                #per_group_num = np.floor(np.sum(1.0*certain_trials)/5)
                hc_df = pd.DataFrame({'sub':[subject]*5, 'epoch':[epoch]*5, 'predicted':[to_predict]*5, 'run':[1,2,3,4,5], 
                            'usable_trials':[np.sum(1.0*certain_trials[:40]), 
                                            np.sum(1.0*certain_trials[40:80]), 
                                            np.sum(1.0*certain_trials[80:120]), 
                                            np.sum(1.0*certain_trials[120:160]), 
                                            np.sum(1.0*certain_trials[160:])]})
                if os.path.exists(os.path.join(out_path, "binary_decoding_high_certainty_trials_only.csv")):
                    df = pd.read_csv(os.path.join(out_path, "binary_decoding_high_certainty_trials_only.csv"))
                    df_combo = pd.concat([df,hc_df], ignore_index=True)
                    df_combo.to_csv(os.path.join(out_path, "binary_decoding_high_certainty_trials_only.csv"), index=False)
                else:
                    hc_df.to_csv(os.path.join(out_path, "binary_decoding_high_certainty_trials_only.csv"), index=False)

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                
                # - - - - - - - - - - - - - - - - - #
                # - -   now do parallel jobs    - - #
                # - - - - - - - - - - - - - - - - - #
                ct = datetime.datetime.now()
                print("start time:-", ct)
                # --- Run searchlight decoding in parallel using Joblib ---
                # reminder of function arguments              searchlight_prediction_job_BINARY(sphere_idx, A, X, Y, certain_trials, belief, use_all_trials)
                results = Parallel(n_jobs=num_cores)( delayed(searchlight_prediction_job_BINARY)(i, A, X, Y, certain_trials, Y_original, use_all_trials) for i in range(num_spheres) )
                ct = datetime.datetime.now()
                print("finish time:-", ct)
                print("Searchlight decoding completed. Processed", len(results), "spheres.")

                # - - - - - - - - - - - - - - 
                # - - - -   save out results
                print("saving out results...")
                if use_all_trials:
                    pickle.dump(results, open(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-binary_results.p")), "wb"))
                else:
                    pickle.dump(results, open(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-binary-HC_results.p")), "wb"))
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                # - - - - - - -      Create .nii files      - - - - - - - 
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                num_of_spheres = len(results)
                for ckey in ['OLS_bic','Logit_bic','score']:
                    cur_nii = create_nii_v2(ckey, results, mask)
                    if use_all_trials:
                        cur_nii.to_filename(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-binary__"+ckey+".nii.gz")))
                    else:
                        cur_nii.to_filename(os.path.join(out_path, "sub-"+str(subject), (to_predict + "_" + epoch + "_searchlight_decoding-binary-HC__"+ckey+".nii.gz")))