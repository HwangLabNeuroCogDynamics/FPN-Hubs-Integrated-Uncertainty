import os
import sys
import argparse
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.input_data import NiftiLabelsMasker
from nilearn.maskers import NiftiMasker
from nilearn.image import index_img
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy.stats import ttest_1samp
import networkx as nx
import community as community_louvain
import warnings
# Set environment variables to limit thread usage.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode subject",
        usage=" [dataset_dir] [OPTIONS] ... ",
    )
    parser.add_argument("dataset_dir", help="dataset directory path e.g., /Shared/lss_kahwang_hpc/data/FPNHIU/ ")
    
    return parser

parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
dataset_dir = args.dataset_dir

# ---- set up other paths
bids_path = os.path.join(dataset_dir, "BIDS")
func_path = os.path.join(dataset_dir, "3dDeconvolve")


###############################################################
### this is a set of ROI operartion functions
###############################################################
def process_t_stat_file(t_stat_path, output_basename, t_threshold=None, top_n=None):
    """
    Process a 3D NIfTI file containing t-statistics by computing the average t value for each ROI
    in the Schaefer 2018 (400 ROI) atlas and selecting ROIs using either a threshold or by selecting
    the top N ROIs.
    
    The function saves a combined NIfTI file in which only the selected ROIs (with new, consecutive labels)
    are retained and returns the list of selected ROI indices.
    
    Parameters
    ----------
    t_stat_path : str
        File path to the 3D NIfTI file containing t-statistics.
    output_basename : str
        Base name for the output file. A combined group file will be saved as 
        `output_basename.nii.gz`.
    t_threshold : float, optional
        The t-value threshold for selecting ROIs. This is used if top_n is None.
    top_n : int, optional
        If provided, the function selects the top_n ROIs by average t value, ignoring the threshold.
        
    Returns
    -------
    selected_roi_indices : list of int
        List of original ROI indices that are selected.
    """

    t_img = nib.load(t_stat_path)
    t_data = t_img.get_fdata()
    print("t-statistics image shape:", t_data.shape)
    schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    atlas_img = nib.load(schaefer.maps)
    
    #Extract Average t Values for Each ROI
    # Add a dummy time axis to make the image 4D.
    t_img_4d = nib.Nifti1Image(t_data[..., np.newaxis], affine=t_img.affine)
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    # Output is (1, 400)
    roi_avg_values = masker.fit_transform(t_img_4d).flatten()
    
    # Define ROI indices (assumed 1-indexed) and pair with average t values.
    roi_indices = np.arange(1, 401)
    roi_data = list(zip(roi_indices, roi_avg_values))
    
    # Rank-order the ROIs by average t value (highest first).
    roi_data_sorted = sorted(roi_data, key=lambda x: x[1], reverse=True)
    if top_n is not None:
        selected_roi_data = roi_data_sorted[:top_n]
    elif t_threshold is not None:
        selected_roi_data = [entry for entry in roi_data_sorted if entry[1] > t_threshold]
    else:
        raise ValueError("Either t_threshold or top_n must be provided.")
    
    selected_roi_indices = [entry[0] for entry in selected_roi_data]
    
    print("\nSelected ROIs:")
    for roi_index, avg_t in selected_roi_data:
        print(f"ROI {roi_index}: Avg t = {avg_t:.3f}")
    
    # Create and Save a Combined Group NIfTI Image with the Selected ROIs
    atlas_data = atlas_img.get_fdata()
    combined_atlas_data = np.zeros_like(atlas_data)
    new_label = 1
    for roi_index, avg_t in selected_roi_data:
        mask = (atlas_data == roi_index)
        if np.any(mask):
            combined_atlas_data[mask] = new_label
            new_label += 1
            
    group_output_path = output_basename + '.nii.gz'
    combined_atlas_img = nib.Nifti1Image(combined_atlas_data, affine=atlas_img.affine)
    nib.save(combined_atlas_img, group_output_path)
    print("\nCombined group NIfTI file saved as:", group_output_path)
    
    return selected_roi_indices

def get_avg_t_values(t_stat_path):
    """
    Process a 3D or 5D NIfTI file containing t-statistics by computing the average t value for each ROI
    in the Schaefer 2018 (400 ROI) atlas. Returns a DataFrame with ROI indices, ROI labels, and 
    average t values.
    
    Parameters
    ----------
    t_stat_path : str
        File path to the 3D NIfTI file containing t-statistics.
        
    Returns
    -------
    roi_df : pd.DataFrame
        DataFrame with columns: 'ROI_index', 'ROI_label', and 'avg_t'.
    """
    # Step 1: Load the 3D t-statistics image.
    t_img = nib.load(t_stat_path)
    t_data = t_img.get_fdata()
    print("t-statistics image shape:", t_data.shape)
    
    # Step 2: Fetch the Schaefer 2018 Atlas (400 ROIs)
    schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    atlas_img = nib.load(schaefer.maps)
    
    # Step 3: Extract Average t Values for Each ROI
    if len(t_data.shape) == 3:
        # Add a dummy time axis to make the image 4D.
        t_img_4d = nib.Nifti1Image(t_data[..., np.newaxis], affine=t_img.affine)
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        # This returns an array of shape (1, 400)
        roi_avg_values = masker.fit_transform(t_img_4d).flatten()
    else:
        # take only tvalues from last dimension to make 4D
        t_img_4d = nib.Nifti1Image(t_data[:,:,:,:,1], affine=t_img.affine)
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        # This returns an array of shape (1, 400)
        roi_avg_values = masker.fit_transform(t_img_4d).flatten()
        
    # Step 4: Create DataFrame with ROI indices, labels, and average t values.
    roi_indices = np.arange(1, 401)  # assuming 1-indexed ROI labels
    roi_labels = schaefer.labels       # list of 400 ROI labels from the atlas
    roi_df = pd.DataFrame({
        'ROI_index': roi_indices,
        'ROI_label': roi_labels,
        'avg_t': roi_avg_values
    })
    
    return roi_df

# Organize Subject LSS Betas Using the Schaefer Atlas
def organize_schaefer_LSS_betas(subjects, df, data_path, file_pattern, output_data_path, output_sdf_path, include_subcortical=False):
    """
    For each subject, load the LSS beta image (cue/probe) and extract trial-by-trial
    beta estimates from all 400 Schaefer ROIs using nilearn’s NiftiLabelsMasker.
    
    The Schaefer atlas (400 ROIs) is fetched automatically. The ROI betas are appended
    to the subject's trial-level dataframe (df). ROI columns are named with both the index
    and the label (e.g., "1:SomatoMotor_A").
    
    Parameters
    ----------
    subjects : array-like
        List/array of subject IDs.
    df : pd.DataFrame
        DataFrame containing behavioral/trial information. Must have a 'Subject' column.
        Also must contain a column named 'pSpP_entropy'.
    data_path : str
        Base directory containing subject folders (e.g., "/Shared/lss_kahwang_hpc/data/TRIIMS/3dDeconvolve/").
    file_pattern : str
        Filename pattern for the LSS files (e.g., "cue.LSS.nii.gz" or "probe.LSS.nii.gz").
    output_data_path : str
        File path (including filename and extension, e.g., .npy) to save the concatenated beta array.
    output_sdf_path : str
        File path to save the concatenated DataFrame (CSV format).
        
    Returns
    -------
    data_vec : np.ndarray
        Concatenated NumPy array of ROI betas with shape (total_trials, 400).
    sdf_final : pd.DataFrame
        Concatenated DataFrame with ROI beta columns appended.
    """
    import os, glob
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import index_img

    # Fetch the Schaefer 2018 atlas (400 ROIs) from nilearn.
    schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    atlas_img = nib.load(schaefer.maps)
    roi_labels = schaefer.labels  # list of 400 ROI labels
    roi_indices = np.arange(1, 401)  # assuming atlas labels are 1-indexed
    
    # OPTIONAL: add subcortical structure ROIs
    if include_subcortical:
        # load local thalamus and basalganglia 
        mask_C_S = "/mnt/nfs/lss/lss_kahwang_hpc/ROIs/Schaefer400+Morel+BG_7T.nii.gz"
        C_S_atlas_img = nib.load(mask_C_S)
        C_S_roi_indices = np.arange(1, 419) # will be 18 subcortical

    data_list = []  # to hold arrays (n_trials, 400)
    sdf_list = []   # to hold dataframes for each run

    # Loop over subjects
    for sub in subjects:
        # Construct file pattern for the subject (assumed folder: sub-[sub])
        fns = np.sort(glob.glob(os.path.join(data_path, f"sub-{sub}/*{file_pattern}")))
        if len(fns) == 0:
            print(f"No files found for subject {sub} with pattern {file_pattern}.")
            continue
        
        for fpath in fns:
            # Filter the trial-level dataframe for this subject
            tdf = df[df['Subject'] == int(sub)].copy().reset_index(drop=True)
            if len(tdf) == 0:
                continue

            # Load the LSS beta image (assumed to be 4D)
            fmri_img = nib.load(fpath)
            n_vol = fmri_img.shape[3]
            if n_vol % 2 != 0:
                raise ValueError(f"Expected even number of volumes in {fpath}, got {n_vol}")
            # Select amplitude volumes (assuming alternating amplitude/derivative)
            tr_mask = np.tile([True, False], n_vol // 2)
            fmri_img = index_img(fmri_img, tr_mask)  # now 4D with n_vol/2 volumes

            # Use NiftiLabelsMasker with the Schaefer atlas to extract ROI betas.
            masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
            # The output is (n_trials, 400)
            rois_betas = masker.fit_transform(fmri_img)

            # Create ROI column names as "index:Label"
            roi_col_names = [f"{roi_indices[i]}:{roi_labels[i]}" for i in range(400)]
            
            # OPTIONAL: add subcortical structure ROIs
            if include_subcortical:
                masker2 = NiftiLabelsMasker(labels_img=C_S_atlas_img, standardize=False)
                rois_betas2 = masker2.fit_transform(fmri_img) # (n_trials, 418)
                for s_i in range(401,419):
                    roi_col_names.append(f"{s_i}:subcortial_{+s_i}")
            
            # Build a DataFrame for the ROI betas and ensure it matches tdf's index.
            roi_df = pd.DataFrame(rois_betas, columns=roi_col_names, index=tdf.index)
            # Concatenate all ROI beta columns to the trial-level dataframe.
            tdf = pd.concat([tdf, roi_df], axis=1)
            
            data_list.append(rois_betas)
            sdf_list.append(tdf)

    if len(data_list) > 0:
        data_vec = np.concatenate(data_list, axis=0)
        sdf_final = pd.concat(sdf_list, axis=0)
    else:
        data_vec = np.array([])
        sdf_final = pd.DataFrame()

    np.save(output_data_path, data_vec)
    sdf_final.to_csv(output_sdf_path, index=False)

    print(f"Saved beta array to {output_data_path} and dataframe to {output_sdf_path}")
    return data_vec, sdf_final

###############################################################
# Functions for FC analyses. Within subject and mixed effects.
###############################################################
def run_model_for_roi_within_subject(roi1_name, roi2_name, subject_df, col_of_interest, winsor_limits=(0.005, 0.005)):
    """
    Run a simple OLS regression within a single subject for a given ROI pair.
    The model is:
        ROI1 ~ 1 + ROI2 * pSpP_entropy
    The ROI column names may contain special characters (e.g., colon) so they are wrapped using Q().
    
    Parameters
    ----------
    roi1_name : str
        Dependent variable column name (e.g., "1:SomatoMotor_A").
    roi2_name : str
        Predictor variable column name.
    subject_df : pd.DataFrame
        DataFrame for a single subject containing the ROI beta columns.
    col_of_interest : str
        Name of column in the subject_df that we want to test interactions with (e.g., 'pSpP_entropy')
    winsor_limits : tuple of float
        Limits for winsorization.
        
    Returns
    -------
    result : RegressionResults
        The fitted OLS model results.
    """
    
    needed_cols = [roi1_name, roi2_name, col_of_interest]
    df_model = subject_df[needed_cols].dropna().copy()

    for col in [roi1_name, roi2_name, col_of_interest]:
        df_model[col] = winsorize(df_model[col], limits=winsor_limits).data

    for col in [roi1_name, roi2_name, col_of_interest]:
        df_model[col] = df_model[col] - df_model[col].mean()

    # Use Q() to handle special characters in variable names.
    formula = f'Q("{roi1_name}") ~ 1 + Q("{roi2_name}") * Q("{col_of_interest}")'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = smf.ols(formula, data=df_model)
        result = model.fit()
    return result

def fc_analysis_within_subjects(sdf, roi_names_list, moderator='pSpP_entropy', winsor_limits=(0.005, 0.005), n_jobs=-1):
    """
    For each subject in the dataframe, perform FC analysis by running a separate OLS regression
    for every ROI pair (excluding self-coupling) in parallel across subjects.
    
    For each regression (model: ROI1 ~ 1 + ROI2 * pSpP_entropy), extract the coefficient, t-value,
    and p-value for:
      - the main effect of ROI2 ("main"), and
      - the interaction between ROI2 and the moderator ("interaction").
    
    The output is organized into one dataframe with columns:
        Subject, ROI1, ROI2, ROI1_index, ROI1_label, ROI2_index, ROI2_label,
        Regressor, Beta, t_value, p_value, n_obs.
    
    Parameters
    ----------
    sdf : pd.DataFrame
        The concatenated dataframe with trial-level data and ROI beta columns.
    roi_names_list : list of str
        List of ROI column names (e.g., "1:Label", "2:Label", ..., "400:Label").
    moderator : str
        Name of the moderator variable (default "pSpP_entropy").
    winsor_limits : tuple of float
        Winsorization limits.
    n_jobs : int
        Number of parallel jobs to run (default -1 uses all available cores).
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per regressor per ROI pair per subject.
    """
    
    subjects = sdf['Subject'].unique()
    
    def process_subject_fc(sub):
        sub_results = []
        sub_df = sdf[sdf['Subject'] == sub]
        for roi1 in roi_names_list:
            for roi2 in roi_names_list:
                if roi1 == roi2:
                    continue
                try:
                    res = run_model_for_roi_within_subject(roi1, roi2, sub_df, col_of_interest=moderator, winsor_limits=winsor_limits)
                except Exception as e:
                    print(f"Error for subject {sub}, ROI pair {roi1} and {roi2}: {e}")
                    continue

                # Define the parameter names as they appear in the fitted model.
                param_roi2 = f'Q("{roi2}")'
                param_interaction = f'Q("{roi2}"):Q("{moderator}")'

                for regressor, param_name in [('main', param_roi2), ('interaction', param_interaction)]:
                    beta = res.params.get(param_name, np.nan)
                    t_val = res.tvalues.get(param_name, np.nan)
                    p_val = res.pvalues.get(param_name, np.nan)
                    sub_results.append({
                        'Subject': sub,
                        'ROI1': roi1,
                        'ROI2': roi2,
                        'ROI1_index': roi1.split(":")[0],
                        'ROI1_label': roi1.split(":")[1] if ":" in roi1 else roi1,
                        'ROI2_index': roi2.split(":")[0],
                        'ROI2_label': roi2.split(":")[1] if ":" in roi2 else roi2,
                        'Regressor': regressor,
                        'Beta': beta,
                        't_value': t_val,
                        'p_value': p_val,
                        'n_obs': res.nobs
                    })
        return sub_results

    # Parallelize the processing across subjects.
    results_list = Parallel(n_jobs=n_jobs)(delayed(process_subject_fc)(sub) for sub in subjects)
    # Flatten the list of lists.
    flat_results = [item for sublist in results_list for item in sublist]
    results_df = pd.DataFrame(flat_results)
    return results_df

def run_model_for_roi_mixed_effects(roi1_name, roi2_name, data_df, moderator='pSpP_entropy', winsor_limits=(0.005, 0.005)):
    """
    Run a mixed effects regression for a given ROI pair across all subjects.
    The model is:
        ROI1 ~ 1 + ROI2 * pSpP_entropy
    with a random intercept for Subject.
    The ROI column names may contain special characters so they are wrapped using Q().
    
    Parameters
    ----------
    roi1_name : str
        Dependent variable column name (e.g., "1:SomatoMotor_A").
    roi2_name : str
        Predictor variable column name.
    moderator : str
        Name of the moderator variable (default "pSpP_entropy").
    data_df : pd.DataFrame
        DataFrame containing data for all subjects with ROI beta columns, 'pSpP_entropy',
        and 'Subject'.
    winsor_limits : tuple of float
        Winsorization limits.
        
    Returns
    -------
    result : RegressionResults
        The fitted mixed effects model results.
    """
    needed_cols = [roi1_name, roi2_name, moderator, 'Subject']
    df_model = data_df[needed_cols].dropna().copy()
    
    for col in [roi1_name, roi2_name, moderator]:
        df_model[col] = winsorize(df_model[col], limits=winsor_limits).data
        df_model[col] = df_model[col] - df_model[col].mean()
    
    # Use Q() to properly handle column names with special characters.
    formula = f'Q("{roi1_name}") ~ 1 + Q("{roi2_name}") * Q("{moderator}")'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = smf.mixedlm(formula, data=df_model, groups=df_model["Subject"], re_formula="1")
        result = model.fit()
    return result

def fc_analysis_mixed_effects(sdf, roi_names_list, moderator='pSpP_entropy', winsor_limits=(0.005,0.005), n_jobs=-1):
    """
    For each ROI pair (excluding self-coupling), run a mixed effects regression across all subjects 
    using a random intercept for subject. The model is:
        ROI1 ~ 1 + ROI2 * pSpP_entropy
    The function extracts the coefficient, t-value, and p-value for:
      - the main effect of ROI2 ("main"), and
      - the interaction between ROI2 and the moderator ("interaction").
    
    The output is organized into a DataFrame with columns:
        Subject, ROI1, ROI2, ROI1_index, ROI1_label, ROI2_index, ROI2_label,
        Regressor, Beta, t_value, p_value, n_obs.
    Here, "Subject" is set to "Mixed" to indicate group-level analysis.
    
    Parameters
    ----------
    sdf : pd.DataFrame
        DataFrame containing trial-level data with ROI beta columns, 'pSpP_entropy', and 'Subject'.
    roi_names_list : list of str
        List of ROI column names (e.g., "1:Label", "2:Label", ..., "400:Label").
    moderator : str
        Name of the moderator variable (default "pSpP_entropy").
    winsor_limits : tuple of float
        Winsorization limits.
    n_jobs : int
        Number of parallel jobs to run (default -1 uses all available cores).
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per regressor per ROI pair for the mixed effects model.
    """
    def process_roi_pair(roi1, roi2):
        if roi1 == roi2:
            return []
        try:
            res = run_model_for_roi_mixed_effects(roi1, roi2, sdf, moderator=moderator, winsor_limits=winsor_limits)
        except Exception as e:
            print(f"Error for ROI pair {roi1} and {roi2}: {e}")
            return []
        
        # Define the parameter names as they appear in the mixed effects output.
        param_roi2 = f'Q("{roi2}")'
        param_interaction = f'Q("{roi2}"):Q("{moderator}")'
        
        results = []
        for regressor, param_name in [('main', param_roi2), ('interaction', param_interaction)]:
            beta = res.params.get(param_name, np.nan)
            t_val = res.tvalues.get(param_name, np.nan)
            p_val = res.pvalues.get(param_name, np.nan)
            results.append({
                'Subject': 'Mixed',
                'ROI1': roi1,
                'ROI2': roi2,
                'ROI1_index': roi1.split(":")[0],
                'ROI1_label': roi1.split(":")[1] if ":" in roi1 else roi1,
                'ROI2_index': roi2.split(":")[0],
                'ROI2_label': roi2.split(":")[1] if ":" in roi2 else roi2,
                'Regressor': regressor,
                'Beta': beta,
                't_value': t_val,
                'p_value': p_val,
                'n_obs': res.nobs
            })
        return results

    # Create all ROI pair combinations (exclude self-coupling).
    roi_pairs = [(roi1, roi2) for roi1 in roi_names_list for roi2 in roi_names_list if roi1 != roi2]
    
    results_list = Parallel(n_jobs=n_jobs)(delayed(process_roi_pair)(roi1, roi2) for roi1, roi2 in roi_pairs)
    flat_results = [item for sublist in results_list for item in sublist]
    results_df = pd.DataFrame(flat_results)
    return results_df

### now group stats function
def group_fc_statistics(results_df, roi1_indices, roi2_indices, effect='main'):
    """
    For each combination of ROI1 (from roi1_indices) and ROI2 (from roi2_indices), perform a 
    group-level one-sample t-test on the FC coefficient (for the specified effect) testing against zero.
    This version uses df.loc filtering and groupby for improved speed.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing FC results with columns including 'Subject', 'ROI1_index', 'ROI2_index',
        'Regressor', and 'Beta'. ROI1_index and ROI2_index are assumed to be integers (or convertible).
    roi1_indices : list of int
        List of ROI indices to be used as ROI1 (e.g., [1, 2, ..., 400]).
    roi2_indices : list of int
        List of ROI indices to be used as ROI2.
    effect : str
        Specifies which FC value to analyze ("main" or "interaction").

    Returns
    -------
    t_matrix : pd.DataFrame
        DataFrame (ROI1 x ROI2) of t-statistics.
    p_matrix : pd.DataFrame
        DataFrame (ROI1 x ROI2) of p-values.
    """
    # Ensure the ROI index columns are integers.
    results_df['ROI1_index'] = results_df['ROI1_index'].astype(int)
    results_df['ROI2_index'] = results_df['ROI2_index'].astype(int)
    
    # Filter for the desired effect.
    df_eff = results_df.loc[results_df['Regressor'] == effect].copy()
    
    # Group by ROI1_index and ROI2_index.
    groups = df_eff.groupby(['ROI1_index', 'ROI2_index'])
    
    # Create dictionaries to store t and p values.
    t_dict = {}
    p_dict = {}
    
    for (roi1, roi2), group_df in groups:
        # Skip self-coupling.
        if roi1 == roi2:
            continue
        # Require at least two observations.
        if group_df.shape[0] < 2:
            continue
        coeffs = group_df['Beta'].values
        t_stat, p_val = ttest_1samp(coeffs, 0.0)
        t_dict[(roi1, roi2)] = t_stat
        p_dict[(roi1, roi2)] = p_val

    # Initialize matrices.
    t_mat = np.full((len(roi1_indices), len(roi2_indices)), np.nan)
    p_mat = np.full((len(roi1_indices), len(roi2_indices)), np.nan)
    
    # Build mapping from ROI index to matrix index.
    roi1_to_idx = {roi: idx for idx, roi in enumerate(roi1_indices)}
    roi2_to_idx = {roi: idx for idx, roi in enumerate(roi2_indices)}
    
    # Populate the matrices using the dictionary values.
    for (roi1, roi2), t_val in t_dict.items():
        if roi1 in roi1_to_idx and roi2 in roi2_to_idx:
            i = roi1_to_idx[roi1]
            j = roi2_to_idx[roi2]
            t_mat[i, j] = t_val
            p_mat[i, j] = p_dict[(roi1, roi2)]
    
    t_matrix = pd.DataFrame(t_mat, index=roi1_indices, columns=roi2_indices)
    p_matrix = pd.DataFrame(p_mat, index=roi1_indices, columns=roi2_indices)
    
    return t_matrix, p_matrix

def run_fc_regression(merged_df, effect_type, moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005)):
    """
    Run a regression model predicting the FC Beta from independent variables with outlier
    removal via winsorization.
    
    For the subset of data corresponding to effect_type (either "main" or "interaction"),
    the dependent variable is Beta. The independent variables (IVs) are:
      - Entropy_ROI1 (from ROI1)
      - Color_ROI2, State_ROI2, Task_ROI2 (from ROI2)
      - Plus the interactions between Entropy_ROI1 and each ROI2 variable.
    
    The function winsorizes each IV (using the specified winsor_limits), then centers them, 
    creates interaction terms, and fits an OLS regression.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame containing the merged FC results and ROI t values. Expected columns include:
        'Regressor', 'Beta', 'Entropy_ROI1', 'Color_ROI2', 'State_ROI2', 'Task_ROI2', etc.
    effect_type : str
        Which effect to model; should match the entries in the 'Regressor' column ("main" or "interaction").
    moderator : str
        Which column to use as the moderator e.g., 'Entropy'
    iv_list : list of str
        List of columns to include e.g., ['Color','State','Task']; must match the formatting in data frame
    winsor_limits : tuple of float, optional
        Limits for winsorization (default (0.005, 0.005)).
    
    Returns
    -------
    result : RegressionResultsWrapper
        The fitted regression model results.
    """
    # Subset the data for the chosen effect type.
    df_subset = merged_df[merged_df['Regressor'] == effect_type].copy()
    
    # Define the IV columns. ['taskPE','statePE','Color','State','Task']
    iv_cols = [(moderator+'_ROI1'), (moderator+'_ROI2')]
    for roi_opt in ['_ROI1','_ROI2']:
        for iv_opt in iv_list:
            iv_cols.append((iv_opt+roi_opt))
    
    # Winsorize and then center the independent variables.
    for col in iv_cols:
        # Winsorize returns a masked array; convert to a regular numpy array with .data.
        df_subset[col + '_w'] = winsorize(df_subset[col], limits=winsor_limits).data
        df_subset[col + '_c'] = df_subset[col + '_w'] - df_subset[col + '_w'].mean()
    
    # Create interaction terms between centered Entropy_ROI1 and each centered ROI2 variable.
    for iv_opt in iv_list:
        df_subset[(moderator+'_ROI1_c:'+iv_opt+'_ROI2_c')] = df_subset[(moderator+'_ROI1_c')] * df_subset[iv_opt+'_ROI2_c']
        df_subset[iv_opt+'_ROI1_c:'+iv_opt+'_ROI2_c'] = df_subset[iv_opt+'_ROI1_c'] * df_subset[iv_opt+'_ROI2_c']
    df_subset[(moderator+'_ROI1_c:'+moderator+'_ROI2_c')]  = df_subset[(moderator+'_ROI1_c')] * df_subset[(moderator+'_ROI2_c')]

    # Define the regression formula.
    # EX: 'Beta ~ entropy_ROI2_c*taskPE_ROI1_c + entropy_ROI2_c*statePE_ROI1_c + entropy_ROI2_c*Color_ROI1_c + entropy_ROI2_c*State_ROI1_c + entropy_ROI2_c*Task_ROI1_c'
    formula = "Beta ~ "
    for idx, iv_opt in enumerate(iv_list):
        formula = formula + moderator + "_ROI2_c*"+iv_opt+"_ROI1_c"
        if int(idx+1) != len(iv_list):
            formula = formula + " + "
    
    # Fit the regression model.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = smf.ols(formula, data=df_subset)
        result = model.fit()
    
    print(f"Regression results for {effect_type} effect with winsorization:")
    print(result.summary())
    
    return result

import networkx as nx
import community as community_louvain  # python-louvain

def detect_modularity(adj_matrix, resolution=1.0, random_state=42):
    """
    Detect communities using the Louvain algorithm on the provided adjacency matrix.
    
    Negative weights are set to zero to ensure non-negative node degrees.
    
    Parameters
    ----------
    adj_matrix : np.ndarray or pd.DataFrame
        A square adjacency matrix (e.g., 400x400). If a DataFrame is provided, its underlying
        values are used.
    resolution : float, optional
        Resolution parameter for Louvain (default 1.0).
    random_state : int, optional
        Seed for reproducibility.
        
    Returns
    -------
    partition : dict
        Mapping from ROI index (assumed 1-indexed) to community label.
    """
    if isinstance(adj_matrix, pd.DataFrame):
        mat = adj_matrix.values
    else:
        mat = adj_matrix
    # Set negative weights to zero.
    mat_nonneg = np.copy(mat)
    mat_nonneg[mat_nonneg < 0] = 0
    
    # Create a graph; nodes will be labeled 0,...,n-1.
    G = nx.from_numpy_array(mat_nonneg)
    partition_0_based = community_louvain.best_partition(G, resolution=resolution, random_state=random_state)
    # Convert node keys to 1-indexed.
    partition = {node + 1: community for node, community in partition_0_based.items()}
    return partition

def visualize_fc_heatmap(fc_df, regressor, fdr_threshold=0.05, title="FC t-stats Heatmap", vmin=None, vmax=None, resolution=1.0):
    """
    Visualize FC t-statistics as a heatmap using data from a regression results DataFrame,
    with additional functionalities:
      1. Specify a color bar range via vmin and vmax.
      2. Determine the modularity partition using the "main" effect t-values, and reorder the matrix
         so that ROIs belonging to the same module are grouped together on both axes.
    
    The input DataFrame (fc_df) is expected to contain at least the columns:
        'ROI1_index', 'ROI2_index', 'Regressor', 't_value', 'p_value'.
    
    Parameters
    ----------
    fc_df : pd.DataFrame
        DataFrame containing regression results.
    regressor : str
        The regressor of interest (e.g., "main" or "interaction") for the displayed heatmap.
    fdr_threshold : float, optional
        The significance threshold for FDR-corrected p-values (default 0.05).
    title : str, optional
        Title for the heatmap.
    vmin : float, optional
        Minimum value for the color bar (if None, determined automatically).
    vmax : float, optional
        Maximum value for the color bar (if None, determined automatically).
    resolution : float, optional
        Resolution parameter for the modularity (Louvain) algorithm.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for the heatmap.
    partition : dict
        Dictionary mapping ROI index (1-indexed) to community label (determined from the "main" effect).
    """
    # 1. Create t and p matrices for the selected regressor.
    df_reg = fc_df[fc_df['Regressor'] == regressor].copy()
    df_grouped = df_reg.groupby(['ROI1_index', 'ROI2_index'], as_index=False).agg({ 't_value': 'mean', 'p_value': 'mean' })
    t_matrix = df_grouped.pivot(index='ROI1_index', columns='ROI2_index', values='t_value')
    p_matrix = df_grouped.pivot(index='ROI1_index', columns='ROI2_index', values='p_value')
    
    # Ensure matrices are sorted.
    t_matrix = t_matrix.sort_index().sort_index(axis=1)
    p_matrix = p_matrix.sort_index().sort_index(axis=1)
    
    # 2. Compute FDR correction.
    p_vals = p_matrix.values.flatten()
    valid = ~np.isnan(p_vals)
    p_vals_valid = p_vals[valid]
    if len(p_vals_valid) > 0:
        rejected, p_vals_corrected = fdrcorrection(p_vals_valid, alpha=fdr_threshold)
        p_corrected = np.full(p_matrix.shape, np.nan)
        p_corrected[np.where(~np.isnan(p_matrix))] = p_vals_corrected
    else:
        p_corrected = p_matrix.values.copy()
    mask_insig = (p_corrected > fdr_threshold)
    t_display = t_matrix.copy()
    t_display[mask_insig] = np.nan
    
    # 3. Determine modularity ordering using the "main" effect.
    df_main = fc_df[fc_df['Regressor'] == "main"].copy()
    df_main_grouped = df_main.groupby(['ROI1_index', 'ROI2_index'], as_index=False).agg({ 't_value': 'mean' })
    t_matrix_main = df_main_grouped.pivot(index='ROI1_index', columns='ROI2_index', values='t_value')
    t_matrix_main = t_matrix_main.sort_index().sort_index(axis=1)
    t_for_modularity = t_matrix_main.fillna(0)
    partition = detect_modularity(t_for_modularity, resolution=resolution)
    # Order ROIs: sort first by module then by ROI index.
    order = sorted(t_matrix_main.index, key=lambda roi: (partition.get(roi, float('inf')), roi))
    
    # Reorder rows and columns in the displayed matrix.
    t_display_ordered = t_display.loc[order, order]
    
    # 4. Plot the heatmap.
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(t_display_ordered, annot=False, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax, cbar_kws={'label': 't-statistic'}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("ROI2")
    ax.set_ylabel("ROI1")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    
    return fig, partition

def plot_regression_effects(result, coef_list=None, custom_tick_labels=None, title="Regression Effects", figsize=(3,5)):
    """
    Plot regression coefficients with their 95% confidence intervals from a statsmodels
    regression results object.
    
    Parameters
    ----------
    result : RegressionResultsWrapper
        A fitted regression model result from statsmodels.
    coef_list : list of str, optional
        List of coefficient names to plot. If None, all coefficients are plotted.
    custom_tick_labels : list of str, optional
        Custom labels for the y-axis ticks. If provided, this list will be used instead of the
        coefficient names.
    title : str, optional
        Title for the plot.
    figsize : tuple of (float, float), optional
        Figure size (width, height). If provided, it overrides the default size. For example, use
        figsize=(3,5) to obtain a 3:5 aspect ratio.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for the plot.
    """
    # Extract coefficient estimates and confidence intervals.
    params = result.params
    conf_int = result.conf_int()  # DataFrame with lower and upper bounds
    conf_int.columns = ['CI_lower', 'CI_upper']
    
    # If a coefficient list is provided, filter the series and confidence intervals.
    if coef_list is not None:
        params = params[coef_list]
        conf_int = conf_int.loc[coef_list]
    
    # Prepare plotting data.
    coef_names = params.index.tolist()
    estimates = params.values
    lower_errors = estimates - conf_int['CI_lower'].values
    upper_errors = conf_int['CI_upper'].values - estimates
    error_bars = np.vstack((lower_errors, upper_errors))
    
    # Determine default figure size if not provided.
    if figsize is None:
        figsize = (8, 0.5 * len(coef_names) + 1)
    
    # Create horizontal forest plot.
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(coef_names))
    ax.errorbar(estimates, y_positions, xerr=error_bars, fmt='o', color='black',
                ecolor='gray', capsize=5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Set y-tick positions and labels.
    ax.set_yticks(y_positions)
    if custom_tick_labels is not None:
        ax.set_yticklabels(custom_tick_labels)
    else:
        ax.set_yticklabels(coef_names)
    
    # Remove tick marks but keep the labels.
    ax.tick_params(axis='y', which='both', length=0)
    
    # Append (A.U.) to indicate arbitrary units.
    ax.set_xlabel("Coefficient Estimate (A.U.)")
    ax.set_title(title)
    
    # Remove the box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout(pad=2)
    plt.show()
    
    return fig

###############################################################
# For cross epoch analysis.
###############################################################
def combine_cue_probe_data(cue_df, probe_df, source_epo='cue', target_epo='probe', key_columns=None):
    """
    Combine cue and probe beta series dataframes by renaming ROI columns and merging them.
    
    The ROI columns are assumed to have names starting with digits and a colon (e.g., "1:...").
    In the cue dataframe these columns are renamed with the prefix "cue_" followed by the ROI index,
    and in the probe dataframe with the prefix "probe_" followed by the ROI index.
    
    Parameters
    ----------
    cue_df : pd.DataFrame
        DataFrame containing beta series for cue epochs.
    probe_df : pd.DataFrame
        DataFrame containing beta series for probe epochs.
    key_columns : list of str, optional
        Columns on which to merge the dataframes (e.g., subject or run identifiers).
        If None, the merge is performed on the index.
    
    Returns
    -------
    merged_df : pd.DataFrame
        Combined dataframe with non-ROI columns (from cue_df, assumed identical to probe_df)
        and with ROI beta series columns renamed as "cue_<index>" and "probe_<index>".
    """
    # Identify ROI columns (those that start with digits followed by a colon)
    roi_pattern = re.compile(r"^(\d+):")
    cue_roi_cols = [col for col in cue_df.columns if roi_pattern.match(col)]
    probe_roi_cols = [col for col in probe_df.columns if roi_pattern.match(col)]
    
    # Create copies so as not to modify the originals.
    cue_df_renamed = cue_df.copy()
    probe_df_renamed = probe_df.copy()
    
    # Rename the ROI columns with "cue_" or "probe_" followed by the ROI index.
    cue_df_renamed = cue_df_renamed.rename(columns={
        col: (source_epo+"_") + roi_pattern.match(col).group(1) for col in cue_roi_cols
    })
    probe_df_renamed = probe_df_renamed.rename(columns={
        col: (target_epo+"_") + roi_pattern.match(col).group(1) for col in probe_roi_cols
    })
    
    # Merge the two dataframes.
    if key_columns is not None:
        merged_df = pd.merge(cue_df_renamed, probe_df_renamed, on=key_columns, suffixes=('', '_probe'))
    else:
        merged_df = pd.merge(cue_df_renamed, probe_df_renamed, left_index=True, right_index=True, suffixes=('', '_probe'))
    
    return merged_df

def fc_analysis_mixed_effects_cross(sdf, probe_roi_names_list, cue_roi_names_list,
                                    moderator='pSpP_entropy', source_epo='cue', target_epo='probe', winsor_limits=(0.005,0.005), n_jobs=-1):
    """
    For each cross-epoch ROI pair (dependent variable from probe and independent variable from cue),
    run a mixed effects regression across all subjects using a random intercept for subject.
    
    The model is:
         probe_ROI ~ 1 + cue_ROI * pSpP_entropy
         
    The function extracts the coefficient, t-value, and p-value for:
      - the main effect of cue ROI ("main"), and
      - the interaction between cue ROI and the moderator ("interaction").
    
    The output is organized into a DataFrame with columns:
        Subject, ROI1, ROI2, ROI1_index, ROI1_label, ROI2_index, ROI2_label,
        Regressor, Beta, t_value, p_value, n_obs.
    Here, "Subject" is set to "Mixed" to indicate group-level analysis.
    
    Parameters
    ----------
    sdf : pd.DataFrame
        DataFrame containing trial-level data with ROI beta columns, moderator (e.g., 'pSpP_entropy'),
        and subject identifiers.
    probe_roi_names_list : list of str
        List of ROI column names from the probe epoch (e.g., "probe_1:Label", "probe_2:Label", ..., "probe_400:Label").
    cue_roi_names_list : list of str
        List of ROI column names from the cue epoch (e.g., "cue_1:Label", "cue_2:Label", ..., "cue_400:Label").
    moderator : str
        Name of the moderator variable (default "pSpP_entropy").
    winsor_limits : tuple of float
        Winsorization limits.
    n_jobs : int
        Number of parallel jobs (default -1 uses all available cores).
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per regressor per ROI pair.
    """
    def process_roi_pair(probe_roi, cue_roi):
        # Run the mixed effects model using your existing function.
        try:
            res = run_model_for_roi_mixed_effects(probe_roi, cue_roi, sdf, moderator=moderator, winsor_limits=winsor_limits)
        except Exception as e:
            print(f"Error for ROI pair {probe_roi} and {cue_roi}: {e}")
            return []
        
        # Construct ROI index and label for probe ROI.
        # If the probe ROI name starts with "probe_", remove it.
        if probe_roi.startswith((target_epo+"_")):
            probe_part = probe_roi.split((target_epo+"_"))[1]
        else:
            probe_part = probe_roi
        # Split on ":" to separate index and label.
        probe_split = probe_part.split(":", 1)
        probe_index = probe_split[0]
        probe_label = probe_split[1] if len(probe_split) > 1 else probe_roi
        
        # Construct ROI index and label for cue ROI.
        if cue_roi.startswith((source_epo+"_")):
            cue_part = cue_roi.split((source_epo+"_"))[1]
        else:
            cue_part = cue_roi
        cue_split = cue_part.split(":", 1)
        cue_index = cue_split[0]
        cue_label = cue_split[1] if len(cue_split) > 1 else cue_roi
        
        # Define parameter names (assuming run_model_for_roi_mixed_effects returns a statsmodels result).
        param_cue = f'Q("{cue_roi}")'
        param_interaction = f'Q("{cue_roi}"):Q("{moderator}")'
        
        results = []
        for regressor, param_name in [('main', param_cue), ('interaction', param_interaction)]:
            beta = res.params.get(param_name, np.nan)
            t_val = res.tvalues.get(param_name, np.nan)
            p_val = res.pvalues.get(param_name, np.nan)
            results.append({
                'Subject': 'Mixed',
                'ROI1': probe_roi,
                'ROI2': cue_roi,
                'ROI1_index': probe_index,
                'ROI1_label': probe_label,
                'ROI2_index': cue_index,
                'ROI2_label': cue_label,
                'Regressor': regressor,
                'Beta': beta,
                't_value': t_val,
                'p_value': p_val,
                'n_obs': res.nobs
            })
        return results

    # Create all ROI pair combinations: probe ROI from probe list and cue ROI from cue list.
    roi_pairs = [(probe, cue) for probe in probe_roi_names_list for cue in cue_roi_names_list]
    
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_roi_pair)(probe_roi, cue_roi) for probe_roi, cue_roi in roi_pairs
    )
    # Flatten the list of results.
    flat_results = [item for sublist in results_list for item in sublist]
    results_df = pd.DataFrame(flat_results)
    return results_df


###############################################################
# For setting up data frames and decoding maps for analyses
###############################################################
def prep_IVs(moderator, source_epoch, target_epoch, data_path, project_folder_path):
    # get entropy, color, state prediction t values. etc
    # These decoding maps will be used in the regression as the IVs predicting FC
    if target_epoch!="feedback":
        df_entropy = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__entropy_'+source_epoch+'__r__tval.nii')))
        df_color = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__color_'+target_epoch+'__r__tval.nii')))
        df_state = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__state_'+target_epoch+'__r__tval.nii')))
        df_task = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__task_'+target_epoch+'__r__tval.nii')))
    else:
        df_entropy = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__entropy_fb__r__tval.nii')))
        df_color = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__color_fb__r__tval.nii')))
        df_state = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__state_fb__r__tval.nii')))
        df_task = get_avg_t_values(os.path.join(project_folder_path,('Decoding/GroupStats/GroupAnalysis_38subjs__task_fb__r__tval.nii')))
    # get taskPE, statePE, colorPE t values. 
    # These parametric modulation maps will be used in the regression as the IVs predicting FC
    df_taskPE = get_avg_t_values(os.path.join(project_folder_path,'3dMEMA/nii3D/'+target_epoch+'__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz'))
    df_statePE = get_avg_t_values(os.path.join(project_folder_path,'3dMEMA/nii3D/'+target_epoch+'__zStatePE_SPMGmodel_stats_REML__tval.nii.gz'))
    df_colorPE = get_avg_t_values(os.path.join(project_folder_path,'3dMEMA/nii3D/'+target_epoch+'__zColorPE_SPMGmodel_stats_REML__tval.nii.gz'))
    #df_stateD = get_avg_t_values(os.path.join(project_folder_path,'3dMEMA/nii3D/'+target_epoch+'__StateD_SPMGmodel_stats_REML__tval.nii.gz'))
    df_stateD = get_avg_t_values(os.path.join(project_folder_path,'3dMEMA/nii3D/cue__StateD_SPMGmodel_stats_REML__tval.nii.gz'))
    
    # Rename the avg_t column for each dataframe to identify the measure
    df_entropy = df_entropy.rename(columns={'avg_t': 'Entropy'})
    df_color   = df_color.rename(columns={'avg_t': 'Color'})
    df_state   = df_state.rename(columns={'avg_t': 'State'})
    df_task    = df_task.rename(columns={'avg_t': 'Task'})
    # Rename the avg_t column for each dataframe to identify the measure
    df_taskPE = df_taskPE.rename(columns={'avg_t': 'taskPE'})
    df_statePE   = df_statePE.rename(columns={'avg_t': 'statePE'})
    df_colorPE   = df_colorPE.rename(columns={'avg_t': 'colorPE'})
    df_stateD   = df_stateD.rename(columns={'avg_t': 'stateD'})
    
    # Merge them using ROI_index and ROI_label as keys
    df_roi = df_entropy.merge(df_taskPE, on=['ROI_index', 'ROI_label']) \
                        .merge(df_statePE, on=['ROI_index', 'ROI_label']) \
                        .merge(df_colorPE, on=['ROI_index', 'ROI_label']) \
                        .merge(df_stateD, on=['ROI_index', 'ROI_label']) \
                        .merge(df_color, on=['ROI_index', 'ROI_label']) \
                        .merge(df_state, on=['ROI_index', 'ROI_label']) \
                        .merge(df_task, on=['ROI_index', 'ROI_label'])
    # Prepare df_merged for merging for ROI1 and ROI2 separately.
    df_merged_roi1 = df_roi.rename(columns={
        'ROI_index': 'ROI1_index',
        'Entropy': 'Entropy_ROI1',
        'taskPE': 'taskPE_ROI1',
        'statePE': 'statePE_ROI1',
        'colorPE': 'colorPE_ROI1',
        'stateD': 'stateD_ROI1',
        'Color':   'Color_ROI1',
        'State':   'State_ROI1',
        'Task':    'Task_ROI1'})
    df_merged_roi2 = df_roi.rename(columns={
        'ROI_index': 'ROI2_index',
        'Entropy': 'Entropy_ROI2',
        'taskPE': 'taskPE_ROI2',
        'statePE': 'statePE_ROI2',
        'colorPE': 'colorPE_ROI2',
        'stateD': 'stateD_ROI2',
        'Color':   'Color_ROI2',
        'State':   'State_ROI2',
        'Task':    'Task_ROI2'})
    return df_merged_roi1, df_merged_roi2

def create_mod_df(mod_column="pSpP_entropy"):
    unusable_subs = ['10118', '10218', '10275', '10282', '10296', '10318', '10319', '10321', '10322', '10351', '10358'] # list of subjects to exclude
    subjects = sorted(os.listdir(bids_path)) # this folder only contains usable subjects
    subject_list = []
    for ii, subj in enumerate(subjects):
        if os.path.isdir(os.path.join(func_path,subj)):
            cur_subj = subj.split('-')[1]
            if cur_subj not in unusable_subs:
                subject_list.append(cur_subj)
    print("a total of ", len(subject_list), " subjects will be included")
    
    all_df_list = []
    for sub_idx, sub in enumerate(subject_list):
        cur_sub_df = pd.read_csv(os.path.join(project_folder_path,"model_data",("sub-"+sub+"_dataframe_3dDeconvolve_pSpP.csv")))
        cur_sub_df['Subject'] = int(sub)
        cur_sub_df = cur_sub_df[:200] # make sure we only use 5 runs
        all_df_list.append(cur_sub_df[["Run","trial",mod_column,"Subject"]])
        
    mod_df = pd.concat(all_df_list, ignore_index=True)

    return mod_df



###############################################################
###############################################################
###############################################################
if __name__ == '__main__':
    
    # Define paths and file patterns.
    if os.path.exists("/Shared"):
        data_path = func_path # "/Shared/lss_kahwang_hpc/data/FPNHIU/3dDeconvolve/"
        project_folder_path = dataset_dir # "/Shared/lss_kahwang_hpc/data/FPNHIU"
    else:
        data_path = func_path # "/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/3dDeconvolve/"
        project_folder_path = dataset_dir # "/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU"
    
    
    # CSV with trial-level info including 'Subject' and 'pSpP_entropy'
    df = create_mod_df(mod_column="pSpP_entropy")
    df.to_csv(os.path.join(project_folder_path,"FC","FC_moderator_dataframe_38subjs.csv"))
    #df = pd.read_csv(os.path.join(project_folder_path,"model_data/entropy_dataframe_38subjs.csv"))
    subjects = df['Subject'].unique()
    
    
    # ######################################################################
    # ####### Do CUE beta series FC
    # ######################################################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="cue", target_epoch="cue", data_path=func_path, project_folder_path=dataset_dir)
    # file_pattern = "cue.LSS.nii.gz"
    # output_data_path = os.path.join(project_folder_path, "FC", "cue_data.npy")
    # output_sdf_path = os.path.join(project_folder_path, "FC", "cue_df.csv")
    
    # # Extract Schaefer ROI betas and organize subject data.
    # data_vec, sdf = organize_schaefer_LSS_betas(
    #                                             subjects=subjects,
    #                                             df=df,
    #                                             data_path=data_path,
    #                                             file_pattern=file_pattern,
    #                                             output_data_path=output_data_path,
    #                                             output_sdf_path=output_sdf_path)
    
    # # Create list of ROI names as they appear in the dataframe.
    # # They are named as "index:Label" from 1 to 400.
    # schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    # roi_indices = np.arange(1, 401)
    # roi_names_list = [f"{roi_indices[i]}:{schaefer.labels[i]}" for i in range(400)]
    
    # # run FC
    # sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "cue_df.csv"))
    # fc_results_df = fc_analysis_mixed_effects(sdf, roi_names_list, moderator='pSpP_entropy', n_jobs=36)
    # fc_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_results.csv"), index=False)
    # fc_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_results.csv"))

    # # now merge with the FC results df
    # fc_results_df['ROI1_index'] = fc_results_df['ROI1_index'].astype(int)
    # fc_results_df['ROI2_index'] = fc_results_df['ROI2_index'].astype(int)
    
    
    # # Merge on ROI1_index first.
    # merged_df = fc_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # merged_df = merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )

    # ## do stats on merged_df
    # cue_result_interaction = run_fc_regression(merged_df, effect_type="interaction", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005)) 
    # cue_result_main = run_fc_regression(merged_df, effect_type="main", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005))

    # fig, part = visualize_fc_heatmap(fc_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "cue_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')
    
    # coefs_to_plot = ["Entropy_ROI2_c", "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Source:Entropy", "Target:Color", "Target:State", "Target:Task", "Source:Entropy * Target:Color", "Source:Entropy * Target:State", "Source:Entropy * Target:Task"]
    # fig = plot_regression_effects(cue_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Cue Beta Series interaction Effects", figsize=(5.4,3))
    # fig.savefig(os.path.join(project_folder_path,"FC","cue_fc_intraction_coefs.png"), dpi=300, bbox_inches='tight')



    # ######################################################################
    # ####### Do PROBE beta series 
    # ######################################################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="probe", target_epoch="probe", data_path=func_path, project_folder_path=dataset_dir)
    # file_pattern = "probe.LSS.nii.gz"
    # output_data_path = os.path.join(project_folder_path, "FC", "probe_data.npy")
    # output_sdf_path = os.path.join(project_folder_path, "FC", "probe_df.csv")
    # probe_data_vec, probe_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                                         df=df,
    #                                                         data_path=data_path,
    #                                                         file_pattern=file_pattern,
    #                                                         output_data_path=output_data_path,
    #                                                         output_sdf_path=output_sdf_path)

    # probe_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "probe_df.csv"))
    
    # # Create list of ROI names as they appear in the dataframe.
    # # They are named as "index:Label" from 1 to 400.
    # schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    # roi_indices = np.arange(1, 401)
    # roi_names_list = [f"{roi_indices[i]}:{schaefer.labels[i]}" for i in range(400)]
    
    # fc_probe_results_df = fc_analysis_mixed_effects(probe_sdf, roi_names_list, moderator='pSpP_entropy', n_jobs=36)
    # fc_probe_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_results.csv"), index=False)
    
    # fc_probe_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_results.csv"))
    # fc_probe_results_df['ROI1_index'] = fc_probe_results_df['ROI1_index'].astype(int)
    # fc_probe_results_df['ROI2_index'] = fc_probe_results_df['ROI2_index'].astype(int)

    # probe_merged_df = fc_probe_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # probe_merged_df = probe_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )

    # probe_result_interaction = run_fc_regression(probe_merged_df, effect_type="interaction", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005))
    # probe_result_main = run_fc_regression(probe_merged_df, effect_type="main", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005))

    # ###################################
    # ## visualize DF
    # ###################################
    # fig, part = visualize_fc_heatmap(fc_probe_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "probe_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # fig, part = visualize_fc_heatmap(fc_probe_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "probe_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    # coefs_to_plot = ["Entropy_ROI2_c", "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Source:Entropy", "Target:Color", "Target:State", "Target:Task", "Source:Entropy * Target:Color", "Source:Entropy * Target:State", "Source:Entropy * Target:Task"]
    # fig = plot_regression_effects(probe_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Probe Beta Series interaction Effects",figsize=(5.4,3))
    # fig.savefig(os.path.join(project_folder_path, "FC", "probe_fc_intraction_coefs.png"), dpi=300, bbox_inches='tight')  

    
    
    # ######################################################################
    # ####### Do FEEDBACK beta series 
    # ######################################################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="feedback", target_epoch="feedback", data_path=func_path, project_folder_path=dataset_dir)
    # file_pattern = "fb.LSS.nii.gz"
    # output_data_path = os.path.join(project_folder_path, "FC", "fb_data.npy")
    # output_sdf_path = os.path.join(project_folder_path, "FC", "fb_df.csv")
    # fb_data_vec, fb_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                                         df=df,
    #                                                         data_path=data_path,
    #                                                         file_pattern=file_pattern,
    #                                                         output_data_path=output_data_path,
    #                                                         output_sdf_path=output_sdf_path)
    # # Create list of ROI names as they appear in the dataframe.
    # # They are named as "index:Label" from 1 to 400.
    # schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    # roi_indices = np.arange(1, 401)
    # roi_names_list = [f"{roi_indices[i]}:{schaefer.labels[i]}" for i in range(400)]
    
    # fb_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "fb_df.csv"))
    
    # fc_fb_results_df = fc_analysis_mixed_effects(fb_sdf, roi_names_list, moderator='pSpP_entropy', n_jobs=36)
    # fc_fb_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_feedback_results.csv"), index=False)
    
    # fc_fb_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_feedback_results.csv"))
    # fc_fb_results_df['ROI1_index'] = fc_fb_results_df['ROI1_index'].astype(int)
    # fc_fb_results_df['ROI2_index'] = fc_fb_results_df['ROI2_index'].astype(int)

    # fb_merged_df = fc_fb_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'taskPE_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # fb_merged_df = fb_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'taskPE_ROI2','Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )
    
    # fb_result_interaction = run_fc_regression(fb_merged_df, effect_type="interaction", moderator="Entropy", iv_list=['taskPE','Color','State','Task'], winsor_limits=(0.005, 0.005))
    # fb_result_main = run_fc_regression(fb_merged_df, effect_type="main", moderator="Entropy", iv_list=['taskPE','Color','State','Task'], winsor_limits=(0.005, 0.005))
    
    # ###################################
    # ## visualize DF
    # ###################################
    # fig, part = visualize_fc_heatmap(fc_fb_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # fig, part = visualize_fc_heatmap(fc_fb_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    # coefs_to_plot = ["Entropy_ROI2_c", "taskPE_ROI1_c",
    #                  "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", 
    #                  "Entropy_ROI2_c:taskPE_ROI1_c",
    #                  "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Source:Entropy", "Target:taskPE",
    #                "Target:Color", "Target:State", "Target:Task", 
    #                "Source:Entropy * Target:taskPE",
    #                "Source:Entropy * Target:Color", "Source:Entropy * Target:State", "Source:Entropy * Target:Task"]
    # # coefs_to_plot = ["Entropy_ROI2_c", "taskPE_ROI1_c", "statePE_ROI1_c", "colorPE_ROI1_c", "stateD_ROI1_c", "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", 
    # #                  "Entropy_ROI2_c:taskPE_ROI1_c", "Entropy_ROI2_c:statePE_ROI1_c", "Entropy_ROI2_c:colorPE_ROI1_c", "Entropy_ROI2_c:stateD_ROI1_c", 
    # #                  "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # # coef_labels = ["Source:Entropy", "Target:taskPE", "Target:statePE", "Target:colorPE", "Target:stateD", "Target:Color", "Target:State", "Target:Task", 
    # #                "Source:Entropy * Target:taskPE", "Target:Entropy * Target:statePE", "Target:Entropy * Target:colorPE", "Target:Entropy * Target:stateD",
    # #                "Source:Entropy * Target:Color", "Source:Entropy * Target:State", "Source:Entropy * Target:Task"]
    # fig = plot_regression_effects(fb_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Feedback Beta Series interaction Effects",figsize=(5.4,3))
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_intraction_coefs.png"), dpi=300, bbox_inches='tight')




    # ###################################
    # ## cross epoch fc of CUE -> PROBE
    # ###################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="cue", target_epoch="probe", data_path=func_path, project_folder_path=dataset_dir)
    # data_vec, sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                             df=df,
    #                                             data_path=data_path,
    #                                             file_pattern="cue.LSS.nii.gz",
    #                                             output_data_path=os.path.join(project_folder_path, "FC", "cue_data.npy"),
    #                                             output_sdf_path=os.path.join(project_folder_path, "FC", "cue_df.csv"))
    # sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "cue_df.csv"))
    # probe_data_vec, probe_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                                         df=df,
    #                                                         data_path=data_path,
    #                                                         file_pattern="probe.LSS.nii.gz",
    #                                                         output_data_path=os.path.join(project_folder_path, "FC", "probe_data.npy"),
    #                                                         output_sdf_path=os.path.join(project_folder_path, "FC", "probe_df.csv"))
    # probe_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "probe_df.csv")) 
    
    # cdf = combine_cue_probe_data(sdf, probe_sdf)

    # roi_pattern = re.compile(r"^(\d+):")
    # probe_roi_names_list = [col for col in probe_sdf.columns if roi_pattern.match(col)]
    # probe_roi_names_list = ["probe_" + roi_pattern.match(col).group(1) for col in probe_roi_names_list]

    # roi_pattern = re.compile(r"^(\d+):")
    # cue_roi_names_list = [col for col in sdf.columns if roi_pattern.match(col)]
    # cue_roi_names_list = ["cue_" + roi_pattern.match(col).group(1) for col in cue_roi_names_list]

    # cue_probe_fc_results_df = fc_analysis_mixed_effects_cross(cdf, probe_roi_names_list, cue_roi_names_list, moderator='pSpP_entropy', n_jobs=36)
    # cue_probe_fc_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_cue_probe_results.csv"), index=False)
    # cue_probe_fc_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_cue_probe_results.csv"))
    # cue_probe_fc_results_df['ROI1_index'] = cue_probe_fc_results_df['ROI1_index'].astype(int)
    # cue_probe_fc_results_df['ROI2_index'] = cue_probe_fc_results_df['ROI2_index'].astype(int)
    # cue_probe_merged_df = cue_probe_fc_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # cue_probe_merged_df = cue_probe_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )

    # cue_probe_result_interaction = run_fc_regression(cue_probe_merged_df, effect_type="interaction", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005))
    # cue_probe_result_main = run_fc_regression(cue_probe_merged_df, effect_type="main", moderator="Entropy", iv_list=['Color','State','Task'], winsor_limits=(0.005, 0.005))  

    # fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "cue_probe_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # # fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # # fig.savefig(os.path.join(project_folder_path, "FC", "cue_probe_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    # coefs_to_plot = ["Entropy_ROI2_c", "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Cue:Entropy", "Probe:Color", "Probe:State", "Probe:Task", "Cue:Entropy * Probe:Color", "Cue:Entropy * Probe:State", "Cue:Entropy * Probe:Task"]
    # fig = plot_regression_effects(cue_probe_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Cue -> Probe Beta Series interaction Effects",figsize=(5.8,3))
    # fig.savefig(os.path.join(project_folder_path, "FC", "cue_probe_fc_intraction_coefs.png"), dpi=300, bbox_inches='tight')
    
    
    # ###################################
    # ## cross epoch fc of PROBE -> FEEDBACK
    # ###################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="probe", target_epoch="feedback", data_path=func_path, project_folder_path=dataset_dir)
    # probe_data_vec, probe_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                                         df=df,
    #                                                         data_path=data_path,
    #                                                         file_pattern="probe.LSS.nii.gz",
    #                                                         output_data_path=os.path.join(project_folder_path, "FC", "probeE_data.npy"),
    #                                                         output_sdf_path=os.path.join(project_folder_path, "FC", "probeE_df.csv"))
    # fb_data_vec, fb_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    #                                                         df=df,
    #                                                         data_path=data_path,
    #                                                         file_pattern="fb.LSS.nii.gz",
    #                                                         output_data_path=os.path.join(project_folder_path, "FC", "fbE_data.npy"),
    #                                                         output_sdf_path=os.path.join(project_folder_path, "FC", "fbE_df.csv"))
    # probe_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "probeE_df.csv"))
    # fb_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "fbE_df.csv"))
    
    # cdf = combine_cue_probe_data(probe_sdf, fb_sdf, source_epo='probe', target_epo='feedback')
    
    # roi_pattern = re.compile(r"^(\d+):")
    # target_roi_names_list = [col for col in fb_sdf.columns if roi_pattern.match(col)]
    # target_roi_names_list = ["feedback_" + roi_pattern.match(col).group(1) for col in target_roi_names_list]

    # roi_pattern = re.compile(r"^(\d+):")
    # source_roi_names_list = [col for col in probe_sdf.columns if roi_pattern.match(col)]
    # source_roi_names_list = ["probe_" + roi_pattern.match(col).group(1) for col in source_roi_names_list]

    # cue_probe_fc_results_df = fc_analysis_mixed_effects_cross(cdf, target_roi_names_list, source_roi_names_list, moderator='pSpP_entropy', source_epo='probe', target_epo='feedback', n_jobs=36)
    # cue_probe_fc_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_feedback_results.csv"), index=False)
    
    # cue_probe_fc_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_feedback_results.csv"))
    # cue_probe_fc_results_df['ROI1_index'] = cue_probe_fc_results_df['ROI1_index'].astype(int)
    # cue_probe_fc_results_df['ROI2_index'] = cue_probe_fc_results_df['ROI2_index'].astype(int)
    # cue_probe_merged_df = cue_probe_fc_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'taskPE_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # cue_probe_merged_df = cue_probe_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'taskPE_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )
    
    # cue_probe_result_interaction = run_fc_regression(cue_probe_merged_df, moderator="Entropy", effect_type="interaction", iv_list=['taskPE','Color','State','Task'], winsor_limits=(0.005, 0.005))
    # cue_probe_result_main = run_fc_regression(cue_probe_merged_df, moderator="Entropy", effect_type="main", iv_list=['taskPE','Color','State','Task'], winsor_limits=(0.005, 0.005))  
    
    # fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "probe_feedback_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # # fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # # fig.savefig(os.path.join(project_folder_path, "FC", "cue_probe_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    # coefs_to_plot = ["Entropy_ROI2_c", "taskPE_ROI1_c",
    #                  "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", 
    #                  "Entropy_ROI2_c:taskPE_ROI1_c",
    #                  "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Probe:Entropy", "Feedback:taskPE",
    #                "Feedback:Color", "Feedback:State", "Feedback:Task", 
    #                "Probe:Entropy * Feedback:taskPE",
    #                "Probe:Entropy * Feedback:Color", "Probe:Entropy * Feedback:State", "Probe:Entropy * Feedback:Task"]
    # fig = plot_regression_effects(cue_probe_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Probe -> Feedback Beta Series interaction Effects",figsize=(5.8,3))
    # fig.savefig(os.path.join(project_folder_path, "FC", "probe_feedback_fc_intraction_coefs.png"), dpi=300, bbox_inches='tight')
    
    
    ####end planned analyses
    
    
    
    
    
    # ############################################################################################################################################
    # ############################################################################################################################################
    # ###########################################              Exporatory anlyses post-review             ########################################
    # ############################################################################################################################################
    # ############################################################################################################################################
    
    
    # ######################################################################
    # ####### Do FEEDBACK beta series 
    # ######################################################################
    # df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="feedback", target_epoch="feedback", data_path=func_path, project_folder_path=dataset_dir)
    # file_pattern = "fb.LSS.nii.gz"
    # output_data_path = os.path.join(project_folder_path, "FC", "fb_data2.npy")
    # output_sdf_path = os.path.join(project_folder_path, "FC", "fb_df2.csv")
    # # fb_data_vec, fb_sdf = organize_schaefer_LSS_betas(subjects=subjects,
    # #                                                         df=df,
    # #                                                         data_path=data_path,
    # #                                                         file_pattern=file_pattern,
    # #                                                         output_data_path=output_data_path,
    # #                                                         output_sdf_path=output_sdf_path)
    # # Create list of ROI names as they appear in the dataframe.
    # # They are named as "index:Label" from 1 to 400.
    # schaefer = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    # roi_indices = np.arange(1, 401)
    # roi_names_list = [f"{roi_indices[i]}:{schaefer.labels[i]}" for i in range(400)]
    
    # fb_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "fb_df2.csv"))
    
    # fc_fb_results_df = fc_analysis_mixed_effects(fb_sdf, roi_names_list, moderator='pSpP_entropy', n_jobs=20)
    # fc_fb_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_feedback_results_EXPLORATORY.csv"), index=False)
    
    # fc_fb_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_feedback_results_EXPLORATORY.csv"))
    # fc_fb_results_df['ROI1_index'] = fc_fb_results_df['ROI1_index'].astype(int)
    # fc_fb_results_df['ROI2_index'] = fc_fb_results_df['ROI2_index'].astype(int)

    # fb_merged_df = fc_fb_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'stateD_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    # fb_merged_df = fb_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'stateD_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )

    # fb_result_interaction = run_fc_regression(fb_merged_df, effect_type="interaction", moderator="Entropy", iv_list=['stateD','Color','State','Task'], winsor_limits=(0.005, 0.005))
    # fb_result_main = run_fc_regression(fb_merged_df, effect_type="main", moderator="Entropy", iv_list=['stateD','Color','State','Task'], winsor_limits=(0.005, 0.005))

    # ###################################
    # ## visualize DF
    # ###################################
    # fig, part = visualize_fc_heatmap(fc_fb_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # fig, part = visualize_fc_heatmap(fc_fb_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    # coefs_to_plot = ["Entropy_ROI2_c", "stateD_ROI1_c", 
    #                  "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", 
    #                  "Entropy_ROI2_c:stateD_ROI1_c", 
    #                  "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    # coef_labels = ["Source:Entropy", "Target:stateD", 
    #                "Target:Color", "Target:State", "Target:Task", 
    #                "Target:Entropy * Target:stateD",
    #                "Source:Entropy * Target:Color", "Source:Entropy * Target:State", "Source:Entropy * Target:Task"]
    # fig = plot_regression_effects(fb_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Feedback Beta Series interaction Effects",figsize=(5.4,3))
    # fig.savefig(os.path.join(project_folder_path, "FC", "fb_fc_intraction_coefs_EXPLORATORY.png"), dpi=300, bbox_inches='tight')


    
    
    ###################################
    ## cross epoch fc of PROBE -> FEEDBACK
    ###################################
    df_merged_roi1, df_merged_roi2 = prep_IVs(moderator="entropy", source_epoch="probe", target_epoch="feedback", data_path=func_path, project_folder_path=dataset_dir)
    probe_data_vec, probe_sdf = organize_schaefer_LSS_betas(subjects=subjects,
                                                            df=df,
                                                            data_path=data_path,
                                                            file_pattern="probe.LSS.nii.gz",
                                                            output_data_path=os.path.join(project_folder_path, "FC", "probeE_data2.npy"),
                                                            output_sdf_path=os.path.join(project_folder_path, "FC", "probeE_df2.csv"))
    fb_data_vec, fb_sdf = organize_schaefer_LSS_betas(subjects=subjects,
                                                            df=df,
                                                            data_path=data_path,
                                                            file_pattern="fb.LSS.nii.gz",
                                                            output_data_path=os.path.join(project_folder_path, "FC", "fbE_data2.npy"),
                                                            output_sdf_path=os.path.join(project_folder_path, "FC", "fbE_df2.csv"))
    probe_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "probeE_df2.csv"))
    fb_sdf = pd.read_csv(os.path.join(project_folder_path, "FC", "fbE_df2.csv"))
    
    cdf = combine_cue_probe_data(probe_sdf, fb_sdf, source_epo='probe', target_epo='feedback')
    
    roi_pattern = re.compile(r"^(\d+):")
    target_roi_names_list = [col for col in fb_sdf.columns if roi_pattern.match(col)]
    target_roi_names_list = ["feedback_" + roi_pattern.match(col).group(1) for col in target_roi_names_list]

    roi_pattern = re.compile(r"^(\d+):")
    source_roi_names_list = [col for col in probe_sdf.columns if roi_pattern.match(col)]
    source_roi_names_list = ["probe_" + roi_pattern.match(col).group(1) for col in source_roi_names_list]

    cue_probe_fc_results_df = fc_analysis_mixed_effects_cross(cdf, target_roi_names_list, source_roi_names_list, moderator='pSpP_entropy', source_epo='probe', target_epo='feedback', n_jobs=20)
    cue_probe_fc_results_df.to_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_feedback_results_EXPLORATORY.csv"), index=False)
    
    cue_probe_fc_results_df = pd.read_csv(os.path.join(project_folder_path, "FC", "group_mixed_effects_fc_probe_feedback_results_EXPLORATORY.csv"))
    cue_probe_fc_results_df['ROI1_index'] = cue_probe_fc_results_df['ROI1_index'].astype(int)
    cue_probe_fc_results_df['ROI2_index'] = cue_probe_fc_results_df['ROI2_index'].astype(int)
    cue_probe_merged_df = cue_probe_fc_results_df.merge( df_merged_roi1[['ROI1_index', 'Entropy_ROI1', 'stateD_ROI1', 'Color_ROI1', 'State_ROI1', 'Task_ROI1']], on='ROI1_index', how='left' )
    cue_probe_merged_df = cue_probe_merged_df.merge( df_merged_roi2[['ROI2_index', 'Entropy_ROI2', 'stateD_ROI2', 'Color_ROI2', 'State_ROI2', 'Task_ROI2']], on='ROI2_index', how='left' )

    cue_probe_result_interaction = run_fc_regression(cue_probe_merged_df, moderator="Entropy", effect_type="interaction", iv_list=['stateD','Color','State','Task'], winsor_limits=(0.005, 0.005))
    cue_probe_result_main = run_fc_regression(cue_probe_merged_df, moderator="Entropy", effect_type="main", iv_list=['stateD','Color','State','Task'], winsor_limits=(0.005, 0.005))  

    fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="interaction", fdr_threshold=0.05, title="", vmin=-10, vmax=10, resolution=1.0)
    fig.savefig(os.path.join(project_folder_path, "FC", "probe_feedback_fc_intraction_heatmap.png"), dpi=300, bbox_inches='tight')

    # fig, part = visualize_fc_heatmap(cue_probe_fc_results_df, regressor="main", fdr_threshold=0.05, title="", vmin=-150, vmax=150, resolution=1.0)
    # fig.savefig(os.path.join(project_folder_path, "FC", "cue_probe_fc_main_heatmap.png"), dpi=300, bbox_inches='tight')

    coefs_to_plot = ["Entropy_ROI2_c", "stateD_ROI1_c",
                     "Color_ROI1_c", "State_ROI1_c", "Task_ROI1_c", 
                     "Entropy_ROI2_c:stateD_ROI1_c", 
                     "Entropy_ROI2_c:Color_ROI1_c", "Entropy_ROI2_c:State_ROI1_c", "Entropy_ROI2_c:Task_ROI1_c"]
    coef_labels = ["Probe:Entropy", "Feedback:stateD",
                   "Feedback:Color", "Feedback:State", "Feedback:Task", 
                   "Probe:Entropy * Feedback:stateD", 
                   "Probe:Entropy * Feedback:Color", "Probe:Entropy * Feedback:State", "Probe:Entropy * Feedback:Task"]
    fig = plot_regression_effects(cue_probe_result_interaction, coef_list=coefs_to_plot, custom_tick_labels=coef_labels, title=" Probe -> Feedback Beta Series interaction Effects",figsize=(5.8,3))
    fig.savefig(os.path.join(project_folder_path, "FC", "probe_feedback_fc_intraction_coefs_EXPLORATORY.png"), dpi=300, bbox_inches='tight')
    