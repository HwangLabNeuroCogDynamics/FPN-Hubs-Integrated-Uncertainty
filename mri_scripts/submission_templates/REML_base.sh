{
deconvolve_dir=${dataset_dir}3dDeconvolve/
fmriprep_dir=${dataset_dir}fmriprep/

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Run 3dDeconvolve WITH parametric modulation - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# # -- cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_errts_REML.nii.gz \
# -GOFORIT 100

# # -- cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_errts_REML.nii.gz \
# -GOFORIT 100

# -- cue_probe_fb__stateD_SPMGmodel_stats -- #
if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz ]; then
    rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz
    rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_REMLvar.nii.gz
    rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_errts_REML.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats.xmat.1D \
-input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
-mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
-Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
-Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__stateD_SPMGmodel_stats_errts_REML.nii.gz


# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # - - - - - - - - - - - - - - - - -   Old Models from Pre-review Process ... Included for Transparency    - - - - - - - - - - - - - - - - - - 
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # prior to review process, input (state and color) uncertainty, output (task) uncertainty, and integrated uncertainty (entropy) were run in 
# # separate models due to moderate to high collinearity between the different uncertainty estimates. However, based on the highly similar beta
# # maps between all of these models, reviewer 2 asked for one model with all uncertainty estimates included. We ended up only doing input and
# # integrated uncertainty in the model. We dropped output uncertainty since it was calculated based on a subset of cells used to calculate the
# # integrated uncertainty. We report the old models here for transparency.

# # -- cue_probe_fb__color_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__color_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__color_SPMGmodel_stats_errts_REML.nii.gz

# # -- cue_probe_fb__state_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__state_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__state_SPMGmodel_stats_errts_REML.nii.gz

# # -- cue_probe_fb__task_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__task_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__task_SPMGmodel_stats_errts_REML.nii.gz

# # -- cue_probe_fb__entropy_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__entropy_SPMGmodel_stats_errts_REML.nii.gz

# # -- cue_probe_fb__statePE_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__statePE_SPMGmodel_stats_errts_REML.nii.gz

# # -- cue_probe_fb__taskPE_SPMGmodel_stats -- #
# if [ -f ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz ]; then
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_REMLvar.nii.gz
#     rm ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_errts_REML.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dREMLfit -matrix ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats.xmat.1D \
# -input "${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-002_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-003_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz ${fmriprep_dir}sub-${sub}/func/sub-${sub}_task-Quantum_run-005_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
# -mask ${fmriprep_dir}sub-${sub}/combined_mask+tlrc.BRIK -fout -tout -rout \
# -Rbuck ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz -Rvar ${deconvolve_dir}sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats_REMLvar.nii.gz -verb $* \
# -Rerrts ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__taskPE_SPMGmodel_stats_errts_REML.nii.gz


# make sure all users have permission to read and write
cd ${deconvolve_dir}sub-${sub}/
chmod 775 *
}