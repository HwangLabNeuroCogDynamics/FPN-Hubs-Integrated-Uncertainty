{
# - - - - - - - - - - - - - - - - - - - - - - - - #
# - - -  run 3dDeconvolve prep script - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - #
python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/prep_3dDeconvolve.py ${sub} ${dataset_dir} --threshold 0.5
# threshold is for censor file generation


# - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Create the combinded mask - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - #
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dmask_tool -input $(find ${dataset_dir}fmriprep/sub-${sub}/func/*run-00[1-5]*mask.nii.gz) -prefix ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Set up for 3dDeconvolve WITH parametric modulation  - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#### amplitude modulation models see: https://afni.nimh.nih.gov/pub/dist/doc/misc/Decon/AMregression.pdf
# # z-score for uncertainty model
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zState_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zState_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zState_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zState_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zState_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zState_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zEntropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zEntropy_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zEntropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zEntropy_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zEntropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zEntropy_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColor_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColor_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zColor_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zColor_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zColor_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zColor_stimtime.1D
# # z-score for prediction error model
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zStatePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zStatePE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zStatePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zStatePE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zStatePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zStatePE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zTaskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zTaskPE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zTaskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zTaskPE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zTaskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zTaskPE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColorPE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zColorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_zColorPE_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zColorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_zColorPE_stimtime.1D
# add state derivative
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stateD_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stateD_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stateD_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stateD_stimtime.1D
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stateD_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stateD_stimtime.1D


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Run 3dDeconvolve WITH parametric modulation - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# note on brik outputs for SPMG with AM2 vs AM1
# AM2 brik: [0]=amplitude(unmodulated)  [1]=derivative(unmodulated)  [2]=amplitude(modulated)  [3]=derivative(modulated)
# AM1 brik: [0]=amplitude(modulated)    [1]=derivative(modulated)


# -- STATE & COLOR & ENTROPY-- #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort A \
-num_stimts 11 \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
-TR_times 2.039 \
-concat '1D: 0 489.36 978.72 1468.08 1957.44' \
-stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zState_stimtime.1D 'SPMG' -stim_label 1 cue_state \
-stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zState_stimtime.1D 'SPMG' -stim_label 2 probe_state \
-stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zState_stimtime.1D 'SPMG' -stim_label 3 feedback_state \
-stim_times_AM2 4 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColor_stimtime.1D 'SPMG' -stim_label 4 cue_color \
-stim_times_AM2 5 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zColor_stimtime.1D 'SPMG' -stim_label 5 probe_color \
-stim_times_AM1 6 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zColor_stimtime.1D 'SPMG' -stim_label 6 feedback_color \
-stim_times_AM2 7 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zEntropy_stimtime.1D 'SPMG' -stim_label 7 cue_entropy \
-stim_times_AM2 8 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zEntropy_stimtime.1D 'SPMG' -stim_label 8 probe_entropy \
-stim_times_AM1 9 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zEntropy_stimtime.1D 'SPMG' -stim_label 9 feedback_entropy \
-stim_times 10 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 10 feedback_incorrect \
-stim_times 11 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 11 feedback_correct \
-fout \
-rout \
-tout \
-bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 15


# -- extra check model putting prediction errors all in the same models
# # -- STATE & COLOR & TASK PE -- #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__colorPE_SPMGmodel_stats.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__colorPE_SPMGmodel_stats.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort A \
-num_stimts 11 \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
-TR_times 2.039 \
-concat '1D: 0 489.36 978.72 1468.08 1957.44' \
-stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zStatePE_stimtime.1D 'SPMG' -stim_label 1 cue_zStatePE \
-stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zStatePE_stimtime.1D 'SPMG' -stim_label 2 probe_zStatePE \
-stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zStatePE_stimtime.1D 'SPMG' -stim_label 3 feedback_zStatePE \
-stim_times_AM2 4 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zColorPE_stimtime.1D 'SPMG' -stim_label 4 cue_zColorPE \
-stim_times_AM2 5 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zColorPE_stimtime.1D 'SPMG' -stim_label 5 probe_zColorPE \
-stim_times_AM1 6 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zColorPE_stimtime.1D 'SPMG' -stim_label 6 feedback_zColorPE \
-stim_times_AM2 7 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_zTaskPE_stimtime.1D 'SPMG' -stim_label 7 cue_zTaskPE \
-stim_times_AM2 8 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_zTaskPE_stimtime.1D 'SPMG' -stim_label 8 probe_zTaskPE \
-stim_times_AM1 9 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_zTaskPE_stimtime.1D 'SPMG' -stim_label 9 feedback_zTaskPE \
-stim_times 10 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 10 feedback_incorrect \
-stim_times 11 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 11 feedback_correct \
-fout \
-rout \
-tout \
-bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 15


# -- STATE DERIVATIVE -- #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats.nii.gz
fi
#stateD
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort A \
-num_stimts 5 \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
-TR_times 2.039 \
-concat '1D: 0 489.36 978.72 1468.08 1957.44' \
-stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stateD_stimtime.1D 'SPMG' -stim_label 1 cue_stateD \
-stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_stateD_stimtime.1D 'SPMG' -stim_label 2 probe_stateD \
-stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_stateD_stimtime.1D 'SPMG' -stim_label 3 feedback_stateD \
-stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
-stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
-fout \
-rout \
-tout \
-bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__stateD_SPMGmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 15





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

# # - - - - - - - -  Models addressing input, integrated, and output uncertainty  - - - - - - - -
# # add uncertainty and entropy estimates
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_state_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_state_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_state_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_state_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_state_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_state_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_task_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_task_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_task_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_task_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_task_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_task_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_color_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_color_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_color_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_color_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_color_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_color_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_entropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_entropy_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_entropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_entropy_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_entropy_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_entropy_stimtime.1D

# #-- COLOR -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__color_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__color_SPMGmodel_stats.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_color_stimtime.1D 'SPMG' -stim_label 1 cue_color \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_color_stimtime.1D 'SPMG' -stim_label 2 probe_color \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_color_stimtime.1D 'SPMG' -stim_label 3 feedback_color \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__color_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12

# # -- STATE -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__state_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__state_SPMGmodel_stats.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_state_stimtime.1D 'SPMG' -stim_label 1 cue_state \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_state_stimtime.1D 'SPMG' -stim_label 2 probe_state \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_state_stimtime.1D 'SPMG' -stim_label 3 feedback_state \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__state_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12

# # -- TASK -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__task_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__task_SPMGmodel_stats.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_task_stimtime.1D 'SPMG' -stim_label 1 cue_task \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_task_stimtime.1D 'SPMG' -stim_label 2 probe_task \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_task_stimtime.1D 'SPMG' -stim_label 3 feedback_task \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__task_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12

# # -- ENTROPY -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats.nii.gz
# fi
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_entropy_stimtime.1D 'SPMG' -stim_label 1 cue_entropy \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_entropy_stimtime.1D 'SPMG' -stim_label 2 probe_entropy \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_entropy_stimtime.1D 'SPMG' -stim_label 3 feedback_entropy \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__entropy_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12


# # - - - - - - - -  Models addressing prediction errors  - - - - - - - -
# # add prediction errors
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_statePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_statePE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_statePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_statePE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_statePE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_statePE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_taskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_taskPE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_taskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_taskPE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_taskPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_taskPE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_colorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/cue_colorPE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_colorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/probe_colorPE_stimtime.1D
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 1dMarry ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_colorPE_am.1D | tail -n +3 > ${dataset_dir}3dDeconvolve/sub-${sub}/fb_colorPE_stimtime.1D

# # -- STATE PE -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats.nii.gz
# fi
# #statePE
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_statePE_stimtime.1D 'SPMG' -stim_label 1 cue_statePE \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_statePE_stimtime.1D 'SPMG' -stim_label 2 probe_statePE \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_statePE_stimtime.1D 'SPMG' -stim_label 3 feedback_statePE \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__statePE_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12

# #-- TASK PE -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats.nii.gz
# fi
# #taskPE
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_taskPE_stimtime.1D 'SPMG' -stim_label 1 cue_taskPE \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_taskPE_stimtime.1D 'SPMG' -stim_label 2 probe_taskPE \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_taskPE_stimtime.1D 'SPMG' -stim_label 3 feedback_taskPE \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__taskPE_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12

# # # -- COLOR PE -- #
# if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__colorPE_SPMGmodel_stats.nii.gz ]; then
#     rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__colorPE_SPMGmodel_stats.nii.gz
# fi
# #colorPE
# singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dDeconvolve \
# -force_TR 2.039 \
# -input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
# -mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
# -polort A \
# -num_stimts 5 \
# -censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
# -ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
# -TR_times 2.039 \
# -concat '1D: 0 489.36 978.72 1468.08 1957.44' \
# -stim_times_AM2 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_colorPE_stimtime.1D 'SPMG' -stim_label 1 cue_colorPE \
# -stim_times_AM2 2 ${dataset_dir}3dDeconvolve/sub-${sub}//probe_colorPE_stimtime.1D 'SPMG' -stim_label 2 probe_colorPE \
# -stim_times_AM1 3 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_colorPE_stimtime.1D 'SPMG' -stim_label 3 feedback_colorPE \
# -stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_incorrect.1D 'SPMG' -stim_label 4 feedback_incorrect \
# -stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}//fb_correct.1D 'SPMG' -stim_label 5 feedback_correct \
# -fout \
# -rout \
# -tout \
# -bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__colorPE_SPMGmodel_stats.nii.gz \
# -GOFORIT 100 \
# -noFDR \
# -jobs 12


# change permissions
cd ${dataset_dir}3dDeconvolve/sub-${sub}/
chmod 775 *
}