{
# modified based on Evan's notebook.
# https://github.com/HwangLabNeuroCogDynamics/Notebooks/blob/master/3dlss.ipynb
# tested using data from /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/sub-10280

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - -   3dTproject CODE   - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# because 3dLSS only takes one input, so can't give multiple runs as inputs.... we would have to do things somewhat creatively.
# first we use 3dTproject to do nuisance regression, you can use the same nuisance regressors you use for standard 3dDeconvolve. 
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dTproject \
-input $(ls ${dataset_dir}fmriprep/sub-${sub}/func/sub-${sub}_task-Quantum_run-00[1-5]_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz | sort -V) \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort 4 \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-cenmode ZERO \
-ort ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D \
-prefix ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-overwrite


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - -  3dDeconvolve CODE  - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# then we need to generate the design matrix by using 3dDeconvlve, and giving it the 1dstop option so it doenst 
# estimate the betas just saving out the design matrix. Because we are giving the concatenated errts as input,
# we need to manually specify the concatenation points. You should also be able to use the censor file in this call.

#As for the stim_times, here we are going to pull out single trial betas for the cue eooch, while putting all other events in the model. 
# We will have to do iterate through this 2 more times to get the probe and fb.  
# each time using a different epoch for the stim_times_IM option, while leaving the others in the stim_times column.
# In AFNI lingo "IM" stands for "Individually Modulated" (aka individual trial modulation) -- that is, each event will get its own amplitude estimated.

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# # - - - - - - - - - - - -     CUE     - - - - - - - - - - - - #
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue.LSS.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_IM.xmat.1D
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_Decon_REML.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_Decon_REMLvar.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue_LSS_errts_REML.nii.gz
fi

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort 0 \
-concat '1D: 0 240 480 720 960' \
-num_stimts 3 \
-local_times \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-stim_times_IM 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D 'SPMG' -stim_label 1 cue \
-stim_times 2 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D 'SPMG' -stim_label 2 probe \
-stim_times 3 ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D 'SPMG' -stim_label 3 feedback \
-x1D ${dataset_dir}3dDeconvolve/sub-${sub}/cue_IM.xmat.1D \
-x1D_stop \
-allzero_OK \
-GOFORIT 2 \
-jobs 5

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dREMLfit -matrix ${dataset_dir}3dDeconvolve/sub-${sub}/cue_IM.xmat.1D \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-GOFORIT 2 \
-Rbuck cue_Decon_REML.nii.gz \
-Rvar cue_Decon_REMLvar.nii.gz \
-Rerrts ${dataset_dir}3dDeconvolve/sub-${sub}/cue_LSS_errts_REML.nii.gz -verb

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# # - - - - - - - - - - - -    PROBE    - - - - - - - - - - - - #
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/probe.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe.LSS.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe_IM.xmat.1D
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe_Decon_REML.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe_Decon_REMLvar.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe_LSS_errts_REML.nii.gz
fi

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort 0 \
-concat '1D: 0 240 480 720 960' \
-num_stimts 3 \
-local_times \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-stim_times 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D 'SPMG' -stim_label 1 cue \
-stim_times_IM 2 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D 'SPMG' -stim_label 2 probe \
-stim_times 3 ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D 'SPMG' -stim_label 3 feedback \
-x1D ${dataset_dir}3dDeconvolve/sub-${sub}/probe_IM.xmat.1D \
-x1D_stop \
-allzero_OK \
-GOFORIT 2 \
-jobs 5

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dREMLfit -matrix ${dataset_dir}3dDeconvolve/sub-${sub}/probe_IM.xmat.1D \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-GOFORIT 2 \
-Rbuck probe_Decon_REML.nii.gz \
-Rvar probe_Decon_REMLvar.nii.gz \
-Rerrts ${dataset_dir}3dDeconvolve/sub-${sub}/probe_LSS_errts_REML.nii.gz -verb

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# # - - - - - - - - - - - -   Feedback  - - - - - - - - - - - - #
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/fb.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb.LSS.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb_IM.xmat.1D
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb_Decon_REML.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb_Decon_REMLvar.nii.gz
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb_LSS_errts_REML.nii.gz
fi

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dDeconvolve \
-force_TR 2.039 \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-polort 0 \
-concat '1D: 0 240 480 720 960' \
-num_stimts 3 \
-local_times \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-stim_times 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_stimtime.1D 'SPMG' -stim_label 1 cue \
-stim_times 2 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_stimtime.1D 'SPMG' -stim_label 2 probe \
-stim_times_IM 3 ${dataset_dir}3dDeconvolve/sub-${sub}/fb_stimtime.1D 'SPMG' -stim_label 3 feedback \
-x1D ${dataset_dir}3dDeconvolve/sub-${sub}/fb_IM.xmat.1D \
-x1D_stop \
-allzero_OK \
-GOFORIT 2 \
-jobs 5

singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dREMLfit -matrix ${dataset_dir}3dDeconvolve/sub-${sub}/fb_IM.xmat.1D \
-input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-GOFORIT 2 \
-Rbuck fb_Decon_REML.nii.gz \
-Rvar fb_Decon_REMLvar.nii.gz \
-Rerrts ${dataset_dir}3dDeconvolve/sub-${sub}/fb_LSS_errts_REML.nii.gz -verb



# # then run lss. See:
# # see https://www.sciencedirect.com/science/article/pii/S1053811911010081
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# # - - - - - - - - - -      LSS CODE     - - - - - - - - - - - #
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# # - - - -  CUE
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/cue.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/cue.LSS.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dLSS -input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-matrix ${dataset_dir}3dDeconvolve/sub-${sub}/cue_IM.xmat.1D \
-prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue.LSS.nii.gz \
-overwrite
# # - - - -  PROBE
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/probe.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/probe.LSS.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dLSS -input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-matrix ${dataset_dir}3dDeconvolve/sub-${sub}/probe_IM.xmat.1D \
-prefix ${dataset_dir}3dDeconvolve/sub-${sub}/probe.LSS.nii.gz \
-overwrite
# # - - - -  FEEDBACK
if [ -f ${dataset_dir}3dDeconvolve/sub-${sub}/fb.LSS.nii.gz ]; then
    rm ${dataset_dir}3dDeconvolve/sub-${sub}/fb.LSS.nii.gz
fi
singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dLSS -input ${dataset_dir}3dDeconvolve/sub-${sub}/errts.nii.gz \
-mask ${dataset_dir}fmriprep/sub-${sub}/combined_mask+tlrc.BRIK \
-matrix ${dataset_dir}3dDeconvolve/sub-${sub}/fb_IM.xmat.1D \
-prefix ${dataset_dir}3dDeconvolve/sub-${sub}/fb.LSS.nii.gz \
-overwrite


# the output should be a 4D data, with 400 datapoints, because we had 200 trials and each trial we estimated with 2 regressors (amplitdue and derivative from SPMG)? 
# We should be able to read those outputs and use nilearn for decoding analysis.
cd ${dataset_dir}3dDeconvolve/sub-${sub}/
chmod 775 *
}