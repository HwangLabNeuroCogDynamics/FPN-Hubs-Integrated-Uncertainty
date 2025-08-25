{
dataset_dir=/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - -   Verify state, color, and entropy are sig. different   - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for sub in 10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10353 10355 10356 10357 10359 10360 10374
do
    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__entropyR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[62]' -overwrite
    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__stateR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[10]' -overwrite
    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__colorR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[36]' -overwrite

    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__entropyR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[72]' -overwrite
    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__stateR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[20]' -overwrite
    3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__colorR2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[46]' -overwrite
done

python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/reviewer_requested_checks.py ${dataset_dir} --compare_state_color_entropy


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - -   Verify statePE, colorPE, and taskPE are sig. different   - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# for sub in 10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10353 10355 10356 10357 10359 10360 10374
# do
#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__taskPER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[58]' -overwrite
#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__statePER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[10]' -overwrite
#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue__colorPER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[36]' -overwrite

#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__taskPER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[68]' -overwrite
#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__statePER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[20]' -overwrite
#     3dTcat -prefix ${dataset_dir}'3dDeconvolve/sub-'${sub}'/probe__colorPER2_SPMGmodel_stats_REML.nii.gz' ${dataset_dir}'3dDeconvolve/sub-'${sub}'/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz[46]' -overwrite
# done

#python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/reviewer_requested_checks.py ${dataset_dir} --compare_statePE_colorPE_taskPE


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - -   Verify state and color decoding are sig. different   - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/reviewer_requested_checks.py ${dataset_dir} --compare_state_and_color_decoding


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - -   Verify probabilistic decoding is better than binary   - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/reviewer_requested_checks.py ${dataset_dir} --compare_BIC


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - -   Run some correlation checks to address uncertainty concerns   - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/reviewer_requested_checks.py ${dataset_dir} --correlation_checks

}