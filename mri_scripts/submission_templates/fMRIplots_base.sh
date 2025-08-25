dataset_dir=/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/
{
# first we need to generate 3D files (original files are 4D) for the 3dMEMA outputs ... decoding is already 3D
# conditions=(cue-zEntropy probe-zEntropy feedback-zEntropy   cue-zState probe-zState feedback-zState   cue-zColor probe-zColor feedback-zColor)
# coef=([58] [68] [74]   [6] [16] [22]    [32] [42] [48])
# tstat=([59] [69] [75]   [7] [17] [23]    [33] [43] [49])
# for sub in 10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10353 10355 10356 10357 10359 10360 10374
# do
#     # State - cue
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[6]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[7]
#     # State - probe
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[16]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[17]
#     # State - feedback
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[22]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[23]

#     # Color - cue
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[32]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[33]
#     # Color - probe
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[42]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[43]
#     # Color - feedback
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[48]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[49]

#     # Entropy - cue
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[58]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[59]
#     # Entropy - probe
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[68]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[69]
#     # Entropy - feedback
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[78]
#     3dTcat -prefix ${dataset_dir}3dDeconvolve/sub-${sub}/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dDeconvolve/sub-${sub}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz[79]
# done

# Color - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# Color - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# Color - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zColor_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zColor_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-zColor__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]

# State - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# State - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# State - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zState_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zState_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-zState__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]

# Entropy - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# Entropy - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]
# Entropy - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zEntropy_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zEntropy_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-zEntropy__38subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz[1]


# ColorPE - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zColorPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zColorPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# ColorPE - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zColorPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zColorPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# ColorPE - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zColorPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zColorPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-colorPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]

# StatePE - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zStatePE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zStatePE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# StatePE - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zStatePE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zStatePE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# StatePE - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zStatePE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zStatePE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-statePE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]

# TaskPE - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zTaskPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# TaskPE - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zTaskPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]
# TaskPE - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zTaskPE_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-taskPE__38subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz[1]


# StateD - cue
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__StateD_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/cue-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/cue__StateD_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/cue-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[1]
# StateD - probe
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__StateD_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/probe-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/probe__StateD_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/probe-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[1]
# StateD - feedback
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__StateD_SPMGmodel_stats_REML__beta.nii.gz ${dataset_dir}3dMEMA/feedback-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[0]
3dTcat -prefix ${dataset_dir}3dMEMA/nii3D/feedback__StateD_SPMGmodel_stats_REML__tval.nii.gz ${dataset_dir}3dMEMA/feedback-stateD__38subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz[1]


# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # # State - cue
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zState_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zState__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zState_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zState__R.shape.gii -trilinear
# # # Color - cue
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zColor__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zColor_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zColor__R.shape.gii -trilinear
# # # Entropy - cue
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zEntropy__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__zEntropy__R.shape.gii -trilinear

# # # ColorPE - feedback
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zColorPE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zColorPE__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zColorPE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zColorPE__R.shape.gii -trilinear
# # # StatePE - feedback
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zStatePE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zStatePE__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zStatePE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zStatePE__R.shape.gii -trilinear
# # # TaskPE - feedback
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zTaskPE__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/feedback__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz  \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/feedback__zTaskPE__R.shape.gii -trilinear

# # # StateD - cue
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__StateD_SPMGmodel_stats_REML__tval.nii.gz \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__StateD__L.shape.gii -trilinear
# wb_command -volume-to-surface-mapping ${dataset_dir}3dMEMA/nii3D/cue__StateD_SPMGmodel_stats_REML__tval.nii.gz \
# /data/backed_up/shared/wb_files/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
# ${dataset_dir}3dMEMA/wb_gii/cue__StateD__R.shape.gii -trilinear
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#python3 /mnt/nfs/lss/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/fMRI_plots.py
}