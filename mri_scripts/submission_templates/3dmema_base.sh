dataset_dir=/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/
deconvolve_dir="/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/3dDeconvolve"
subjects=(10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 \
        10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 \
        10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 \
        10350 10353 10355 10356 10357 10359 10360 10374)  
num_of_subs=38
# 10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10353 10355 10356 10357 10359 10360 10374
#   1     2     3     4     5     6     7     8     9    10     11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34   35    36    37     38   

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Run 3dDeconvolve WITH parametric modulation - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# -- cue_probe_fb__color__SPMGmodel_stats -- #
conditions=(cue-zEntropy probe-zEntropy feedback-zEntropy cue-zState probe-zState feedback-zState cue-zColor probe-zColor feedback-zColor)
coef=([58] [68] [74] [6] [16] [22] [32] [42] [48])
tstat=([59] [69] [75] [7] [17] [23] [33] [43] [49])
for c in ${!conditions[@]}; do
echo ${conditions[$c]}
#singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz \
-jobs 12 \
-set ${conditions[$c]} \
sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
-cio \
-missing_data 0 \
-model_outliers
done


# -- cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel_stats -- #
conditions=(cue-statePE probe-statePE feedback-statePE cue-colorPE probe-colorPE feedback-colorPE cue-taskPE probe-taskPE feedback-taskPE)
coef=([6] [16] [22] [32] [42] [48] [58] [68] [74])
tstat=([7] [17] [23] [33] [43] [49] [59] [69] [75])
for c in ${!conditions[@]}; do
echo ${conditions[$c]}
#rm ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz
#singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz \
-jobs 12 \
-set ${conditions[$c]} \
sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
-cio \
-missing_data 0 \
-model_outliers
done


# -- cue_probe_fb__stateD__SPMGmodel_stats -- #
conditions=(cue-stateD probe-stateD feedback-stateD)
coef=([6] [16] [22])
tstat=([7] [17] [23])
for c in ${!conditions[@]}; do
echo ${conditions[$c]}
#rm ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz
#singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz \
-jobs 12 \
-set ${conditions[$c]} \
sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__stateD_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
-cio \
-missing_data 0 \
-model_outliers
done



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

# # -- cue_probe_fb__color__SPMGmodel_stats -- #
# conditions=(cue-color probe-color feedback-color)
# #coef=([6] [16] [26]) # need to change to correct numbers based on REML
# #tstat=([7] [17] [27]) # need to change to correct numbers based on REML
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__color_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__color_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done


# # -- cue_probe_fb__state__SPMGmodel_stats -- #
# conditions=(cue-state probe-state feedback-state)
# #coef=([6] [16] [26]) # need to change to correct numbers based on REML
# #tstat=([7] [17] [27]) # need to change to correct numbers based on REML
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__state_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__state_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done

# # -- cue_probe_fb__task__SPMGmodel_stats -- #
# conditions=(cue-task probe-task feedback-task)
# #coef=([6] [16] [26]) # need to change to correct numbers based on REML
# #tstat=([7] [17] [27]) # need to change to correct numbers based on REML
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__task_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__task_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done

# # -- cue_probe_entropy_SPMGmodel_stats -- #
# conditions=(cue-entropy probe-entropy feedback-entropy)
# #coef=([6] [16] [26]) # need to change to correct numbers based on REML
# #tstat=([7] [17] [27]) # need to change to correct numbers based on REML
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__entropy_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__entropy_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done

# # -- cue_probe_fb__statePE__SPMGmodel_stats -- #
# conditions=(cue-statePE probe-statePE feedback-statePE)
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# rm ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__statePE_SPMGmodel.nii.gz
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__statePE_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__statePE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done

# # -- cue_probe_fb__taskPE__SPMGmodel_stats -- #
# conditions=(cue-taskPE probe-taskPE feedback-taskPE)
# coef=([6] [16] [22]) # need to change to correct numbers based on REML
# tstat=([7] [17] [23]) # need to change to correct numbers based on REML
# #conditions=(feedback-taskPE)
# #coef=([22]) # need to change to correct numbers based on REML
# #tstat=([23]) # need to change to correct numbers based on REML
# for c in ${!conditions[@]}; do
# echo ${conditions[$c]}
# rm ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__taskPE_SPMGmodel.nii.gz
# #singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
# 3dMEMA -prefix ${dataset_dir}3dMEMA/${conditions[$c]}__${num_of_subs}subjs_cue_probe_fb__taskPE_SPMGmodel.nii.gz \
# -jobs 12 \
# -set ${conditions[$c]} \
# sub-${subjects[0]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[0]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[1]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[1]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[2]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[2]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[3]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[3]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[4]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[4]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[5]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[5]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[6]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[6]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[7]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[7]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[8]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[8]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[9]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[9]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[10]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[10]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[11]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[11]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[12]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[12]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[13]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[13]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[14]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[14]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[15]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[15]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[16]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[16]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[17]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[17]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[18]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[18]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[19]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[19]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[20]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[20]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[21]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[21]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[22]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[22]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[23]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[23]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[24]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[24]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[25]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[25]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[26]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[26]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[27]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[27]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[28]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[28]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[29]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[29]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[30]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[30]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[31]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[31]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[32]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[32]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[33]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[33]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[34]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[34]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[35]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[35]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[36]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[36]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# sub-${subjects[37]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${coef[$c]} ${deconvolve_dir}/sub-${subjects[37]}/cue_probe_fb__taskPE_SPMGmodel_stats_REML.nii.gz${tstat[$c]} \
# -cio \
# -missing_data 0 \
# -model_outliers
# done


#done
cd ${dataset_dir}3dMEMA/
chmod 775 *