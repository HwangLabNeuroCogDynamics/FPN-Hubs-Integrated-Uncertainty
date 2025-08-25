# list of usable subjects
subjects=(10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 \ 
        10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 \ 
        10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 \ 
        10350 10353 10355 10356 10357 10359 10360 10374)
num_subjects="38"
# we get -acf arguments from 3dFWHMx !!!


## ---------------------- ##
## ---- 3dDeconvolve ---- ##
## ---------------------- ##
if [ $post_3dmema -gt 0 ]; then
    dataset_dir=/Shared/lss_kahwang_hpc/data/FPNHIU/3dMEMA/
    conditions=(cue probe feedback)
    for con in ${conditions[@]}
    do
    # -- STATE COLOR ENTROPY -- cue_probe_fb__state_SPMGmodel_stats 
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zState__SPMGmodel__cluststim.txt
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zColor__SPMGmodel__cluststim.txt
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zEntropy__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.671025 1.39137 3.49768 -mask ${dataset_dir}${con}-zState__${num_subjects}subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zState__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.671025 1.39137 3.49768 -mask ${dataset_dir}${con}-zColor__${num_subjects}subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zColor__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.671025 1.39137 3.49768 -mask ${dataset_dir}${con}-zEntropy__${num_subjects}subjs_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zEntropy__SPMGmodel__cluststim.txt
    
    # -- STATE PE -- cue_probe_fb__statePE_SPMGmodel_stats
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zStatePE__SPMGmodel__cluststim.txt
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zColorPE__SPMGmodel__cluststim.txt
    # rm ${dataset_dir}clustsim/${num_subjects}subs_${con}__cue_probe_fb__zTaskPE__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.670549 1.39125 3.48985 -mask ${dataset_dir}${con}-statePE__${num_subjects}subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zStatePE__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.670549 1.39125 3.48985 -mask ${dataset_dir}${con}-colorPE__${num_subjects}subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zColorPE__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.670549 1.39125 3.48985 -mask ${dataset_dir}${con}-taskPE__${num_subjects}subjs_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__zTaskPE__SPMGmodel__cluststim.txt
    
    # -- STATE D -- cue_probe_fb__stateD_SPMGmodel_stats
    # rm ${dataset_dir}${num_subjects}subs_${con}__cue_probe_fb__stateD__SPMGmodel__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf 0.671434 1.39183 3.49855 -mask ${dataset_dir}${con}-stateD__${num_subjects}subjs_cue_probe_fb__stateD_SPMGmodel.nii.gz >> ${dataset_dir}clustsim/new/${num_subjects}subs_${con}__cue_probe_fb__stateD__SPMGmodel__cluststim.txt
    
    done
fi


## ---------------------- ##
## ----     3dLSS    ---- ##
## ---------------------- ##
num_subjects="38"
if [ $post_3dmema -lt 1 ]; then
    dataset_dir=/Shared/lss_kahwang_hpc/data/FPNHIU/Decoding/
    loopidxnum=(0 1 2)
    conditions=(cue probe fb)
    acf1=(0.681074 0.679676 0.68174) # order is (cue_acf1 probe_acf1 fb_acf1)
    acf2=(1.38175 1.37978 1.38105)
    acf3=(3.10683 3.09502 3.07534)
    for idx in ${loopidxnum[@]}
    do
    # ---- r value from probabilistic decoding
    # # -- color
    # #rm ${dataset_dir}clustsim/${num_subjects}subjs__color_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    # 3dClustSim -acf ${acf1[${idx}]} ${acf2[${idx}]} ${acf3[${idx}]} -mask ${dataset_dir}GroupStats/GroupAnalysis_${num_subjects}subjs__color_${conditions[${idx}]}__r__avg_coeff.nii >> ${dataset_dir}clustsim/${num_subjects}subjs__color_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # # -- state
    # #rm ${dataset_dir}clustsim/${num_subjects}subjs__state_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    # 3dClustSim -acf ${acf1[${idx}]} ${acf2[${idx}]} ${acf3[${idx}]} -mask ${dataset_dir}GroupStats/GroupAnalysis_${num_subjects}subjs__state_${conditions[${idx}]}__r__avg_coeff.nii >> ${dataset_dir}clustsim/${num_subjects}subjs__state_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # # -- task
    # #rm ${dataset_dir}clustsim/${num_subjects}subjs__task_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    # 3dClustSim -acf ${acf1[${idx}]} ${acf2[${idx}]} ${acf3[${idx}]} -mask ${dataset_dir}GroupStats/GroupAnalysis_${num_subjects}subjs__task_${conditions[${idx}]}__r__avg_coeff.nii >> ${dataset_dir}clustsim/${num_subjects}subjs__task_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # # -- entropy
    # #rm ${dataset_dir}clustsim/${num_subjects}subjs__entropy_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    # 3dClustSim -acf ${acf1[${idx}]} ${acf2[${idx}]} ${acf3[${idx}]} -mask ${dataset_dir}GroupStats/GroupAnalysis_${num_subjects}subjs__entropy_${conditions[${idx}]}__r__avg_coeff.nii >> ${dataset_dir}clustsim/${num_subjects}subjs__entropy_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    # -- jointP
    #rm ${dataset_dir}clustsim/${num_subjects}subjs__${conditions[${idx}]}_jointP__r__avg_coeff__cluststim.txt
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dClustSim -acf ${acf1[${idx}]} ${acf2[${idx}]} ${acf3[${idx}]} -mask ${dataset_dir}GroupStats/GroupAnalysis_${num_subjects}subjs__jointP_${conditions[${idx}]}__r__avg_coeff.nii >> ${dataset_dir}clustsim/${num_subjects}subjs__jointP_${conditions[${idx}]}__r__avg_coeff__cluststim.txt
    done
    
fi
