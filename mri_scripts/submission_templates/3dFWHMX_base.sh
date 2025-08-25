dataset_dir="/Shared/lss_kahwang_hpc/data/FPNHIU/"
deconvolve_dir="/Shared/lss_kahwang_hpc/data/FPNHIU/3dDeconvolve/"
# full usable subject list
subjects=(10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 \ 
        10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 \ 
        10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 \ 
        10350 10353 10355 10356 10357 10359 10360 10374)
num_subjects="38"


## ---------------------- ##
## ---- 3dDeconvolve ---- ##
## ---------------------- ##
if [ $post_3dmema -gt 0 ]; then
    # ---- parametric modulation models
    # -- zState zColor zEntropy
    if [ $numtorun -eq 1 ]
    then
    echo running cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats
    if [ -f ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt ]; then
        rm ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt
    fi
    for sub in ${subjects[@]}
    do
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dFWHMx -acf -input ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__zState-zColor-zEntropy_SPMGmodel_stats_errts_REML.nii.gz | tail -1 >> ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt
    done
    # output is 4 columns... just use first 3
    echo cue_probe_fb__zState-zColor-zEntropy__SPMGmodel_stats
    awk '{total+=$1} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$2} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$3} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zState-zColor-zEntropy__SPMGmodel__${num_subjects}subs.txt
    fi
    
    # -- statePE
    if [ $numtorun -eq 2 ]
    then
    echo running cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats
    if [ -f ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt ]; then
        rm ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt
    fi
    for sub in ${subjects[@]}
    do
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dFWHMx -acf -input ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__zStatePE-zColorPE-zTaskPE_SPMGmodel_stats_errts_REML.nii.gz | tail -1 >> ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt
    done
    # output is 4 columns... just use first 3
    echo cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel_stats
    awk '{total+=$1} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$2} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$3} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__zStatePE-zColorPE-zTaskPE__SPMGmodel__${num_subjects}subs.txt
    fi

    # -- stateD
    if [ $numtorun -eq 3 ]
    then
    echo running cue_probe_fb__stateD_SPMGmodel_stats
    if [ -f ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt ]; then
        rm ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt
    fi
    for sub in ${subjects[@]}
    do
    singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
    3dFWHMx -acf -input ${deconvolve_dir}sub-${sub}/sub-${sub}_cue_probe_fb__stateD_SPMGmodel_stats_errts_REML.nii.gz | tail -1 >> ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt
    done
    # output is 4 columns... just use first 3
    echo cue_probe_fb__stateD_SPMGmodel_stats
    awk '{total+=$1} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$2} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt
    awk '{total+=$3} END {print total/NR}' ${dataset_dir}3dMEMA/acf/acf_parameters__cue_probe_fb__stateD_SPMGmodel__${num_subjects}subs.txt
    fi

fi


## ---------------------------- ##
## ----       3dLSS        ---- ##
## ---------------------------- ##
cue_probe_fb=(cue probe fb) # list of time-locked 3dLSS files
if [ $post_3dmema -lt 1 ]; then
    echo running LSS errts
    if [ -f ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt ]; then
        rm ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt
    fi
    # loop through time-locked cue probe and fb
    for con in ${cue_probe_fb[@]}
        do
        echo RUNNING ${con}
        # loop through subject list and run 3dFWHMx 
        for sub in ${subjects[@]}
            do
            singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif \
            3dFWHMx -acf -input ${deconvolve_dir}sub-${sub}/${con}_LSS_errts_REML.nii.gz | tail -1 >> ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt
        done

        # output is 4 columns... just use first 3
        awk '{total+=$1} END {print total/NR}' ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt
        awk '{total+=$2} END {print total/NR}' ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt
        awk '{total+=$3} END {print total/NR}' ${dataset_dir}Decoding/acf/${con}_LSS_acf_parameters_${num_subjects}subjs.txt
    done
fi