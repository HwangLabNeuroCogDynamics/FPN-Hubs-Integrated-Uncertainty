#!/bin/bash
# Script to submit jobs

# ALL SUBJECTS
#sub_list="10008 10024 10118 10218 10263 10275 10280 10281 10282 10283 10296 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10318 10319 10320 10321 10322 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10351 10353 10355 10356 10357 10358 10359 10360 10374"
#           1     2     3     4     5     6     7     8     9    10     11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34   35    36    37     38    39    40    41    42    43    44    45    46    47    48    49
# ONLY USABLE SUBJECTS
sub_list="10008 10024 10263 10280 10281 10283 10303 10305 10306 10309 10310 10311 10313 10314 10315 10316 10320 10323 10324 10326 10327 10332 10334 10336 10340 10341 10344 10346 10348 10349 10350 10353 10355 10356 10357 10359 10360 10374"
#           1     2     3     4     5     6     7     8     9    10     11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34   35    36    37     38    
subtxt="subjects=($sub_list)"
arrtxt="#$ -t 9-9" # which subjects to run... max index 38 .. 14, 15, 20, 23, 29, 
emailtxt="#$ -M stephanie-leach@uiowa.edu" # email for hpc notifications

cur_datetime=$(date +%F_%H-%M-%S)
# define number of cores for different scripts and job names
h_threads="#$ -pe smp 5" # cores for heudiconv, REML, 3dFWHMX, 3dClustsim
mf_threads="#$ -pe smp 15" # cores for mriqc and fmriprep
c_threads="#$ -pe smp 15" # cores for computational model code, Decoding
d_threads="#$ -pe smp 15" # cores for 3dDeconvolve
h_name="#$ -N run_heudiconv"
m_name="#$ -N run_mriqc"
f_name="#$ -N run_fmriprep"
d_name="#$ -N run_3dDeconvolve"
p_name="#$ -N prep_3dDeconvolve"
c_name="#$ -N run_comp_mods"
r_name="#$ -N run_REML"
a_name="#$ -N run_3dmema"
l_name="#$ -N run_LSS"
de_name="#$ -N run_Decoding"
fw_name="#$ -N run_3dFWHMX"
cs_name="#$ -N run_3dClustSim"

project_dir="/Shared/lss_kahwang_hpc/scripts/fpnhiu/"

# # ---------------------------------------------------------------------------------------------------------------------------------------
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # - - - - - - - - - - - - - - - - - - - - - - -         Part 1 - Initial Preprocessing        - - - - - - - - - - - - - - - - - - - - - -
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # ---------------------------------------------------------------------------------------------------------------------------------------

# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # - - - - - - - - - - - -     Computational Model     - - - - - - - - - - - - 
# # # Run script to convert subject behavioral data to model inputs and run models to get model outputs
# # # Concatenate header and compmodels and then run (python scripts automatically check if subject models already exist so no need to do that here)
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/genmodel_base.sh >> ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# sed -i "11i ${c_threads}" ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# sed -i "5i ${c_name}" ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# qsub ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh
# #qsub -hold_jid 198572 ${project_dir}mri_scripts/submitted_scripts/genmodel_$cur_datetime.sh

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # - - - - - - - - - - - -     heudiconv     - - - - - - - - - - - - 
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/heudiconv_base.sh >> ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# sed -i "5i ${h_name}" ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh
# h_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/heudiconv_$cur_datetime.sh`
# h_jid=$(echo ${h_jobid}| cut -d'.' -f -1) 
# echo $h_jid

# #h_jid=0 # uncomment this if you comment out the "heudiconv" code above
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # - - - - - - - - - - - -     mriqc     - - - - - - - - - - - - 
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/mriqc_base.sh >> ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# sed -i "11i ${mf_threads}" ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# sed -i "5i ${m_name}" ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# if [ $h_jid -lt 1 ]; then
#     echo skipped heudiconv so no need to hold job
#     qsub ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# else
#     qsub -hold_jid $h_jid ${project_dir}mri_scripts/submitted_scripts/mriqc_$cur_datetime.sh
# fi

# #h_jid=0 # uncomment this if you comment out the "heudiconv" code above
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # - - - - - - - - - - - -     fmriprep     - - - - - - - - - - - - 
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/fmriprep2_base.sh >> ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# sed -i "11i ${mf_threads}" ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# sed -i "5i ${f_name}" ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh
# if [ $h_jid -lt 1 ]; then
#     echo skipped heudiconv so no need to hold job
#     f_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh`
#     f_jid=$(echo ${f_jobid}| cut -d'.' -f -1)
#     echo $f_jid
# else
#     f_jobid=`qsub -terse -hold_jid $h_jid ${project_dir}mri_scripts/submitted_scripts/fmriprep_$cur_datetime.sh`
#     f_jid=$(echo ${f_jobid}| cut -d'.' -f -1)
#     echo $f_jid
# fi

# # # ---- pause script until fmriprep is mostly finished
# # echo waiting to submit prep and 3dDeconvolve script... will submit once prep script finishes since 3dDeconvolve uses a lot of cores 
# # until [ -f /Shared/lss_kahwang_hpc/data/TRIIMS/fmriprep/sub-${sub1}.html ]; do # sub-10310_task-Quantum_run-5_desc-confounds_regressors.tsv
# #     # do nothing... make script sit
# #     echo fmri script still running... will check again after 5 minutes
# #     sleep 300
# # done




# ---------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - -         Part 2 - Post-preprocessing           - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---------------------------------------------------------------------------------------------------------------------------------------

# f_jid=0 # uncomment this if you comment out the "fmriprep" code above
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # - - - - - - - - - - - -    3dDeconvolve    - - - - - - - - - - - - 
# echo prep script has finished running... submitting 3dDeconvolve now
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/Deconvolve_base.sh >> ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# sed -i "11i ${d_threads}" ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# sed -i "5i ${d_name}" ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# if [ $f_jid -lt 1 ]; then
#     echo skipped model generation and prep so no need to hold job
#     qsub ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# else
#     echo holding job until model generation and prep finishes
#     qsub -hold_jid $f_jid ${project_dir}mri_scripts/submitted_scripts/Deconvolve_$cur_datetime.sh
# fi

# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # - - - - - - - - - - - -    Run REML script     - - - - - - - - - - - - 
# # # # Concatenate header and REML and then run
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/REML_base.sh >> ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# sed -i "5i ${r_name}" ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh
# r_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh`
# r_jid=$(echo ${r_jobid}| cut -d'.' -f -1)
# echo $r_jid
# # qsub -hold_jid $f_jid ${project_dir}mri_scripts/submitted_scripts/REML_$cur_datetime.sh


# # #r_jid=0 # uncomment this if you comment out the "REML" code above
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # - - - - - - - - - - - -    Run 3dmema script    - - - - - - - - - - - - 
# # # ISSUES WITH 3DMEMA ON ARGON !!!! HAVE TO RUN ON THALAMEGE
# # # Concatenate header and 3dmema and then run
# # cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/3dmema_base.sh >> ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # sed -i "11i ${d_threads}" ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # sed -i "5i ${a_name}" ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # chmod 775 ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # if [ $r_jid -lt 1 ]; then
# #     echo skipped REML so no need to hold job
# #     qsub ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # else
# #     echo holding job until REML finishes
# #     qsub -hold_jid $r_jid ${project_dir}mri_scripts/submitted_scripts/3dmema_$cur_datetime.sh
# # fi

# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # - - - - - - - - - - - -    MISC Scripts for Post-3dMEMA    - - - - - - - - - - - - 
# # need to run 3dFWHMX.sh to get acf inputs for 3dClustSim, which outputs a txt file with min sig. cluster size
post_3dmema="post_3dmema=1"
# # # # - - - - submit 3dFWHMX.sh
# # submitting each 3dFWHMX as a separate job to speed things up
# for ii in {1..3}; do
# cat ${project_dir}mri_scripts/submission_templates/header_info_reduced.sh ${project_dir}mri_scripts/submission_templates/3dFWHMX_base.sh >> ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# sed -i "38i ${post_3dmema}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# sed -i "38i numtorun=${ii}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# sed -i "5i ${fw_name}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh
# fw_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_${ii}_$cur_datetime.sh`
# fw_jid=$(echo ${fw_jobid}| cut -d'.' -f -1)
# echo $fw_jid
# done
# # # !!! make sure to enter acf values calculated above before you run the code below
# # # - - - - submit 3dClustSim.sh
# cat ${project_dir}mri_scripts/submission_templates/header_info_reduced.sh ${project_dir}mri_scripts/submission_templates/3dClustSim_base.sh >> ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "38i ${post_3dmema}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "5i ${cs_name}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# cs_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh`
# cs_jid=$(echo ${cs_jobid}| cut -d'.' -f -1)
# echo $cs_jid


# f_jid=0 # uncomment this if you comment out the "fmriprep" code above
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # - - - - - - - - - - - -    3dLSS     - - - - - - - - - - - - 
# echo prep script has finished running... submitting LSS now
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/LSS_base.sh >> ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# sed -i "5i ${l_name}" ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh
# if [ $f_jid -lt 1 ]; then
#     echo skipped model generation and prep so no need to hold job
#     l_jobid=`qsub ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh`
# else
#     echo holding job until model generation and prep finishes
#     l_jobid=`qsub -hold_jid $f_jid ${project_dir}mri_scripts/submitted_scripts/LSS_$cur_datetime.sh`
# fi
# l_jid=$(echo ${l_jobid} | cut -d'.' -f -1 | grep -oP "\d{2,8}")
# echo l_jid is $l_jid

l_jid=0 # uncomment this if you comment out the "3dLSS" code above
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # - - - - - - - - - - - -    DECODING     - - - - - - - - - - - - 
echo prep script has finished running... submitting decoding now
cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/Decoding_base.sh >> ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "34d" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "11i ${c_threads}" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
sed -i "5i ${de_name}" ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
chmod 775 ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
if [ $l_jid -lt 1 ]; then
    echo skipped LSS so no need to hold job
    qsub ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
else
    echo holding job ... waiting on LSS
    qsub -hold_jid $l_jid ${project_dir}mri_scripts/submitted_scripts/Decoding_$cur_datetime.sh
fi

# # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # # - - - - - - - - - - - -    MISC Scripts for Post-DECODING    - - - - - - - - - - - - 
# # # # need to run 3dFWHMX.sh to get acf inputs for 3dClustSim, which outputs a txt file with min sig. cluster size
post_3dmema="post_3dmema=0"
# # # - - - - submit 3dFWHMX.sh
# cat ${project_dir}mri_scripts/submission_templates/header_info_reduced.sh ${project_dir}mri_scripts/submission_templates/3dFWHMX_base.sh >> ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# sed -i "38i ${post_3dmema}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# sed -i "5i ${fw_name}" ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh
# fw_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/3dFWHMX_$cur_datetime.sh`
# fw_jid=$(echo ${fw_jobid}| cut -d'.' -f -1)
# echo $fw_jid
# # # !!! make sure to enter acf values calculated above before you run the code below  3476796
# # # - - - - submit 3dClustSim.sh
# cat ${project_dir}mri_scripts/submission_templates/header_info_reduced.sh ${project_dir}mri_scripts/submission_templates/3dClustSim_base.sh >> ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "38i ${post_3dmema}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "11i ${h_threads}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# sed -i "5i ${cs_name}" ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh
# cs_jobid=`qsub -terse ${project_dir}mri_scripts/submitted_scripts/3dClustSim_$cur_datetime.sh`
# cs_jid=$(echo ${cs_jobid}| cut -d'.' -f -1)
# echo $cs_jid



# # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # # - - - - - - - - - - - -    Network Hubs     - - - - - - - - - - - - 
# #l_jid=0 # uncomment this if you comment out the "3dLSS" code above
# echo 3dLSS script has finished running... submitting Network Hubs now
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/NetworkHubs_base.sh >> ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "34d" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "11i ${c_threads}" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# sed -i "5i ${de_name}" ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# if [ $l_jid -lt 1 ]; then
#     echo skipped LSS so no need to hold job
#     qsub ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# else
#     echo holding job ... waiting on LSS
#     qsub -hold_jid $l_jid ${project_dir}mri_scripts/submitted_scripts/NetworkHubs_$cur_datetime.sh
# fi


# # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
# # # # - - - - - - - - - - - -    Functional Connectivity     - - - - - - - - - - - - 
# #l_jid=0 # uncomment this if you comment out the "3dLSS" code above
# echo 3dLSS script has finished running... submitting Functional Connectivity now
# cat ${project_dir}mri_scripts/submission_templates/header_info.sh ${project_dir}mri_scripts/submission_templates/FuncConnectivity_base.sh >> ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "34d" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "23i ${arrtxt}" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "21i ${subtxt}" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "14i ${emailtxt}" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "11i ${c_threads}" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# sed -i "5i ${de_name}" ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# chmod 775 ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# if [ $l_jid -lt 1 ]; then
#     echo skipped LSS so no need to hold job
#     qsub ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# else
#     echo holding job ... waiting on LSS
#     qsub -hold_jid $l_jid ${project_dir}mri_scripts/submitted_scripts/FuncConnectivity_$cur_datetime.sh
# fi
