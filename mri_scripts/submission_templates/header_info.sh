#!/bin/bash
# SGE

## Name of the job


## queue to run job on ... UI-GPU or SEASHORE
#$ -q SEASHORE

## call for parallel environment (-pe) and shared memory multiprocessing (smp) and total number of cores to use (80)


## email address to notify about job submision 
## settings for emails: b = notify of job start, e = notify of job end, a = notify of job abort, s = notify of job suspension, n = no notifications
#$ -m bes

## I do not understand the variable below... commented out for now
##export OMP_NUM_THREADS=12

## variables and lists for code below
## array job arguments: '-t n[-m][:s]]' n=the lowest index number, m=the highest index number, and s=the step size
## THE CALL BELOW HAS TO BE CHANGED SO THAT "M" MATCHES THE NUMBER OF SUBJECTS IN YOUR LIST ABOVE

## where error and output logs should be saved to
#$ -e /Shared/lss_kahwang_hpc/data/FPNHIU/job_logs/
#$ -o /Shared/lss_kahwang_hpc/data/FPNHIU/job_logs/

dataset_dir=/Shared/lss_kahwang_hpc/data/FPNHIU/ # quotes are not needed unless the path has spaces
echo subjects: ${subjects[@]}
echo total_subjects=${#subjects[@]}
sub="${subjects[$SGE_TASK_ID-1]}"
echo ${sub}
## activate virtual environment so packages are the same every time
source /Shared/lss_kahwang_hpc/virtualenvs/base/bin/activate
# cd ${dataset_dir}
# chmod -R 775 *
# run code below
