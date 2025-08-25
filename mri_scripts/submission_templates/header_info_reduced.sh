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

## where error and output logs should be saved to
#$ -e /Shared/lss_kahwang_hpc/data/FPNHIU/job_logs/
#$ -o /Shared/lss_kahwang_hpc/data/FPNHIU/job_logs/

## activate virtual environment so packages are the same every time
source /Shared/lss_kahwang_hpc/virtualenvs/base/bin/activate
