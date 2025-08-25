# FPN-Hubs-Integrated-Uncertainty
Code repository for manuscript titled "Frontoparietal Hubs Leverage Probabilistic Representations and Integrated Uncertainty to Guide Cognitive Flexibility"

The scripts assume the following folder structure

outermost_folder
 |
 |-- data
 |     |-- FPNHIU
 |     |     |-- 3dDeconvolve
 |     |     |-- 3dMEMA
 |     |     |-- BIDS
 |     |     |-- CSVs
 |     |     |-- Decoding
 |     |     |-- FC
 |     |     |-- fmriprep
 |     |     |-- freesurfer
 |     |     |-- Hubs
 |     |     |-- job_logs
 |     |     |-- model_data
 |     |     |-- mriqc
 |     |     |-- work
 |
 |
 |-- opt
 |    |-- afni
 |    |    | afni.sif
 |    |-- fmriprep
 |    |    | fmriprep23.3.0.simg
 |    |-- mriqc
 |    |    | mriqc.simg
 |
 |
 |-- scripts
 |      |-- fpnhiu
 |      |     |-- model_scripts
 |      |     |-- mri_scripts
 |      |     |       |-- submission_templates
 |      |     |       |-- submitted_scripts
 |      |     |-- plotting_env
 |      |     |-- py_scripts
 |      |     | pipeline.sh
 |      |     | plotting_notebook.ipynb

To re-generate results from scratch, you will need to download BIDS-formatted subject data to the BIDS folder and behavioral data files (.csv format) to the CSVs folder. All other folders will be filled in as scripts are run

The pipeline.sh script indicates which scripts to run and in what order to run them. This script is set up to submit jobs to a HCP (at uiowa we use the argon hcp). Jobs are set up to run in parallel--each subject is run separately

