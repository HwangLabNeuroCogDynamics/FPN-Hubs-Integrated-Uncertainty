{
# run the python computational model code
# options: 
#   --mpe_model ... runs probabilistic model
#   --control_model ... runs control models
python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/model_scripts/run_compmodels.py ${sub} ${dataset_dir} --mpe_model --control_model

# modify permissions
cd ${dataset_dir}model_data/
chmod 775 *
}