{
# run the python decoding script
# options: 
#   --subject ... subject ID to use for decoding
#   --dataset_dir ... path to dataset
#   --ttest ... runs mass univariate t-tests on decoding maps (ignores sub argument since ttest is on the group; defaults to false)
#   --binary ... runs binary decoding instead or probabilistic (defaults to false)
# 
python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/searchlight_decoding.py --subject ${sub} --dataset_dir /Shared/lss_kahwang_hpc/data/FPNHIU/ --binary


#change permissions
cd ${dataset_dir}Decoding/sub-${sub}/
chmod 775 *
}