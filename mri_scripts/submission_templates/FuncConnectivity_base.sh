{
# run the python functional connectivity script
# 
python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/functional_connectivity.py ${dataset_dir}


#change permissions
cd ${dataset_dir}3dDeconvolve/sub-${sub}/
chmod 775 *
}