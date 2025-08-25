{
# run the python network hubs script ... calls a function from the human connectome toolbox
# set up to be run on thalamege not argon...

python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/network_hubs.py


#change permissions
cd ${dataset_dir}3dDeconvolve/sub-${sub}/
chmod 775 *
}