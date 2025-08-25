{
singularity run -B /Shared:/Shared/ /Shared/lss_kahwang_hpc/opt/bin/heudiconv_latest.sif \
-d ${dataset_dir}Raw/{subject}/SCANS/*/DICOM/*.dcm \
-o ${dataset_dir}BIDS \
-b \
-f /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/heuristics/quantum.py -s ${sub} -c dcm2niix --overwrite
python3 /Shared/lss_kahwang_hpc/scripts/fpnhiu/py_scripts/heudiconv_post.py ${sub} ${dataset_dir}

cd ${dataset_dir}BIDS/sub-${sub}/
chmod 775 *
}