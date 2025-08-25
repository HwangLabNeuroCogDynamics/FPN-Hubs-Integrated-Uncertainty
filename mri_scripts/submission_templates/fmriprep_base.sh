{
#rm -f /Shared/lss_kahwang_hpc/data/TRIIMS/fmriprep/sourcedata/freesurfer/sub-${sub}/scripts/IsRunning.lh
rm /Shared/lss_kahwang_hpc/data/TRIIMS/fmriprep/sourcedata/freesurfer/sub-${sub}/scripts/IsRunning.lh+rh
if [ -f /Shared/lss_kahwang_hpc/data/TRIIMS/BIDS/scans.json ]; then
echo scans json file exists and will be deleted so fmriprep bids validator works
rm -f /Shared/lss_kahwang_hpc/data/TRIIMS/BIDS/scans.json
fi


singularity run --cleanenv -B /Shared:/Shared/ /Shared/lss_kahwang_hpc/opt/fmriprep/fmriprep-23.2.0.simg \
${dataset_dir}BIDS/ \
${dataset_dir}fmriprep \
participant --participant_label ${sub} \
--bold2t1w-dof 12 \
--nthreads 15 \
--omp-nthreads 15 \
--fs-license-file /Shared/lss_kahwang_hpc/opt/fmriprep/fs_license.txt \
--fs-subjects-dir ${dataset_dir}/fmriprep/sourcedata/freesurfer/ \
-w ${dataset_dir}work/

# modify permissions
cd ${dataset_dir}fmriprep/
chmod 775 *
}