{
singularity run --cleanenv -B /Shared:/Shared/ /Shared/lss_kahwang_hpc/opt/mriqc/mriqc.simg \
${dataset_dir}BIDS/ \
${dataset_dir}mriqc/ \
participant --participant_label ${sub} --n_procs 13 --ants-nthreads 13 \
-w ${dataset_dir}work/
# modify permissions
cd ${dataset_dir}mriqc/
chmod 775 *
}