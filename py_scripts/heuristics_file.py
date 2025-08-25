import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    t1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')
    fm_pe0 = create_key('sub-{subject}/fmap/sub-{subject}_dir-pe0_run-{item:03d}_epi')
    fm_pe1 = create_key('sub-{subject}/fmap/sub-{subject}_dir-pe1_run-{item:03d}_epi')
    task_Quantum = create_key('sub-{subject}/func/sub-{subject}_task-Quantum_run-{item:03d}_bold')


    info = {t1w: [], fm_pe0: [], fm_pe1:[], task_Quantum:[] }

    for idx, seq in enumerate(seqinfo):
        '''
        seq contains the following fields
        * total_files_till_now
        * example_dcm_file
        * series_number
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        '''

        x,y,z,n_vol,protocol,dcm_dir, TR, TE, image_type, series, total = (seq[6], seq[7], seq[8], seq[9], seq[12], seq[3], seq[10], seq[11], seq[19], seq[18], seq[0] )
        #x,y,z,n_vol,protocol,dcm_dir, TR, TE, image_type, series, total = (seq[6], seq[7], seq[8], seq[9], seq[12], seq[3], seq[10], seq[11], seq[19], seq[18], seq[0] )

        # t1_mprage --> T1w
        if (series == 'T1 SAG'):
            info[t1w] = [seq[2]]


        if (z == 14400 ) and (TR == 2.039) :
            info[task_Quantum].append({'item': seq[2]})


        if (series == 'EPI FM pepolar=0' ) : #a hack for 7002, where forgot to switch sequence
            info[fm_pe0].append({'item': seq[2]})

        if (series == 'EPI FM pepolar=1' ) : #a hack for 7002, where forgot to switch sequence
            info[fm_pe1].append({'item': seq[2]})


    print(info[t1w])
    print(info[task_Quantum])
    print(info[fm_pe0])
    print(info[fm_pe1])
    return info