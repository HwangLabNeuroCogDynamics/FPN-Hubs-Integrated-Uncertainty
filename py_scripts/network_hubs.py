import numpy as np
import bct
import pandas as pd
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
from itertools import combinations
import nibabel as nib
from nibabel import cifti2
import os
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore
from nilearn import masking
import nilearn

def threshold(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Threshold a numpy matrix to obtain a certain "cost".
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix[np.isnan(matrix)] = 0.0
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		if mst == False:
			matrix[matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)] = 0.
		else:
			if test_matrix == True: t_m = matrix.copy()
			assert (np.tril(matrix,-1) == np.triu(matrix,1).transpose()).all()
			matrix = np.tril(matrix,-1)
			mst = minimum_spanning_tree(matrix*-1)*-1
			mst = mst.toarray()
			mst = mst.transpose() + mst
			matrix = matrix.transpose() + matrix
			if test_matrix == True: assert (matrix == t_m).all() == True
			matrix[(matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)) & (mst==0.0)] = 0.
	if binary == True:
		matrix[matrix>0] = 1
	if normalize == True:
		matrix = matrix/np.sum(matrix)
	return matrix

def write_graph_to_vol_yeo_template_nifti(graph_metric, fn, resolution=400):
	'''short hand to write vol based nifti file of the graph metrics
	assuming Cole 718 parcels, voxels in each parcel will be replaced with the graph metric'''

	#roi_df = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/Updated_CA_ROI_List.csv')
	#roi_df.loc[0:359,'KEYVALUE'] = np.arange(1,361)
	if resolution == 400:
		vol_template = nib.load('/home/kahwang/bsh/ROIs/Yeo425x17LiberalCombinedMNI.nii.gz')
		roisize = 425
	elif resolution == 900:
		vol_template = nib.load('/home/kahwang/bsh/ROIs/Schaefer900.nii.gz')
		roisize = 900
	elif resolution == 'voxelwise':
		vol_template = nib.load('/home/kahwang/bsh/ROIs/CA_4mm.nii.gz')
		#roisize = 18166
		parcel_mask = nilearn.image.new_img_like(vol_template, 1*(vol_template.get_data()>0), copy_header = True)
		vox_index = masking.apply_mask(nib.load('/data/backed_up/shared/ROIs/CA_4mm.nii.gz'), parcel_mask)
	else:
		print ('Error with template')
		return

	v_data = vol_template.get_data()
	graph_data = np.zeros((np.shape(v_data)))

	if resolution == 'voxelwise':
		for ix, i in enumerate(vox_index):
			graph_data[v_data == i] = graph_metric[ix]
	else:
		for i in np.arange(roisize):
			#key = roi_df['KEYVALUE'][i]
			graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn)


########################################################################
#voxel wise graph
########################################################################
roi='CA_4mm'
#MGH_avadj, _ = cal_dataset_adj(dset='MGH', roifile = roi)
#NKI_avadj, MGH_avadj = gen_groupave_adj(roi)
#fn = 'MGH_adj_%s' %roi
#np.save(fn, MGH_avadj)

parcel_template = '/data/backed_up/shared/ROIs/' + roi + '.nii.gz'
parcel_template = nib.load(parcel_template)

parcel_mask = nilearn.image.new_img_like(parcel_template, 1*(parcel_template.get_data()>0), copy_header = True)
CI = masking.apply_mask(nib.load('/data/backed_up/shared/ROIs/CA_4mm_network.nii.gz'), parcel_mask)

MGH_avadj = np.load('MGH_adj_CA_4mm.npy')
#NKI_avadj = np.load('NKI_adj_CA_4mm.npy')

max_cost = .15
min_cost = .01

#MATS = [MGH_avadj, NKI_avadj]
#dsets = ['MGH', 'NKI']

MATS = [MGH_avadj]
dsets = ['MGH']

# import thresholded matrix to BCT, import partition, run WMD/PC
PC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))
WMD = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))
EC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))
GC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))
SC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))
ST = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 18166))

for ix, matrix in enumerate(MATS):
    for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):

            tmp_matrix = threshold(matrix.copy(), cost)

            PC[i,:] = bct.participation_coef(tmp_matrix, CI)
            #WMD[i,:] = bct.module_degree_zscore(tmp_matrix,CI)
            #EC[i,:] = bct.eigenvector_centrality_und(tmp_matrix)
            #GC[i,:], _ = bct.gateway_coef_sign(tmp_matrix, CI)
            #SC[i,:] = bct.subgraph_centrality(tmp_matrix)
            #ST[i,:] = bct.strengths_und(tmp_matrix)

            mes = 'finished running cost:%s' %cost
            print(mes)

    # fn = 'images/Voxelwise_4mm_%s_WMD.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(np.nanmean(WMD,axis=0), fn, 'voxelwise')
    #
    # fn = 'images/Voxelwise_4mm_%s_WeightedDegree.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(np.nanmean(ST,axis=0), fn, 'voxelwise')
    #
    # #zscore version, eseentialy ranking across parcels/roi
    # fn = 'images/Voxelwise_4mm_%s_zWMD.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(WMD,axis=0)), fn, 'voxelwise')
    #
    # fn = 'images/Voxelwise_4mm_%s_zWeightedDegree.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(ST,axis=0)), fn, 'voxelwise')

    fn = 'images/Voxelwise_4mm_%s_PC.nii' %dsets[ix]
    write_graph_to_vol_yeo_template_nifti(np.nanmean(PC,axis=0), fn, 'voxelwise')

    #zscore version, eseentialy ranking across parcels/roi
    fn = 'images/Voxelwise_4mm_%s_zPC.nii' %dsets[ix]
    write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(PC,axis=0)), fn, 'voxelwise')

    # fn = 'images/Voxelwise_4mm_%s_EigenC.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(np.nanmean(EC,axis=0), fn, 'voxelwise')
    #
    # #zscore version, eseentialy ranking across parcels/roi
    # fn = 'images/Voxelwise_4mm_%s_zs_EigenC.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(EC,axis=0)), fn, 'voxelwise')

    # fn = 'images/Voxelwise_4mm_%s_GatewayCentC.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(np.nanmean(GC,axis=0), fn, 'voxelwise')
    #
    # #zscore version, eseentialy ranking across parcels/roi
    # fn = 'images/Voxelwise_4mm_%s_zs_GatewayCentC.nii' %dsets[ix]
    # write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(GC,axis=0)), fn, 'voxelwise')

    #fn = 'images/Voxelwise_4mm_%s_SubgraphCent.nii' %dsets[ix]
    #write_graph_to_vol_yeo_template_nifti(np.nanmean(SC,axis=0), fn, 'voxelwise')

    #zscore version, eseentialy ranking across parcels/roi
    #fn = 'images/Voxelwise_4mm_%s_zs_SubgraphCent.nii' %dsets[ix]
    #write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(SC,axis=0)), fn, 'voxelwise')