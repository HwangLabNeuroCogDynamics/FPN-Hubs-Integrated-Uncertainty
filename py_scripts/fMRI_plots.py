import pandas as pd
import numpy as np
import numpy.matlib as mb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import os
import sys
import re
import argparse
import nilearn
from nilearn import datasets, plotting, surface
from nilearn.image import resample_img
from nilearn import masking
from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.glm import threshold_stats_img
#from nilearn.datasets import load_fsaverage_data
import nibabel as nib

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode subject",
        usage="[Subject] [dataset_dir] [OPTIONS] ... ",
    )
    parser.add_argument("dataset_dir", help="path to project folder i.e., /mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/")
    parser.add_argument("--networks17_indiv_plots", help="plot each of the 17 networks individually",
                        default=False, action="store_true")
    parser.add_argument("--networks17_1plot", help="plot all 17 networks in one single plot",
                        default=False, action="store_true")
    parser.add_argument("--networks17_subgroup_plots", help="plot each of the sub-groups of the 17 networks individually",
                        default=False, action="store_true")
    parser.add_argument("--networks17_bar_plots", help="plot each of the sub-groups of the 17 networks individually",
                        default=False, action="store_true")
    parser.add_argument("--PM_stats_plots", help="plot each of the sub-groups of the 17 networks individually",
                        default=False, action="store_true")
    parser.add_argument("--D_stats_plots", help="plot each of the sub-groups of the 17 networks individually",
                        default=False, action="store_true")
    return parser

parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
dataset_dir = args.dataset_dir


def create_nii(stats_mat, cortical_mask):
    cortical_masker = NiftiMasker(cortical_mask)
    cortical_masker.fit()
    stat_nii = cortical_masker.inverse_transform(stats_mat)
    return stat_nii

#dataset_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/FPNHIU/"
model_data_output = os.path.join(dataset_dir,"model_data")
# thresh_img = image.threshold_img(img, cluster_threshold=9, threshold=0.55)

mask_dir = "/mnt/nfs/lss/lss_kahwang_hpc/ROIs/"
#mask = nib.load(mask_dir + "Cortical_Mask_task-Quantum.nii.gz")
cortical_mask = nib.load(mask_dir + "CorticalMask_RSA_task-Quantum.nii.gz")
cortical_mask_data = cortical_mask.get_fdata()

# --------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - -   Set up Some Variables for Plots   - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - -
#Load cortical surface mesh
fsaverage = datasets.fetch_surf_fsaverage()
#load atlas
atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2) # fetch_atlas_yeo_2011()
atlas = atlas_data.maps
atlas_labels = atlas_data.labels
networks = []
for ii in range(len(atlas_labels)):
    networks.append(atlas_labels[ii].astype("str").split("_")[2])

#project atlas onto fsaverage
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
textures=[]
textures.append(surface.vol_to_surf(atlas, fsaverage['pial_left'], inner_mesh=fsaverage['white_left'], interpolation='nearest', n_samples=1, radius=0.0))
textures.append(surface.vol_to_surf(atlas, fsaverage['pial_left'], inner_mesh=fsaverage['white_left'], interpolation='nearest', n_samples=1, radius=0.0))
textures.append(surface.vol_to_surf(atlas, fsaverage['pial_right'], inner_mesh=fsaverage['white_right'], interpolation='nearest', n_samples=1, radius=0.0))
textures.append(surface.vol_to_surf(atlas, fsaverage['pial_right'], inner_mesh=fsaverage['white_right'], interpolation='nearest', n_samples=1, radius=0.0))

# -- set up rois as 17 networks (remove specific roi info
n_color_dict = {'ContA':'indianred',
                'ContB':'firebrick',
                'ContC':'maroon',
                'DefaultA':'royalblue',
                'DefaultB':'mediumblue',
                'DefaultC':'navy',
                'DorsAttnA':'forestgreen',
                'DorsAttnB':'darkgreen',
                'LimbicA':'lightyellow',
                'LimbicB':'beige',
                'SalVentAttnA':'darkorchid',
                'SalVentAttnB':'mediumorchid',
                'SomMotA':'saddlebrown',
                'SomMotB':'sienna',
                'TempPar':'peachpuff',
                'VisCent':'lightgrey',
                'VisPeri':'darkgrey'}
n_textures=[]
n_textures.append(np.zeros(len(textures[0])))
n_textures.append(np.zeros(len(textures[1])))
n_textures.append(np.zeros(len(textures[2])))
n_textures.append(np.zeros(len(textures[3])))
custom_color_list = []
for n_idx, c_network in enumerate(sorted(list(set(networks)))):
    custom_color_list.append(n_color_dict[c_network])
    matching_indices = [index for index, value in enumerate(networks) if value == c_network]
    for idx, ctext in enumerate(textures):
        n_textures[idx][np.isin(ctext,matching_indices)] = n_idx+1.0
# -- set up custom color map for networks
discrete_cmap = mcolors.ListedColormap(custom_color_list)

# load the Schaefer 2018 atlas (400 ROIs)
atlas_img = nib.load(atlas)
roi_labels = atlas_data.labels  # list of 400 ROI labels
roi_indices = np.arange(1, 401)  # assuming atlas labels are 1-indexed
# Use NiftiLabelsMasker with the Schaefer atlas to extract ROI betas.
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
resampled_atlas_img = resample_img(atlas_img, target_affine=cortical_mask.affine, target_shape=cortical_mask_data.shape, interpolation='nearest', force_resample=True)



# --------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - -
# - - -   Parametric Modulation   - - -
# - - - - - - - - - - - - - - - - - - -
PM_path = os.path.join(dataset_dir, "3dMEMA")

# - - - - lists for loading my own data
mask_list = ["feedback_zTaskPE_masked.BRIK",
            "cue_stateD_masked.BRIK",
            "cue_zState_masked.BRIK",
            "cue_zEntropy_masked.BRIK"]
model_list = ["feedback__zTaskPE_SPMGmodel_stats_REML__tval.nii.gz",
            "cue__StateD_SPMGmodel_stats_REML__tval.nii.gz",
            "cue__zState_SPMGmodel_stats_REML__tval.nii.gz",
            "cue__zEntropy_SPMGmodel_stats_REML__tval.nii.gz"]
vmin_list = [-6, -6, -6, -6]
vmax_list = [6, 6, 6, 6]

if args.networks17_indiv_plots:
    # - - - - - - - - -
    # plot all 17 networks separately
    # - - - - - - - - -
    for m_idx, c_model in enumerate(model_list):
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(PM_path, "nii3D", mask_list[m_idx])) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        roi_data = np.where(np.squeeze(brik_data)>0,1,0) # binarize just in case its not already    
        roi_data = np.where(np.squeeze(cortical_mask_data)>0,roi_data,0) # apply cortical mask too
        mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
        # -- load stat image
        stat_img0 = nib.load(os.path.join(PM_path, "nii3D", c_model))
        stat_masked = nilearn.masking.apply_mask(stat_img0, mask_img)
        stat_img = masking.unmask(stat_masked, mask_img)
        print("cur stat: ", os.path.join(PM_path, "nii3D", c_model))

        for n_idx, c_network in enumerate(sorted(list(set(networks)))):
            print("plotting with network ", c_network)
            matching_indices = [index for index, value in enumerate(networks) if value == c_network]
            
            # -- this is to pull out only the current network we want to plot
            n_textures=[]
            for ctext in textures:
                temp = np.zeros(len(ctext))
                temp[np.isin(ctext,matching_indices)] = n_idx + 1 # should assign colors based on custom color map
                n_textures.append( temp )
            
            figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
            
            stat_img_list = []
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
            
            # ------------------ plot sagital surface plots w/ 17 networks ------------------ #
            # plot atlas 
            cur_color_map = mcolors.ListedColormap(n_color_dict[c_network])
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[0], hemi='left', view='lateral',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[0])
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[1], hemi='left', view='medial',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[1])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[2], hemi='right', view='medial',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[2])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[3], hemi='right', view='lateral',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[3])
            
            #plot maps
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                    darkness=0.1, alpha=0.005, axes=axes[0], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                    darkness=0.1, alpha=0.005, axes=axes[1], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                    darkness=0.1, alpha=0.005, axes=axes[2], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                    darkness=0.1, alpha=0.005, axes=axes[3], threshold=2.985, vmin=-10, vmax=10)
            
            figures.subplots_adjust(wspace=0.01,hspace=0.00)
            figures.savefig(os.path.join(PM_path,"nilearn_plots", (c_network+"_"+c_model[:-34]+'.png')), bbox_inches='tight')
            plt.close('all')


if args.networks17_subgroup_plots:
    # - - - - - - - - -
    # plot subgroups of 17 networks 
    # - - - - - - - - -
    cortical_mask_data = cortical_mask.get_fdata()
    for m_idx, c_model in enumerate(model_list):
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(PM_path, "nii3D", mask_list[m_idx])) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        roi_data = np.where(np.squeeze(brik_data)>0,1,0) # binarize just in case its not already    
        roi_data = np.where(np.squeeze(cortical_mask_data)>0,roi_data,0) # apply cortical mask too
        mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
        # -- load stat image
        stat_img0 = nib.load(os.path.join(PM_path, "nii3D", c_model))
        stat_masked = nilearn.masking.apply_mask(stat_img0, mask_img)
        stat_img = masking.unmask(stat_masked, mask_img)
        print("cur stat: ", os.path.join(PM_path, "nii3D", c_model))

        sub_network_groups = {'Control': {'ContA':1, 'ContB':2, 'ContC':3},
                            'Default': {'DefaultA':4, 'DefaultB':5, 'DefaultC':6},
                            'DorsalAttention': {'DorsAttnA':7, 'DorsAttnB':8},
                            'Limbic': {'LimbicA':9, 'LimbicB':10},
                            'SalientVentralAttention': {'SalVentAttnA':11, 'SalVentAttnB':12},
                            'SomatoMotor': {'SomMotA':13, 'SomMotB':14},
                            'TemporalParietal': {'TempPar':15},
                            'Visual': {'VisCent':16, 'VisPeri':17}}
        
        for n_idx, c_network_group in enumerate(sub_network_groups.keys()):
            print("plotting with network group ", c_network_group)
            all_matching_indices = {}
            for c_network in sub_network_groups[c_network_group].keys():
                matching_indices = [index for index, value in enumerate(networks) if value == c_network]
                all_matching_indices[c_network] = matching_indices
            
            # -- this is to pull out only the current network we want to plot
            n_textures=[]
            for ctext in textures:
                temp = np.zeros(len(ctext))
                cur_color_list = ['silver']
                for c_network in all_matching_indices.keys():
                    matching_indices = all_matching_indices[c_network]
                    temp[np.isin(ctext,matching_indices)] = sub_network_groups[c_network_group][c_network] # should assign colors based on custom color map
                    cur_color_list.append(n_color_dict[c_network])
                cur_color_map = mcolors.ListedColormap(cur_color_list)
                n_textures.append( temp )
            
            figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
            #figures.suptitle(c_network, fontsize=16)
            
            stat_img_list = []
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
            
            # ------------------ plot sagital surface plots w/ 17 networks ------------------ #
            # plot atlas 
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[0], hemi='left', view='lateral',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[0])
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[1], hemi='left', view='medial',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[1])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[2], hemi='right', view='medial',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[2])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[3], hemi='right', view='lateral',
                                darkness=0.3, cmap=cur_color_map, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[3])
            
            #plot maps
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                    darkness=0.1, alpha=0.005, axes=axes[0], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                    darkness=0.1, alpha=0.005, axes=axes[1], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                    darkness=0.1, alpha=0.005, axes=axes[2], threshold=2.985, vmin=-10, vmax=10)
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                    darkness=0.1, alpha=0.005, axes=axes[3], threshold=2.985, vmin=-10, vmax=10)
            
            figures.subplots_adjust(wspace=0.01,hspace=0.00)
            figures.savefig(os.path.join(PM_path,"nilearn_plots", (c_network_group+"_"+c_model[:-34]+'.png')), bbox_inches='tight')
            plt.close('all')


if args.networks17_1plot:
    # - - - - - - - - -
    # plot all 17 networks together
    # - - - - - - - - -
    for m_idx, c_model in enumerate(model_list):
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(PM_path, "nii3D", mask_list[m_idx])) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        roi_data = np.where(np.squeeze(brik_data)>1,1,0) # binarize just in case its not already    
        roi_data = np.where(np.squeeze(cortical_mask_data)>0,roi_data,0) # apply cortical mask too
        mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
        
        stat_img0 = nib.load(os.path.join(PM_path, "nii3D", c_model))
        stat_masked = nilearn.masking.apply_mask(stat_img0, mask_img)
        stat_img = masking.unmask(stat_masked, mask_img)
        print("cur stat: ", os.path.join(PM_path, "nii3D", c_model))
        
        stat_img_list = []
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right']))
        
        # ------------------ plot sagital surface plots w/ 17 networks ------------------ #
        figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
        figures.suptitle(c_network, fontsize=16)
        
        # plot atlas 
        plotting.plot_surf_roi(fsaverage.pial_left, roi_map=textures[0], hemi='left', view='lateral',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[0])
        plotting.plot_surf_roi(fsaverage.pial_left, roi_map=textures[1], hemi='left', view='medial',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[1])
        plotting.plot_surf_roi(fsaverage.pial_right, roi_map=textures[2], hemi='right', view='medial',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[2])
        plotting.plot_surf_roi(fsaverage.pial_right, roi_map=textures[3], hemi='right', view='lateral',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[3])
        
        #plot maps
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                darkness=0.1, alpha=0.005, axes=axes[0], threshold=2.985, vmin=-10, vmax=10)
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                darkness=0.1, alpha=0.005, axes=axes[1], threshold=2.985, vmin=-10, vmax=10)
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                darkness=0.1, alpha=0.005, axes=axes[2], threshold=2.985, vmin=-10, vmax=10)
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                darkness=0.1, alpha=0.005, axes=axes[3], threshold=2.985, vmin=-10, vmax=10)
        
        figures.subplots_adjust(wspace=0.01,hspace=0.00)
        figures.savefig(os.path.join(PM_path,"nilearn_plots", ("17networks_"+c_model[:-34]+'.png')), bbox_inches='tight')
        plt.close('all')


if args.networks17_bar_plots:
    # - - - - - - - - -
    # bar plots of number of clusters in each network
    # - - - - - - - - -
    # loop through each sig roi mask
    for m_idx, c_mask in enumerate(mask_list):
        print("working with ", c_mask)
        # -- set up dict to save rois per network for bar plots
        results_dict = {'ContA':0, 'ContB':0, 'ContC':0,
                    'DefaultA':0, 'DefaultB':0, 'DefaultC':0,
                    'DorsAttnA':0, 'DorsAttnB':0,
                    'LimbicA':0, 'LimbicB':0,
                    'SalVentAttnA':0, 'SalVentAttnB':0,
                    'SomMotA':0, 'SomMotB':0,
                    'TempPar':0,
                    'VisCent':0, 'VisPeri':0}
        total_voxel_count = 0
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(PM_path, "nii3D", c_mask)) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        for roi in range(int(brik_data.max())):
            sig_clust_data = np.where(np.squeeze(brik_data)==(roi+1),1,0) # binarize current roi
            sig_clust_data = np.where(np.squeeze(cortical_mask_data)>0,sig_clust_data,0) # apply cortical mask too
            sig_clust_img = nib.Nifti1Image(sig_clust_data, brik_mask.affine, brik_mask.header)
            
            try:
                atlas_masked = nilearn.masking.apply_mask(resampled_atlas_img, sig_clust_img)
                atlas_masked = atlas_masked[atlas_masked>0] # only use voxels with atlas labels so it all sums to 1
            except:
                print("error probably due to no data after masking.. just move on and skip this roi")
                continue
            
            #num_voxels_w_roi_labels = atlas_masked.shape 
            # -- okay pull networks that match rois with data
            c_networks_list = [networks[(i-1)] for i in atlas_masked.astype(int)]
            for c_net in c_networks_list:
                results_dict[c_net] += 1 # add 1 to show 1 voxel overlapped with this sig cluster
                total_voxel_count += 1
        
        for net_key in results_dict.keys():
            results_dict[net_key] = results_dict[net_key] / total_voxel_count
        
        results_df = pd.DataFrame(results_dict, index=[0])
        melt_df = pd.melt(results_df)
        print(melt_df)
        sns.barplot(x='variable', y='value', data=melt_df, palette=custom_color_list, edgecolor='black', linewidth=1.5)
        plt.title(c_mask[:-11])
        plt.xlabel("Network")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Percent of significant voxels overlapping with each network")
        
        plt.savefig(os.path.join(PM_path,"nilearn_plots",("17networks__bar_plot__"+c_mask[:-11]+".png")), bbox_inches='tight')
        
        plt.show()


if args.PM_stats_plots:
    for m_idx, c_model in enumerate(model_list):
        # -- load current stat image sig clusters mask
        brik_mask = nib.load(os.path.join(PM_path, "nii3D", mask_list[m_idx])) # load sig rois for state or color
        brik_data = brik_mask.get_fdata()
        roi_data = np.where(np.squeeze(brik_data)>0,1,0) # binarize just in case its not already    
        roi_data = np.where(np.squeeze(cortical_mask_data)>0,roi_data,0) # apply cortical mask too
        mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
        
        stat_img0 = nib.load(os.path.join(PM_path, "nii3D", c_model))
        stat_masked = nilearn.masking.apply_mask(stat_img0, mask_img)
        stat_img = masking.unmask(stat_masked, mask_img)
        print("cur stat: ", os.path.join(PM_path, "nii3D", c_model))
        
        if c_model == "cue__StateD_SPMGmodel_stats_REML__tval":
            cut_cords_list = [[-15], [-6, -3], [6, 15], [30, 42, 48]] #left-medial, right-medial, left-lateral, right-lateral
        else:
            cut_cords_list = [[-15], [-6, -3], [3, 12, 18], [39, 48, 54]]

        # ------------------ plot axial slices ------------------ #
        L_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[0], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        L_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'__Inferior_Ax-1.png')))
        plt.close('all')
        R_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[1], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        R_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'__Inferior_Ax-2.png')))
        plt.close('all')
        
        L_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[2], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        L_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'__Superior_Ax-1.png')))
        plt.close('all')
        R_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[3], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        R_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'__Superior_Ax-2.png')))
        plt.close('all')

        figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)

        # ------------------ plot sagital surface plots ------------------ #
        stat_img_list = []
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
        stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
        
        figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
        
        #plot maps
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                darkness=0.1, axes=axes[0], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                darkness=0.1, axes=axes[1], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                darkness=0.1, axes=axes[2], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                darkness=0.1, axes=axes[3], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        
        figures.subplots_adjust(wspace=0.01,hspace=0.00)
        figures.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'.png')), bbox_inches='tight')
        plt.close('all')
        
        # ------------------ plot sagital surface plots w/ 17 networks ------------------ #
        figures, axes = plt.subplots(figsize=(15,15),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
        
        # plot atlas 
        plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[0], hemi='left', view='lateral',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[0])
        plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[1], hemi='left', view='medial',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[1])
        plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[2], hemi='right', view='medial',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[2])
        plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[3], hemi='right', view='lateral',
                            darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[3])
        
        #plot maps
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                darkness=0.1, alpha=0.005, axes=axes[0], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                darkness=0.1, alpha=0.005, axes=axes[1], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                darkness=0.1, alpha=0.005, axes=axes[2], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                darkness=0.1, alpha=0.005, axes=axes[3], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
        
        figures.subplots_adjust(wspace=0.01,hspace=0.00)
        figures.savefig(os.path.join(PM_path,"nilearn_plots", (c_model[:-34]+'_plus17networks.png')), bbox_inches='tight')
        plt.close('all')
        


# --------------------------------------------------------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - -
# - - -   Probabalistic Decoding  - - -
# - - - - - - - - - - - - - - - - - - -
PM_path = os.path.join(dataset_dir, "Decoding", "GroupStats")
# ---- set up plot variables
model_list = ["color", "state", "jointP", "task"]
vmin_list = [-6, -6, -10, -10]
vmax_list = [6, 6, 10, 10]

if args.D_stats_plots:
    for epoch in ["cue","probe"]:
        for m_idx, c_model in enumerate(model_list):
            # -- load current stat image sig clusters mask
            brik_mask = nib.load(os.path.join(PM_path, (epoch+"_"+c_model+"_masked.BRIK"))) # load sig rois for state or color
            brik_data = brik_mask.get_fdata()
            if c_model=="jointP":
                roi_data = np.where(np.squeeze(brik_data)>0,1,0) # binarize just in case its not already    
            else:
                roi_data = np.where(np.squeeze(brik_data)>1,1,0)
            roi_data = np.where(np.squeeze(cortical_mask_data)>0,roi_data,0) # apply cortical mask too
            mask_img = nib.Nifti1Image(roi_data, brik_mask.affine, brik_mask.header)
            
            stat_img0 = nib.load(os.path.join(PM_path, ("GroupAnalysis_38subjs__"+c_model+"_"+epoch+"__r__tval.nii")))
            stat_masked = nilearn.masking.apply_mask(stat_img0, mask_img)
            stat_img = masking.unmask(stat_masked, mask_img)
            print("cur stat: ", c_model)
            
            if c_model == "state":
                cut_cords_list = [[-30, -16], [-12, -8, -2], [12, 20], [50, 56]] #left-medial, right-medial, left-lateral, right-lateral
            elif c_model == "color":
                if epoch=="cue":
                    cut_cords_list = [[-8], [-8], [30, 33, 37], [45, 54, 64]]
                else:
                    cut_cords_list = [[-6], [-6], [39], [52, 59]]
            elif c_model=="task":
                cut_cords_list = [[-16, -9], [-1], [7, 16, 21, 30], [35, 47, 54, 64]]
            elif c_model=="jointP":
                cut_cords_list = [[-16, -12, -9], [-1], [7, 16, 21, 30], [35, 47, 54, 64]]
            

            # ------------------ plot axial slices ------------------ #
            L_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[0], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            L_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model+"_"+epoch+'__Inferior_Ax-1.png')))
            plt.close('all')
            R_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[1], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            R_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model+"_"+epoch+'__Inferior_Ax-2.png')))
            plt.close('all')
            
            L_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[2], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            L_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model+"_"+epoch+'__Superior_Ax-1.png')))
            plt.close('all')
            R_sag_cuts = plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[3], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            R_sag_cuts.savefig(os.path.join(PM_path,"nilearn_plots", (c_model+"_"+epoch+'__Superior_Ax-2.png')))
            plt.close('all')
            
            
            # ------------------ plot axial slices w/ 17 networks ------------------ #
            figures, axes = plt.subplots(figsize=(25,25),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
            
            plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[0], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[0])
            plotting.plot_roi(atlas, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[0], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[0])
            
            plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[1], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[1])
            plotting.plot_roi(atlas, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[1], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[1])
            
            plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[2], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[2])
            plotting.plot_roi(atlas, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[2], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[2])
            
            plotting.plot_stat_map(stat_map_img=stat_img, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[3], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[2])
            plotting.plot_roi(atlas, threshold=2.985, cmap=plt.cm.RdBu_r, display_mode='z', cut_coords=cut_cords_list[3], vmin=vmin_list[m_idx], vmax=vmax_list[m_idx], axes=axes[2])
            
            figures.savefig(os.path.join(PM_path,"nilearn_plots", (c_model+"_"+epoch+'__Axial_Slices.png')))
            plt.close('all')
            
            
            # ------------------ plot sagital surface plots ------------------ #
            stat_img_list = []
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_left'], inner_mesh=fsaverage['white_left']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right'], inner_mesh=fsaverage['white_right']))
            stat_img_list.append(surface.vol_to_surf(stat_img, fsaverage['pial_right']))
            
            figures, axes = plt.subplots(figsize=(15,15),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
            
            #plot maps
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                    darkness=0.95, alpha=0.01, axes=axes[0], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                    darkness=0.95, alpha=0.01, axes=axes[1], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                    darkness=0.95, alpha=0.01, axes=axes[2], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                    darkness=0.95, alpha=0.01, axes=axes[3], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            
            figures.subplots_adjust(wspace=0.01,hspace=0.00)
            figures.savefig(os.path.join(PM_path, "nilearn_plots", (c_model+"_"+epoch+'.png')), bbox_inches='tight')
            plt.close('all')
            
            
            # ------------------ plot sagital surface plots w/ 17 networks ------------------ #
            figures, axes = plt.subplots(figsize=(15,15),nrows=1,ncols=4,subplot_kw={'projection': '3d'},sharex=True,sharey=True)
            
            # plot atlas 
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[0], hemi='left', view='lateral',
                                darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[0])
            plotting.plot_surf_roi(fsaverage.pial_left, roi_map=n_textures[1], hemi='left', view='medial',
                                darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_left, axes=axes[1])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[2], hemi='right', view='medial',
                                darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[2])
            plotting.plot_surf_roi(fsaverage.pial_right, roi_map=n_textures[3], hemi='right', view='lateral',
                                darkness=0.3, cmap=discrete_cmap, bg_on_data=True, bg_map=fsaverage.sulc_right, axes=axes[3])
            
            #plot maps
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[0], bg_map=fsaverage.sulc_left, hemi='left', view='lateral',
                                    darkness=0.1, alpha=0.01, axes=axes[0], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_left, colorbar=False, stat_map=stat_img_list[1], bg_map=fsaverage.sulc_left, hemi='left', view='medial',
                                    darkness=0.1, alpha=0.01, axes=axes[1], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[2], bg_map=fsaverage.sulc_right, hemi='right', view='medial',
                                    darkness=0.1, alpha=0.01, axes=axes[2], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            plotting.plot_surf_stat_map(fsaverage.pial_right, colorbar=False, stat_map=stat_img_list[3], bg_map=fsaverage.sulc_right, hemi='right', view='lateral',
                                    darkness=0.1, alpha=0.01, axes=axes[3], threshold=2.985, vmin=vmin_list[m_idx], vmax=vmax_list[m_idx])
            
            figures.subplots_adjust(wspace=0.01,hspace=0.00)
            figures.savefig(os.path.join(PM_path, "nilearn_plots", (c_model+"_"+epoch+'_plus17networks.png')), bbox_inches='tight')
            plt.close('all')


if args.networks17_bar_plots:
    # - - - - - - - - -
    # bar plots of number of clusters in each network
    # - - - - - - - - -
    # loop through each sig roi mask
    for epoch in ["cue","probe"]:
        for m_idx, c_mask in enumerate(model_list):
            print("working with ", c_mask)
            # -- set up dict to save rois per network for bar plots
            results_dict = {'ContA':0, 'ContB':0, 'ContC':0,
                        'DefaultA':0, 'DefaultB':0, 'DefaultC':0,
                        'DorsAttnA':0, 'DorsAttnB':0,
                        'LimbicA':0, 'LimbicB':0,
                        'SalVentAttnA':0, 'SalVentAttnB':0,
                        'SomMotA':0, 'SomMotB':0,
                        'TempPar':0,
                        'VisCent':0, 'VisPeri':0}
            total_voxel_count = 0
            
            # -- load current stat image sig clusters mask
            brik_mask = nib.load(os.path.join(PM_path, (epoch+"_"+c_mask+"_masked.BRIK"))) # load sig rois for state or color
            brik_data = brik_mask.get_fdata()
            for roi in range(int(brik_data.max())):
                if c_mask!="jointP":
                    if roi==0:
                        continue # skip first roi for color, state, and task
                sig_clust_data = np.where(np.squeeze(brik_data)==(roi+1),1,0) # binarize current roi
                sig_clust_data = np.where(np.squeeze(cortical_mask_data)>0,sig_clust_data,0) # apply cortical mask too
                sig_clust_img = nib.Nifti1Image(sig_clust_data, brik_mask.affine, brik_mask.header)
                
                try:
                    atlas_masked = nilearn.masking.apply_mask(resampled_atlas_img, sig_clust_img)
                    atlas_masked = atlas_masked[atlas_masked>0] # only use voxels with atlas labels so it all sums to 1
                except:
                    print("error probably due to no data after masking.. just move on and skip this roi")
                    continue
                
                #num_voxels_w_roi_labels = atlas_masked.shape 
                # -- okay pull networks that match rois with data
                c_networks_list = [networks[(i-1)] for i in atlas_masked.astype(int)]
                for c_net in c_networks_list:
                    results_dict[c_net] += 1 # add 1 to show 1 voxel overlapped with this sig cluster
                    total_voxel_count += 1
                
            for net_key in results_dict.keys():
                results_dict[net_key] = results_dict[net_key] / total_voxel_count
            
            results_df = pd.DataFrame(results_dict, index=[0])
            melt_df = pd.melt(results_df)
            print(melt_df)
            sns.barplot(x='variable', y='value', data=melt_df, palette=custom_color_list, edgecolor='black', linewidth=1.5)
            plt.title(c_mask)
            plt.xlabel("Network")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Percent of significant voxels overlapping with each network")
            
            plt.savefig(os.path.join(PM_path, "nilearn_plots", ("17networks__bar_plot__"+c_mask+"_"+epoch+".png")), bbox_inches='tight')
            
            plt.show()