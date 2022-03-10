import nibabel as nib
import numpy as np
import os
import gdist

base_dir = '/home/fralberti/Data/HCP/'
f = open('/home/fralberti/Data/HCP/subj_IDs_200.txt', 'r')
subjects = f.read().splitlines()
print('Current subject:')
for sub in subjects[51:-1]:
    
    if os.path.exists(os.path.join(base_dir, '%s/%s.dist_prim.32k_fs_LR.dlabel.nii' % (sub, sub))) is False:

        print(f'\t{sub}')
        
        lab_header = nib.cifti2.cifti2.load(os.path.join(base_dir, '%s/Structural/%s.aparc.a2009s.32k_fs_LR.dlabel.nii' % (sub, sub)))
        dlabel = nib.load(os.path.join(base_dir, '%s/Structural/%s.aparc.a2009s.32k_fs_LR.dlabel.nii' % (sub, sub)))
        labels = dlabel.get_fdata()[0]
        bms = {'L':dlabel.header.get_index_map(1)[0], 'R':dlabel.header.get_index_map(1)[1]}
        lab = np.zeros(lab_header.shape[1])
        zone = np.array([],dtype='int32')
        dist = np.array([],dtype='int32')
        
        for hemi in ['L', 'R']:

            surf = nib.load(os.path.join(base_dir, '%s/Structural/%s.%s.midthickness_MSMAll.32k_fs_LR.surf.gii' % (sub, sub, hemi)))
            offset = bms[hemi].index_offset
            count = bms[hemi].index_count
            cort = np.asarray(bms[hemi].vertex_indices[:])
            if hemi == 'L':
                regions = [46, 45, 33]
            elif hemi == 'R':
                regions = [121, 120, 108]
                
            # load central sulcus and calcarine
            # regions = ['L_S_central', 'L_S_calcarine', 'L_G_temp_sup-G_T_transv']
            # labels[1][0].label
            src = []
            for r in regions:
                src_64 = cort[labels[offset:offset+count] == r]
                src_32 = np.zeros(len(src_64),dtype='int32')
                src_32[:] = src_64[:]
                src.append(src_32)
                lab[offset:offset+count][labels[offset:offset+count] == r] = 1

            vertices = np.array(surf.darrays[0].data, dtype=np.float64)
            triangles = np.array(surf.darrays[1].data, dtype=np.int32)
            all_vert = np.array(np.unique(triangles))
            dist_vals = np.zeros((len(src), 32492))

            for x in range(len(src)):
                dist_vals[x, :] = gdist.compute_gdist(vertices, triangles, source_indices=src[x]) # src_new
            dist_vals = dist_vals[:,cort]
            if hemi == 'L':
                zone = np.concatenate((zone, np.argsort(dist_vals, axis=0)[0, :] + 1))
                dist = np.concatenate((dist, np.sort(dist_vals, axis=0)[0, :]))
            elif hemi == 'R':
                zone = np.concatenate((zone, np.argsort(dist_vals, axis=0)[0, :] + 1))
                dist = np.concatenate((dist, np.sort(dist_vals, axis=0)[0, :]))
#         z = np.asanyarray(np.expand_dims(zone[cort], axis=0))
        z = np.zeros([1,lab_header.shape[1]])
        z[0,0:len(zone)] = zone[:]
        
        img = nib.Cifti2Image(z, header = lab_header.header, nifti_header=lab_header.nifti_header)
        img.update_headers()
        img.to_filename(os.path.join(base_dir, '%s/Structural/%s.zone_prim.32k_fs_LR.dlabel.nii' % (sub, sub)))
        
print('Done!')