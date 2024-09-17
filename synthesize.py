import pandas as pd
import numpy as np
import xarray as xr
import cv2 as cv2
import matplotlib.pyplot as plt
import tpae.data.samples as tds

cells = pd.read_csv('../alz-data/SEAAD_MTG_MERFISH_metadata.2024-05-03.noblanks.harmonized.txt', delimiter='\t')

def get_region(s, cell_types=['L2/3 IT']):
    mask = tds.get_mask(s)
    layer = xr.zeros_like(mask)

    # find cells
    mycells = cells[cells.Section == f'{s.attrs['donor']}_{s.attrs['sid']}']
    mycells_ = mycells[mycells.subclass_name.isin(cell_types)]
    for x, y in mycells_[['x','y']].values:
        nearest = layer.sel(x=x, y=y, method='nearest')
        layer.loc[nearest.y.item(), nearest.x.item()] = 1

    # smooth
    layer.data = cv2.morphologyEx(layer.data.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40)))
    layer.data = cv2.morphologyEx(layer.data.astype(np.uint8), cv2.MORPH_OPEN, np.ones((10,10),np.uint8)) 
    return layer.astype('bool')

def add_aggregates_v_diffuse(samples, pc='hPC2', seed=0, plot=False):
    np.random.seed(seed)

    # determine case/ctrl status
    samplemeta = pd.DataFrame({'donor':[s.attrs['donor'] for s in samples.values()]},
                               index=[s.attrs['sid'] for s in samples.values()]).drop_duplicates()
    samples = {sid:samples[sid] for sid in samplemeta.index}
    cases = np.random.choice(samplemeta.index, size=len(samplemeta)//2, replace=False)
    samplemeta['case'] = 0
    samplemeta.loc[cases, 'case'] = 1

    # add in signal
    region_masks = {}
    for sid, r in samplemeta.iterrows():
        print(sid, r.case, end='|')
        s = samples[sid]

        # determine tissue boundaries and region of interest
        tissue_mask = cv2.morphologyEx(tds.get_mask(s).data.astype(np.uint8), cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))).astype('bool')
        region_mask = get_region(s).data
        region_mask = region_mask & tissue_mask
        region_masks[sid] = region_mask

        # create case and ctrl signals
        celltype = s.sel(marker=pc)
        celltype_mask = celltype.where(celltype > 5, other=0).data
        newpx_case = cv2.GaussianBlur(celltype_mask, (7,7), 0)
        newpx_ctrl = cv2.GaussianBlur(celltype_mask, (51,51), 0)
        
        # normalize
        newpx_case[~region_mask] = 0
        newpx_case *= (region_mask.flatten().sum() / newpx_case.flatten().sum())
        newpx_ctrl[~region_mask] = 0
        newpx_ctrl *= (region_mask.flatten().sum() / newpx_ctrl.flatten().sum())
        
        # add
        dist = newpx_case if r.case == 1 else newpx_ctrl
        celltype += dist

        # visualize
        if plot:
            ax = plt.subplot(1,3,1)
            plt.imshow(0.5*tissue_mask[::-1,:] + region_mask[::-1,:], vmin=0, vmax=1)
            plt.title('region')
            ax = plt.subplot(1,3,2)
            plt.imshow(newpx_case[::-1,:], vmin=0, vmax=2)
            plt.title('case')
            ax = plt.subplot(1,3,3)
            plt.imshow(newpx_ctrl[::-1,:], vmin=0, vmax=2)
            plt.title('ctrl')
            plt.show()

            ax = plt.subplot(1,2,1)
            (celltype-dist).plot(ax=ax)
            ax.axis('equal')
            ax = plt.subplot(1,2,2)
            celltype.plot(ax=ax)
            ax.axis('equal')
            plt.show()

    return samples, samplemeta, region_masks

def annotate_patches(patchmeta, region_masks):
    def frac_in_region(patch):
        return region_masks[patch.sid][patch.y:patch.y+patch.patchsize, patch.x:patch.x+patch.patchsize].mean()
    patchmeta['region'] = [frac_in_region(p) for _, p in patchmeta.iterrows()]
    print(patchmeta.region.mean())