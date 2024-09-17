import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpae.data.samples as tds
import tpae.data.patchcollection as tdp
import tpae.model as tm
import tpae.association as ta
import tpae.vis as tv
from tpaesim import synthesize
import tpaesim.cc
import gc, os
import cv2 as cv2
import xarray as xr
import argparse

import torch
torch.set_default_device('mps') # for running on Yakir's macbook's GPU

def make_trivial(P):
    P.augmentation_off()
    P.numpy_mode()

    Zs = {}
    Zs['trivial-avg'] = P[:][:,:,:,:].mean(axis=(1,2))
    Zs['trivial-pixels'] = P[:].reshape((len(P), -1))
    Zs['trivial-cov'] = np.array([z.T.dot(z).flatten()
        for z in P[:].reshape((len(P), -1, P.nchannels))])    

    return Zs

def make_simplecnn(P):
    return

def make_resnet(P):
    return

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="perform single simulation replicate for a specified family of methods")
    
    # Add arguments
    parser.add_argument("signal_type", type=str, help="The signal type (string)")
    parser.add_argument("rep_family", type=str, help="The rep family (string)")
    parser.add_argument("seed", type=int, help="The seed (integer)")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("--data_dir", type=str, help="Path to data", default="/Users/yakir/ST/ALZ/alz-data/10u/pca_k=10_harmony")
    parser.add_argument("--npcs", type=int, help="number of PCs to use for building UMAP; if none then no PCA used.", default=20)

    # Parse the arguments
    args = parser.parse_args()

    # Read data
    print('reading samples')
    samples = tds.read_samples(f'{args.data_dir}/*.nc', tds.default_parser)#, stop_after=10)

    # Spike in signal
    print('adding in case/ctrl signal')
    np.random.seed(args.seed)
    if args.signal_type == 'agg_v_diffuse':
        samples, samplemeta, region_masks = synthesize.add_aggregates_v_diffuse(samples, plot=False)
    else:
        #todo
        pass

    # Generate patches
    print('choosing patches')
    patchsize = 40; patchstride = 10
    patchmeta = tds.choose_patches(samples, patchsize, patchstride, max_frac_empty=0.8) #0.2)
    synthesize.annotate_patches(patchmeta, region_masks)
    P = tdp.PatchCollection(patchmeta, samples, standardize=True)
    print(len(patchmeta), 'patches generated')

    # Generate representations
    print('making representations')
    if args.rep_family == 'trivial':
        Zs = make_trivial(P)
    elif args.rep_family == 'simplecnn':
        Zs = make_simplecnn(P)
    elif args.rep_family == 'resnet':
        Zs = make_resnet(P)

    # Generate anndata
    print('making anndata objects')
    kwargs = {'use_rep':'X_pca', 'n_comps':args.npcs} if args.npcs is not None else {'use_rep':'X'}
    suffix = f'-{args.npcs}pcs' if args.npcs is not None else '-raw'
    Ds = {
        repname+suffix : ta.anndata(P.meta, Z, samplemeta, sampleid='sid', **kwargs)
        for repname, Z in Zs.items()
        }

    # Perform two case-control anlayses for each representation and noise level
    results = pd.DataFrame(columns=['repname', 'style', 'noise', 'P', 'accuracy'])
    noises = [0, 0.25, 0.5]
    for repname, D in Ds.items():
        for noise in noises:
            D.samplem['noisy_case'] = (D.samplem.case + np.random.binomial(1, noise, size=D.N)) % 2
            for style, cc in [('clust', tpaesim.cc.cluster_cc), ('cna', tpaesim.cc.cna_cc)]:
                p, accuracy = cc(D)
                results.loc[len(results)] = {"repname": repname, "style": style, "noise": noise, "P": p, "accuracy": accuracy}
                print(results.iloc[-1])

    # Write output as tsv with repname, p, accuracy
    results.to_csv(f'{args.outdir}/{args.signal_type}.{args.rep_family}.{args.seed}.tsv', sep='\t', index=False)