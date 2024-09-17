import scanpy as sc
import numpy as np
import pandas as pd
import cna
from scipy.stats import ttest_ind
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def annotate_patches(d):
	if 'case' not in d.obs.columns:
		d.obs['case'] = pd.merge(d.obs[['sid']], d.samplem[['case']], left_on='sid', right_index=True, how='left').case

	if 'case_region' not in d.obs.columns:
		d.obs['case_region'] = d.obs.case * d.obs.region
		d.obs['ctrl_region'] = (d.obs.case - 1) * (d.obs.region)

def accuracy(d, method):
	annotate_patches(d)
	return np.corrcoef((d.obs.case == 1) * d.obs.region - (d.obs.case == 0) * d.obs.region, d.obs[method])[0,1]

def cna_cc(d, seed=0, **kwargs):
	d.samplem = d.samplem.loc[d.obs.sid.unique()]
	cna.tl.nam(d, show_progress=True, force_recompute=True)
	np.random.seed(seed)
	res = cna.tl.association(d, d.samplem.noisy_case == 1, allow_low_sample_size=True, Nnull=10000, **kwargs)
	print(f'P = {res.p}, used {res.k} PCs')

	d.obs['ncorr_cna'] = res.ncorrs
	return res.p, accuracy(d, 'ncorr_cna')

def cluster_cc(d, seed=0, Nnull=2000):
	if 'leiden1' not in d.obs.columns:
		sc.tl.leiden(d, resolution=1, key_added='leiden1')
	abundances = pd.crosstab(d.obs.sid, d.obs.leiden1)
	clusters = abundances.columns
	abundances['y'] = d.samplem.noisy_case

	null_ys = np.array([np.random.permutation(abundances.y.values) for _ in range(Nnull)])

	pvals = []
	Ts = []
	for col in pb(sorted(clusters)):
		T, p_val = ttest_ind(abundances[col][abundances.y==0], abundances[col][abundances.y==1])
		null_dist = np.array([ttest_ind(abundances[col][y_==0], abundances[col][y_==1])[1] for y_ in null_ys])
		pvals.append(((p_val >= null_dist).sum()+1)/(Nnull+1))
		Ts.append(-T)
	p = pd.Series(pvals, index=sorted(clusters))
	t = pd.Series(Ts, index=sorted(clusters))

	d.obs['ncorr_cluster'] = 0.
	for clust in clusters:
		d.obs.loc[d.obs.leiden1 == clust, 'ncorr_cluster'] = t.loc[clust]

	return p.min() * len(p), accuracy(d, 'ncorr_cluster')