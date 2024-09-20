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
		d.obs['ctrl_region'] = (1 - d.obs.case) * d.obs.region

def global_accuracy(d, method):
	return {
		'correlation' : np.corrcoef((d.obs.case == 1) * d.obs.region - (d.obs.case == 0) * d.obs.region, d.obs[f'{method}_ncorr'])[0,1]
	}

def statistical_accuracy(d, method):
	case_regions = d.obs[d.obs.case_region > 0.9]
	ctrl_regions = d.obs[d.obs.ctrl_region > 0.9]
	null_regions = d.obs[d.obs.region == 0]
	
	nct = f'{method}_ncorr_thresh'
	sensitivity = ((case_regions[nct] > 0).sum() + (ctrl_regions[nct] < 0).sum()) / (len(case_regions) + len(ctrl_regions))
	fdr = (null_regions[nct] != 0).sum() / (d.obs[nct] != 0).sum()
	
	return {
		'sensitivity' : sensitivity,
		'fdr': fdr
	}

def roc_curve(d, method, polarity):
	nc = f'{method}_ncorr'
	fpr = []; tpr = []
	incr = (d.obs[nc].max()-d.obs[nc].min())/200
	for t in np.arange(d.obs[nc].min()-incr, d.obs[nc].max()+2*incr, incr):
		if polarity > 0:
			pos = d.obs.case_region > 0.9
			null = (d.obs.region == 0) | (d.obs.ctrl_region > 0.9)
			pred_pos = d.obs[nc] > t
			pred_null = d.obs[nc] <= t
		else:
			pos = d.obs.ctrl_region > 0.9
			null = (d.obs.region == 0) | (d.obs.case_region > 0.9)
			pred_pos = d.obs[nc] < t
			pred_null = d.obs[nc] >= t
		fpr.append((pred_pos & null).sum() / null.sum())
		tpr.append((pos & pred_pos).sum() / pos.sum())
	roc = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
	roc = roc.dropna().sort_values(by='fpr')
	return np.trapz(roc.tpr, x=roc.fpr), roc
def auroc(d, method):
	return {
		'auroc': (roc_curve(d, method, +1)[0] + roc_curve(d, method, -1)[0])/2
	}

def metrics(d, method):
	return global_accuracy(d, method) | auroc(d, method)

def metric_names():
	return ['correlation', 'auroc']

def cna_cc(d, seed=0, **kwargs):
	annotate_patches(d)
	d.samplem = d.samplem.loc[d.obs.sid.unique()]
	cna.tl.nam(d, show_progress=True, force_recompute=True)
	np.random.seed(seed)
	res = cna.tl.association(d, d.samplem.noisy_case == 1, allow_low_sample_size=True, Nnull=10000, **kwargs)
	print(f'P = {res.p}, used {res.k} PCs')

	if (res.fdrs.fdr <= 0.05).sum() > 0:
		thresh = res.fdrs[res.fdrs.fdr <= 0.05].threshold.iloc[0]
	else:
		thresh = 1.1

	d.obs['cna_ncorr'] = res.ncorrs
	d.obs['cna_ncorr_thresh'] = res.ncorrs * (np.abs(res.ncorrs >= thresh))
	return res.p, metrics(d, 'cna')

def cluster_cc(d, seed=0, Nnull=2000):
	annotate_patches(d)
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

	d.obs['clust_ncorr'] = 0.
	d.obs['clust_ncorr_thresh'] = 0.
	for clust in clusters:
		d.obs.loc[d.obs.leiden1 == clust, 'clust_ncorr'] = t.loc[clust]
		if p.loc[clust] * len(p) <= 0.05:
			d.obs.loc[d.obs.leiden1 == clust, 'clust_ncorr_thresh'] = t.loc[clust]

	return p.min() * len(p), metrics(d, 'clust')