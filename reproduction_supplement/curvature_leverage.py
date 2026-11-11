"""Read saved chains and matching data; write diagnostics only, never sample.

Run from any directory. Chains may remain in the unchanged release ZIP.
Uses NumPy and the standard library, without scipy.stats or np.corrcoef.
"""
from pathlib import Path
import csv, hashlib, io, json, math, zipfile, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    table = ROOT/'reproduction/match_quality/matched_sgls_milp_dd_des.csv'
    rows = list(csv.DictReader(table.open(encoding='utf-8-sig')))
    col = lambda name: np.array([float(r[name]) for r in rows])
    dl = 70/299792.458*10**((col('sn_lens_MU')-25)/5)/(1+col('zl'))
    ds = 70/299792.458*10**((col('sn_source_MU')-25)/5)/(1+col('zs'))
    g = -.5*((dl**2-dl*ds)/2)/(1-dl/ds)
    assert len(rows)==88 and np.allclose(g,dl*ds/4)
    relerr = col('dsigma_ap')/col('sigma_ap')
    summary = json.loads((ROOT/'reproduction_physical/nonlinear_base_summary.json').read_text())
    name = 'reproduction_physical/nonlinear_base_chains.npy'
    local = ROOT/name
    if local.exists():
        raw = local.read_bytes(); source = name
    else:
        archive = ROOT.parent/'release_assets/saved_chains_apj_v6_20261025.zip'
        with zipfile.ZipFile(archive) as z:
            matches=[n for n in z.namelist() if n.endswith(name)]
            assert len(matches)==1, matches
            raw=z.read(matches[0]); source=archive.name+'::'+matches[0]
    chain=np.load(io.BytesIO(raw),allow_pickle=False)
    assert chain.shape==(4,100000,5)
    x=chain[:,33333:,:].reshape(-1,5)
    xc=x-x.mean(axis=0); cross=xc.T@xc
    corr=cross/np.sqrt(np.outer(np.diag(cross),np.diag(cross)))
    med=np.median(x,axis=0)
    assert np.allclose(med,summary['quantiles'][1],atol=1e-12)
    # Same spherical power-law structural factor as reproduction/inference.py.
    def structural(gamma,delta,beta):
        xi=gamma+delta-2
        return ((3-delta)/((xi-2*beta)*(3-xi))
                *(math.gamma((xi-1)/2)/math.gamma(xi/2)
                  -beta*math.gamma((xi+1)/2)/math.gamma((xi+2)/2))
                *math.gamma(gamma/2)*math.gamma(delta/2)
                /(math.gamma((gamma-1)/2)*math.gamma((delta-1)/2)))
    def log_sigma_slope(gamma,delta,beta,ap_ratio):
        return .5*(math.log(structural(gamma,delta,beta))+(2-gamma)*math.log(ap_ratio))
    derivatives=[]
    for r in rows:
        gamma=med[1]+med[2]*float(r['zl']); delta=float(r['delta'])
        ar=float(r['thetaap'])/float(r['thetaE']); eps=1e-5
        derivatives.append((log_sigma_slope(gamma+eps,delta,med[3],ar)-log_sigma_slope(gamma-eps,delta,med[3],ar))/(2*eps))
    derivative=np.array(derivatives)
    q=med[0]; al=np.sqrt(1+q*dl**2); ass=np.sqrt(1+q*ds**2)
    g_at_med=-.25*(dl**2/al-dl*ds/ass)/(al-dl/ds*ass)
    result={'scope':'Fixed lens parameters, observed SN moduli; Fisher estimate ignores SN errors and nuisance marginalization. Not a posterior precision or rigorous bound for this truncated posterior.',
      'N':len(rows),'H0':70,'c_kms':299792.458,'burn':33333,'shape':list(chain.shape),
      'inputs':{'table':str(table.relative_to(ROOT)),'table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'chain':source,'chain_array_file_sha256':hashlib.sha256(raw).hexdigest()},
      'distance_l_median':float(np.median(dl)),'distance_s_median':float(np.median(ds)),
      'sensitivity_median':float(np.median(g)),'sensitivity_mean':float(np.mean(g)),'sensitivity_max':float(np.max(g)),
      'fractional_measurement_error_median':float(np.median(relerr)),
      'fractional_sigma_change_Omega_0p1':float(np.median(g)*.1),
      'fisher_sigma':{str(f):float(1/np.sqrt(np.sum(g*g/(relerr**2+f*f)))) for f in (0.,.111)},
      'labels':summary['labels'],'pearson':corr.tolist(),'posterior_medians':med.tolist(),
      'gamma_derivative_median':float(np.median(derivative)),
      'abs_gamma_derivative_median':float(np.median(np.abs(derivative))),
      'Omega_equivalent_to_gamma_0p01_median':float(np.median(np.abs(.01*derivative/g_at_med))),
      'gamma_derivative_step':1e-5,
      'per_lens':{'curvature_derivative_at_zero':g.tolist(),'gamma_derivative_at_posterior_medians':derivative.tolist()}}
    result['CMB_curvature_lower']=summary['central_geometric_lower']
    wide=ROOT/'reproduction_physical/nonlinear_gzwide_summary.json'
    result['wide_gamma_z_prior']={
        'source':str(wide.relative_to(ROOT)),
        'Omega_median':json.loads(wide.read_text())['quantiles'][1][0],
        'gamma_z_bounds':[-1,1]}
    result['flat_simulations']=[{'source':str(p.relative_to(ROOT)),
        'Omega_median':json.loads(p.read_text())['quantiles'][1][0]}
        for p in sorted((ROOT/'reproduction_supplement/simulations').glob('*_summary.json'))
        if p.name.startswith(('constant_', 'evolving_'))]
    result['no_CMB_diagnostic']={
        'Omega_approx':-.95,
        'provenance':'Existing manuscript Appendix A, inherited diagnostic; not newly computed here.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_lens','inputs','scope')},indent=2))

if __name__=='__main__': main()
