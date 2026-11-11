"""Independent scalar-chain ESS/Metropolis sampler for supplementary checks.

The target is the same normalized nonlinear Gaussian-distance target used
in the primary analysis. Inputs are supplied explicitly by a model object.
No existing chain or original data file is overwritten.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time,sys,hashlib
import numpy as np
from scipy.stats import rankdata,norm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from latent_validation import loglike,diagnostics

def _rhat(x):
    half=x.shape[1]//2
    y=np.concatenate([x[:,:half],x[:,-half:]],axis=0)
    within=y.var(axis=1,ddof=1).mean(axis=0)
    between=half*y.mean(axis=1).var(axis=0,ddof=1)
    return np.sqrt(((half-1)*within+between)/half/within)

def full_diagnostics(x):
    d=diagnostics(x)
    values=[]
    for array in [x,np.abs(x-np.median(x,axis=(0,1)))]:
        normalized=np.empty_like(array)
        for i in range(array.shape[-1]):
            ranks=rankdata(array[:,:,i])
            normalized[:,:,i]=norm.ppf((ranks-.375)/(ranks.size+.25)).reshape(array.shape[:2])
        values.append(_rhat(normalized).tolist())
    d['rank_normalized_split_rhat']=values[0]
    d['folded_rank_split_rhat']=values[1]
    d['max_rhat']=float(np.max([d['split_rhat'],*values]))
    d['min_ess']=float(np.min(d['conservative_ess']))
    return d

def sample_joint(model,output_prefix,steps=100000,nchains=4,seed=2026102101,
                 initial=None,overwrite=False):
    prefix=Path(output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    summary_path=Path(str(prefix)+'_summary.json')
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f'Existing run preserved: {summary_path}')
    n=model.n; center=np.r_[model.ml,model.ms]
    covariance=np.block([[model.cl,model.cls],[model.cls.T,model.cs]])
    assert covariance.dtype==np.float64 and covariance.shape==(2*n,2*n)
    chol=np.linalg.cholesky(covariance+np.eye(2*n)*1e-13)
    bounds=np.asarray(model.bounds,dtype=np.float64)
    args=(model.zl,model.zs,model.te,model.ta,model.delta,model.obs,
          model.err,bounds,False,False)
    burn=steps//3
    mu_indices=np.arange(((burn+99)//100)*100,steps,100,dtype=int)
    chains=[]; mu_chains=[]; rates=[]; started=time.perf_counter()
    initial=np.asarray([.1,2.07,-.25,-.15,.11] if initial is None else initial,dtype=float)
    for ch in range(nchains):
        rng=np.random.default_rng(seed+ch*1009)
        theta=initial+rng.normal(size=5)*[.025,.02,.025,.025,.005]
        mu=center.copy(); ll=loglike(theta,mu,*args)
        for attempt in range(1000):
            if np.isfinite(ll): break
            theta=initial+rng.normal(size=5)*[.03,.03,.04,.04,.007]
            mu=center+(chol@rng.normal(size=2*n) if attempt>20 else 0)
            ll=loglike(theta,mu,*args)
        else: raise RuntimeError('No finite initial state')
        proposal=np.diag(np.array([.04,.018,.035,.035,.006])**2)
        proposal_chol=np.linalg.cholesky(proposal)
        chain=np.empty((steps,5),dtype=np.float64)
        mus=np.empty((len(mu_indices),2*n),dtype=np.float64)
        accepted=0; evaluations=0; mu_counter=0
        for it in range(steps):
            for _ in range(3):
                trial=theta+proposal_chol@rng.normal(size=5)
                value=loglike(trial,mu,*args)
                if np.log(rng.random())<value-ll:
                    theta=trial; ll=value; accepted+=1
            nu=chol@rng.normal(size=2*n)
            level=ll+np.log(rng.random())
            angle=rng.uniform(0,2*np.pi); lo=angle-2*np.pi; hi=angle
            shifted=mu-center
            for _ in range(1000):
                candidate=center+shifted*np.cos(angle)+nu*np.sin(angle)
                value=loglike(theta,candidate,*args); evaluations+=1
                if value>level:
                    mu=candidate; ll=value; break
                if angle<0: lo=angle
                else: hi=angle
                angle=rng.uniform(lo,hi)
            else: raise RuntimeError('ESS bracket exhausted')
            chain[it]=theta
            if it<burn and it>1000 and (it+1)%500==0:
                proposal=2.38**2/5*(np.cov(chain[:it+1].T)+np.diag([1e-5,1e-6,1e-6,1e-6,1e-8]))
                proposal_chol=np.linalg.cholesky(proposal)
            if it>=burn and it%100==0:
                mus[mu_counter]=mu; mu_counter+=1
        assert mu_counter==len(mu_indices)
        chains.append(chain); mu_chains.append(mus)
        rates.append({'metropolis_acceptance':accepted/(3*steps),
                      'elliptical_evaluations_per_iteration':evaluations/steps})
        print(prefix.name,'chain',ch+1,'/',nchains,'elapsed',round(time.perf_counter()-started,1),'s',flush=True)
        np.save(str(prefix)+'_chains.npy',np.asarray(chains))
        np.save(str(prefix)+'_mu.npy',np.asarray(mu_chains))
    array=np.asarray(chains); kept=array[:,burn:]; flat=kept.reshape(-1,5)
    diagnostic=full_diagnostics(kept)
    inputs=hashlib.sha256()
    for arr in [center,covariance,model.zl,model.zs,model.te,model.ta,
                model.delta,model.obs,model.err,bounds]: inputs.update(np.asarray(arr).tobytes())
    result={'N':n,'labels':['Omega_k','gamma0','gamma_z','beta','f_int'],
            'steps':steps,'chains':nchains,'burn':burn,'seed':seed,
            'bounds':bounds.tolist(),'adaptation_stops_at_burn':True,
            'theta_updates_per_iteration':3,'mu_saved_indices':mu_indices.tolist(),
            'quantiles':np.percentile(flat,[15.865,50,84.135],axis=0).tolist(),
            'diagnostics':diagnostic,'rates':rates,
            'boundary_mass_gamma_z_below_minus_0p45':float(np.mean(flat[:,2]<-.45)),
            'p_Omega_ge_zero':float(np.mean(flat[:,0]>=0)),
            'input_array_sha256':inputs.hexdigest(),
            'convergence_screen':bool(diagnostic['max_rhat']<1.01 and diagnostic['min_ess']>1000),
            'seconds':time.perf_counter()-started}
    summary_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(prefix.name,'DONE',json.dumps({'Omega':result['quantiles'][1][0],
          'gamma_z':result['quantiles'][1][2],'max_Rhat':diagnostic['max_rhat'],
          'min_ESS':diagnostic['min_ess']}),flush=True)
    return result
