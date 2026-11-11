"""Revision with an explicit CMB geometric consistency constraint.

Preserves all previous chains. Main: gamma_z in [-0.5,0.5], H0=70,
D_A*=12.8 Mpc, hard central-value geometric boundary. A separate soft
geometric weight marginalizes the quoted 0.07-Mpc Gaussian distance error.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize
import emcee
from inference import Model as OriginalModel,ROOT,OUT as OLDOUT,C
from latent_validation import loglike as joint_velocity_loglike,diagnostics

OUT=ROOT/'reproduction_physical'; OUT.mkdir(exist_ok=True)
H0=70.; ZSTAR=1090.; DASTAR=12.8; DASTAR_ERR=.07
DSTAR=(1+ZSTAR)*H0/C*DASTAR
LOWER=-1/DSTAR**2

class PhysicalModel(OriginalModel):
    def __init__(self,case='base'):
        underlying=case if case not in {'soft07','soft70'} else 'base'
        super().__init__(underlying)
        self.analysis_case=case; self.underlying=underlying
        self.soft_sigma=.07 if case=='soft07' else .7 if case=='soft70' else None
        self.qbound=(-2.,2.) if self.soft_sigma is not None else (LOWER,2.)
        self.bounds[0]=self.qbound
    def geometric_weight(self,q):
        q=np.asarray(q); result=np.zeros_like(q,dtype=float); negative=q<0
        if self.soft_sigma is not None:
            maximum=C/(H0*(1+ZSTAR)*np.sqrt(-q[negative]))
            result[negative]=log_ndtr((maximum-DASTAR)/self.soft_sigma)
        return result
    def __call__(self,theta):
        t=np.atleast_2d(theta)
        return super().__call__(t)+self.geometric_weight(t[:,0])

def save(case,kind,chains,burn,extra):
    kept=chains[:,burn:]; flat=kept.reshape(-1,chains.shape[-1]); model=PhysicalModel(case)
    r={'case':case,'kind':kind,'N':model.n,'labels':model.names,'bounds':model.bounds,'gamma_z_prior':list(model.gzbound),'H0_fixed':H0,'z_star':ZSTAR,'DA_star_Mpc':DASTAR,'DA_star_error_Mpc':DASTAR_ERR,'central_geometric_lower':LOWER,'external_geometric_model':'Gaussian-distance CDF consistency weight' if model.soft_sigma else 'hard central-value distance bound','soft_sigma_Mpc':model.soft_sigma,'burn':burn,'quantiles':np.percentile(flat,[15.865,50,84.135],axis=0).tolist(),'p_Omega_ge_zero':float(np.mean(flat[:,0]>=0)),'corr':np.corrcoef(flat.T).tolist(),**extra}
    r['lower_edge_mass_within_0p02']=float(np.mean(flat[:,0]<LOWER+.02))
    r['gamma_lower_edge_mass_within_0p05']=float(np.mean(flat[:,2]<model.gzbound[0]+.05))
    path=OUT/f'{kind}_{case}_summary.json'; path.write_text(json.dumps(r,indent=2),encoding='utf-8'); print(kind,case,'quantiles',r['quantiles'],'seconds',r['seconds'],flush=True)

def run_nonlinear(case,steps=100000,nchains=4):
    model=PhysicalModel(case); n=model.n; mu0=np.r_[model.ml,model.ms]
    cov=np.block([[model.cl,model.cls],[model.cls.T,model.cs]]); chol=np.linalg.cholesky(cov+np.eye(2*n)*1e-13)
    args=(model.zl,model.zs,model.te,model.ta,model.delta,model.obs,model.err,np.array(model.bounds),model.underlying=='lens_noise',model.underlying=='beta_prior')
    def target(t,mu):
        result=joint_velocity_loglike(t,mu,*args)
        if np.isfinite(result) and model.soft_sigma is not None:
            result+=float(model.geometric_weight(t[:1])[0])
        return result
    burn=steps//3; chains=[]; rates=[]; allmu=[]; begun=time.perf_counter()
    for ch in range(nchains):
        rng=np.random.default_rng(94122+ch*551); theta=np.array([.1,2.07,-.25,-.15,.1])+rng.normal(size=5)*[.02,.01,.01,.01,.005]
        mu=mu0.copy(); ll=target(theta,mu); assert np.isfinite(ll)
        proposal=np.diag(np.array([.04,.018,.035,.035,.006])**2); history=[]; chain=np.empty((steps,5)); draws=[]; accepted=0; evaluations=0
        for it in range(steps):
            for _ in range(3):
                trial=theta+rng.multivariate_normal(np.zeros(5),proposal); val=target(trial,mu)
                if np.log(rng.random())<val-ll: theta=trial; ll=val; accepted+=1
            nu=chol@rng.normal(size=2*n); level=ll+np.log(rng.random()); angle=rng.uniform(0,2*np.pi); lo=angle-2*np.pi; hi=angle; shifted=mu-mu0
            for _ in range(1000):
                candidate=mu0+shifted*np.cos(angle)+nu*np.sin(angle); val=target(theta,candidate); evaluations+=1
                if val>level: mu=candidate; ll=val; break
                if angle<0: lo=angle
                else: hi=angle
                angle=rng.uniform(lo,hi)
            else: raise RuntimeError('Elliptical slice bracketing failure')
            chain[it]=theta
            if it<burn:
                history.append(theta.copy())
                if it>1000 and (it+1)%500==0: proposal=2.38**2/5*(np.cov(np.asarray(history).T)+np.diag([1e-5,1e-6,1e-6,1e-6,1e-8]))
            if it>=burn and it%100==0: draws.append(mu.copy())
        chains.append(chain); allmu.append(draws); rates.append({'metropolis_acceptance':accepted/(3*steps),'elliptical_evaluations_per_step':evaluations/steps})
        np.save(OUT/f'nonlinear_{case}_chains.npy',np.array(chains)); np.save(OUT/f'nonlinear_{case}_mu.npy',np.array(allmu))
        print('nonlinear',case,'chain',ch+1,'complete',round(time.perf_counter()-begun,1),'s',flush=True)
    chains=np.array(chains)
    mu_saved_indices=list(range(((burn+99)//100)*100,steps,100))
    save(case,'nonlinear',chains,burn,{'steps':steps,'chains':nchains,'adaptation_stops_at_burn':True,'mu_saved_indices':mu_saved_indices,'rates':rates,'diagnostics':diagnostics(chains[:,burn:]),'chain_medians':np.median(chains[:,burn:],axis=1).tolist(),'seconds':time.perf_counter()-begun})

def run_linear(case,steps=12000,walkers=40):
    model=PhysicalModel(case); ndim=5; start=time.perf_counter(); initial=np.array([.1,2.1,-.3,-.2,.1])
    fit=minimize(lambda t:-model(t)[0],initial,method='Nelder-Mead',options={'maxiter':1800}); center=fit.x
    chains=[]; accepts=[]; tau=[]
    for e in range(2):
        np.random.seed(20261021+e*1009); rng=np.random.default_rng(20261021+e*1009); pos=[]
        while len(pos)<walkers:
            t=center+rng.normal(size=ndim)*[.12,.06,.08,.1,.015]
            if np.isfinite(model(t)[0]): pos.append(t)
        sampler=emcee.EnsembleSampler(walkers,ndim,model,vectorize=True); sampler.run_mcmc(np.array(pos),steps,progress=False)
        chains.append(sampler.get_chain()); accepts.append(float(np.mean(sampler.acceptance_fraction))); tau.append(sampler.get_autocorr_time(tol=0).tolist()); print('linear',case,'ensemble',e+1,'complete',flush=True)
    chains=np.array(chains); np.save(OUT/f'linear_{case}_chains.npy',chains)
    save(case,'linear',chains,steps//3,{'steps':steps,'ensembles':2,'walkers':walkers,'acceptance':accepts,'tau':tau,'seconds':time.perf_counter()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['linear','nonlinear'],default='nonlinear'); p.add_argument('--cases',default='base'); a=p.parse_args()
    for case in a.cases.split(','):
        (run_linear if a.kind=='linear' else run_nonlinear)(case)
