"""CPU float64 lens/SN inference with analytic covariance propagation.

q is curvature in the supplied distance-zero-point convention, not an
absolute-calibration-independent Omega_k measurement. No CMB distance prior.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import json,time,argparse
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.integrate import quad
import emcee

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent; DATA=ROOT/'codeanddata'
C=299792.458; A=np.log(10)/5
LABELS=['q','gamma0','gamma_z','beta','scatter']

def distance(z,om=.3):
    return np.array([quad(lambda t:1/np.sqrt(om*(1+t)**3+1-om),0,float(x),epsabs=1e-12)[0] for x in z])
def kernel(x,y,var=1.5,scale=1.2): return var*np.exp(-.5*((x[:,None]-y[None,:])/scale)**2)
def gp_predict(z,optimized=False):
    nodes=np.loadtxt(DATA/'lcparam_full.txt',usecols=(1,4)); zn,mu=nodes.T
    raw=np.loadtxt(DATA/'mag_covmat.txt'); nc=int(raw[0]); cov=raw[1:].reshape(nc,nc)+np.eye(nc)*1e-6
    mean=lambda zz:5*np.log10(zz)+43
    var,scale=1.5,1.2
    if optimized:
        def nll(t):
            k=kernel(zn,zn,*np.exp(t))+cov; f=cho_factor(k); y=mu-mean(zn)
            return .5*y@cho_solve(f,y)+np.log(np.diag(f[0])).sum()
        fit=minimize(nll,np.log([var,scale]),bounds=[(-5,5),(-4,3)])
        var,scale=np.exp(fit.x)
    f=cho_factor(kernel(zn,zn,var,scale)+cov); ks=kernel(z,zn,var,scale)
    pred=mean(z)+ks@cho_solve(f,mu-mean(zn))
    pc=kernel(z,z,var,scale)-ks@cho_solve(f,ks.T); pc=(pc+pc.T)/2
    return pred,pc,{'variance':float(var),'length_scale':float(scale),'optimized':optimized}

def structural(g,d,b):
    xi=g+d-2
    t1=gamma((xi-1)/2)/gamma(xi/2)
    t2=b*gamma((xi+1)/2)/gamma((xi+2)/2)
    t3=gamma(g/2)*gamma(d/2)/(gamma((g-1)/2)*gamma((d-1)/2))
    return (3-d)/((xi-2*b)*(3-xi))*(t1-t2)*t3

class Model:
    def __init__(self,case='base'):
        self.case=case; self.flat=case=='flat'; self.names=LABELS[1:] if self.flat else LABELS
        self.qbound=(-2.,2.) if case!='qwide' else (-5.,5.)
        if case=='cmbcut': self.qbound=(-.1,2.)
        self.gzbound=(-1.,1.) if case=='gzwide' else (-.5,.5)
        self.beta_prior=case=='beta_prior'
        if case.startswith('union'):
            self.df=pd.read_csv(DATA/'130sgls2.CSV')
            self.df=self.df[self.df.zs<=2.262260].reset_index(drop=True)
            if case!='union_all':
                chosen=pd.read_csv(OUT/'match_quality'/'matched_sgls_milp_dd_des.csv')['Lens name']
                self.df=self.df.set_index('Lens name').loc[chosen].reset_index()
            n=len(self.df); zz=np.r_[self.df.zl,self.df.zs]
            mu,cov,self.gp=gp_predict(zz,case=='union_opt')
            self.ml,self.ms=mu[:n],mu[n:]; self.cl=cov[:n,:n]; self.cs=cov[n:,n:]; self.cls=cov[:n,n:]
            self.dm_l=np.zeros(n); self.dm_s=np.zeros(n)
        else:
            folder='match_quality'
            if case=='quality035': folder='match_quality035'
            if case=='tol03': folder='match_tol03'
            if case=='om04': folder='match_om04'
            self.df=pd.read_csv(OUT/folder/'matched_sgls_milp_dd_des.csv')
            df=self.df; par=np.load(OUT/'des_covariance.npy'); il=df.sn_lens_idx.to_numpy(int); iss=df.sn_source_idx.to_numpy(int)
            self.cl=par[np.ix_(il,il)].copy(); self.cs=par[np.ix_(iss,iss)].copy(); self.cls=par[np.ix_(il,iss)].copy()
            self.ml=df.sn_lens_MU.to_numpy(); self.ms=df.sn_source_MU.to_numpy()
            self.dm_l=np.asarray(5*np.log10((1+df.zl)*distance(df.zl)/((1+df.sn_lens_zHD)*distance(df.sn_lens_zHD))))
            self.dm_s=np.asarray(5*np.log10((1+df.zs)*distance(df.zs)/((1+df.sn_source_zHD)*distance(df.sn_source_zHD))))
            if case=='transfer': self.ml+=self.dm_l; self.ms+=self.dm_s
            if case!='nomismatch':
                self.cl+=np.diag(self.dm_l**2); self.cs+=np.diag(self.dm_s**2)
        df=self.df; self.n=len(df); self.zl=df.zl.to_numpy(); self.zs=df.zs.to_numpy()
        self.delta=df.delta.to_numpy()+(.05 if case=='delta_plus' else -.05 if case=='delta_minus' else 0)
        self.te=df.thetaE.to_numpy(); self.ta=df.thetaap.to_numpy(); self.obs=df.sigma_ap.to_numpy(); self.err=df.dsigma_ap.to_numpy()
        self.xl=70/C*10**((self.ml-25)/5)/(1+self.zl); self.xs=70/C*10**((self.ms-25)/5)/(1+self.zs)
        if case=='diagonal': self.cl=np.diag(self.cl.diagonal()); self.cs=np.diag(self.cs.diagonal()); self.cls*=0
        self.bounds=([self.qbound] if not self.flat else [])+[(1.1,2.8),self.gzbound,(-.8,.8),(0,.3)]

    def predict(self,theta,return_parts=False):
        tt=np.atleast_2d(theta)
        if self.flat: tt=np.c_[np.zeros(len(tt)),tt]
        q,g0,gz,b,s=tt.T; q=q[:,None]; b=b[:,None]
        g=g0[:,None]+gz[:,None]*self.zl; d=self.delta[None,:]
        al=1+q*self.xl**2; ass=1+q*self.xs**2
        with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
            al=np.sqrt(al); ass=np.sqrt(ass); r=al-self.xl/self.xs*ass
            f=structural(g,d,b); factor=C*C/(2*np.sqrt(np.pi))*self.te*np.pi/648000
            sig=np.sqrt(factor/r*f*(self.ta/self.te)**(2-g))
            jl=-.5*sig/r*A*(q*self.xl**2/al-self.xl/self.xs*ass)
            js=-.5*sig/r*A*self.xl/(self.xs*ass)
        valid=(np.all(np.isfinite(sig),axis=1)&np.all(r>0,axis=1)&np.all(f>0,axis=1)&np.all((g>1.05)&(g<2.95),axis=1)&np.all((g+d-2>1)&(g+d-2<3)&(g+d-2>2*b),axis=1))
        if return_parts: return sig,jl,js,valid
        return sig,valid

    def __call__(self,theta):
        t=np.atleast_2d(theta); result=np.full(len(t),-np.inf)
        bounds=np.array(self.bounds); valid=np.all((t>bounds[:,0])&(t<bounds[:,1]),axis=1)
        ids=np.flatnonzero(valid)
        if not len(ids): return result
        tt=t[ids]; sig,jl,js,physical=self.predict(tt,True)
        cov=jl[:,:,None]*self.cl*jl[:,None,:]+js[:,:,None]*self.cs*js[:,None,:]+jl[:,:,None]*self.cls*js[:,None,:]+js[:,:,None]*self.cls.T*jl[:,None,:]
        variance=self.err**2+(sig*tt[:,-1,None])**2
        if self.case=='lens_noise':
            # Illustrative independent scatters, not per-lens measured errors.
            full=tt if not self.flat else np.c_[np.zeros(len(tt)),tt]
            g=full[:,1,None]+full[:,2,None]*self.zl; b=full[:,3,None]; eps=1e-4
            fs=structural(g,self.delta,b)
            dd=.5*sig*(np.log(structural(g,self.delta+eps,b))-np.log(structural(g,self.delta-eps,b)))/(2*eps)
            db=.5*sig*(np.log(structural(g,self.delta,b+eps))-np.log(structural(g,self.delta,b-eps)))/(2*eps)
            variance+=(.05*dd)**2+(.1*db)**2+(sig*.03)**2+(sig*.025)**2
        j=np.arange(self.n); cov[:,j,j]+=variance
        for k in np.flatnonzero(physical):
            try:
                f=cho_factor(cov[k],lower=True,check_finite=False); residual=self.obs-sig[k]
                result[ids[k]]=-.5*residual@cho_solve(f,residual,check_finite=False)-np.log(np.diag(f[0])).sum()-.5*self.n*np.log(2*np.pi)
                if self.beta_prior: result[ids[k]]-=.5*((tt[k,-2]-.18)/.13)**2
            except (ValueError,np.linalg.LinAlgError): pass
        return result

def split_rhat(chain):
    # Different ensembles are independent. Walkers are coupled, so use ensemble
    # mean series; this is explicitly an ensemble-mean convergence diagnostic.
    m=chain.mean(axis=2); half=m.shape[1]//2
    x=np.concatenate([m[:,:half],m[:,-half:]],axis=0)
    within=np.var(x,axis=1,ddof=1).mean(axis=0); between=half*np.var(x.mean(axis=1),axis=0,ddof=1)
    return np.sqrt(((half-1)/half*within+between/half)/within)

def run(case,steps=10000,ensembles=2,walkers=40):
    model=Model(case); ndim=len(model.names); chains=[]; logps=[]; accepts=[]; taus=[]
    initial=np.array([.1,2.1,-.25,-.2,.1]); initial=initial[1:] if model.flat else initial
    fit=minimize(lambda x:-model(x)[0],initial,method='Nelder-Mead',options={'maxiter':1800,'xatol':1e-7})
    center=fit.x if np.isfinite(model(fit.x)[0]) else initial
    begun=time.perf_counter()
    for e in range(ensembles):
        np.random.seed(20261021+e*1009); rng=np.random.default_rng(20261021+e*1009)
        scale=np.array([.15,.06,.08,.1,.015]); scale=scale[1:] if model.flat else scale
        positions=[]
        while len(positions)<walkers:
            p=center+scale*rng.normal(size=ndim)
            if np.isfinite(model(p)[0]): positions.append(p)
        sampler=emcee.EnsembleSampler(walkers,ndim,model,vectorize=True)
        for i,state in enumerate(sampler.sample(np.array(positions),iterations=steps)):
            if (i+1)%2000==0: print(case,'ensemble',e+1,'step',i+1,'seconds',round(time.perf_counter()-begun,1),flush=True)
        chains.append(sampler.get_chain()); logps.append(sampler.get_log_prob()); accepts.append(sampler.acceptance_fraction); taus.append(sampler.get_autocorr_time(tol=0))
        np.save(OUT/f'{case}_chains.npy',np.array(chains)); np.save(OUT/f'{case}_logp.npy',np.array(logps))
    chains=np.array(chains); burn=steps//3; kept=chains[:,burn:]; flat=kept.reshape(-1,ndim)
    quant=np.percentile(flat,[15.865,50,84.135],axis=0)
    result={'case':case,'N':model.n,'labels':model.names,'bounds':model.bounds,'steps':steps,'ensembles':ensembles,'walkers':walkers,'burn':burn,'seed_base':20261021,'quantiles':quant.tolist(),'tau_per_ensemble':np.array(taus).tolist(),'mean_acceptance':np.mean(accepts,axis=1).tolist(),'ensemble_mean_split_rhat':split_rhat(kept).tolist(),'retained_steps_over_max_tau':((steps-burn)/np.max(taus,axis=0)).tolist(),'seconds':time.perf_counter()-begun,'ensemble_medians':[np.median(c.reshape(-1,ndim),axis=0).tolist() for c in kept], 'corr':np.corrcoef(flat.T).tolist()}
    for i,n in enumerate(model.names):
        lo,hi=model.bounds[i]; result.setdefault('boundary_mass_5pct',{})[n]=[float(np.mean(flat[:,i]<lo+.05*(hi-lo))),float(np.mean(flat[:,i]>hi-.05*(hi-lo)))]
    (OUT/f'{case}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cases',default='base'); p.add_argument('--steps',type=int,default=10000); p.add_argument('--ensembles',type=int,default=2); p.add_argument('--walkers',type=int,default=40); p.add_argument('--benchmark',action='store_true'); args=p.parse_args()
    if args.benchmark:
        m=Model(); points=np.tile([.1,2.1,-.25,-.2,.1],(40,1)); t=time.perf_counter()
        for _ in range(100): value=m(points)
        print('4000 evaluations seconds',time.perf_counter()-t,'logp',value[0])
    else:
        for case in args.cases.split(','): run(case,args.steps,args.ensembles,args.walkers)
