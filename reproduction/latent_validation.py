"""Nonlinear validation: elliptical-slice SN distances + adaptive Metropolis.

The SN Gaussian factor retains all selected covariance and the explicitly
assumed independent mismatch envelope; no Jacobian approximation is used.
Proposal adaptation stops at burn-in. Four independent scalar chains.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,time,argparse,math
import numpy as np
from numba import njit
from inference import Model,OUT,C,A
from emcee.autocorr import integrated_time

@njit(cache=True)
def sf(g,d,b):
    xi=g+d-2
    return (3-d)/((xi-2*b)*(3-xi))*(math.gamma((xi-1)/2)/math.gamma(xi/2)-b*math.gamma((xi+1)/2)/math.gamma((xi+2)/2))*math.gamma(g/2)*math.gamma(d/2)/(math.gamma((g-1)/2)*math.gamma((d-1)/2))

@njit(cache=True)
def loglike(t,mu,zl,zs,te,ta,delta,obs,err,bounds,lens_noise,beta_prior):
    q,g0,gz,b,s=t; n=len(zl)
    for k in range(5):
        if not bounds[k,0]<t[k]<bounds[k,1]: return -np.inf
    result=0.
    for i in range(n):
        g=g0+gz*zl[i]; d=delta[i]; xi=g+d-2
        if not (1.05<g<2.95 and 1<xi<3 and xi>2*b): return -np.inf
        xl=70/C*10**((mu[i]-25)/5)/(1+zl[i]); xs=70/C*10**((mu[n+i]-25)/5)/(1+zs[i])
        al=1+q*xl*xl; ass=1+q*xs*xs
        if al<=0 or ass<=0: return -np.inf
        r=math.sqrt(al)-xl/xs*math.sqrt(ass)
        if r<=0: return -np.inf
        t1=math.gamma((xi-1)/2)/math.gamma(xi/2); t2=b*math.gamma((xi+1)/2)/math.gamma((xi+2)/2)
        t3=math.gamma(g/2)*math.gamma(d/2)/(math.gamma((g-1)/2)*math.gamma((d-1)/2))
        f=(3-d)/((xi-2*b)*(3-xi))*(t1-t2)*t3
        ss=C*C/(2*math.sqrt(math.pi))*te[i]*math.pi/648000/r*f*(ta[i]/te[i])**(2-g)
        if ss<=0 or not math.isfinite(ss): return -np.inf
        sigma=math.sqrt(ss); variance=err[i]**2+ss*s*s
        if lens_noise:
            eps=1e-4
            dd=.5*sigma*(math.log(sf(g,d+eps,b))-math.log(sf(g,d-eps,b)))/(2*eps)
            db=.5*sigma*(math.log(sf(g,d,b+eps))-math.log(sf(g,d,b-eps)))/(2*eps)
            variance+=(.05*dd)**2+(.1*db)**2+(.03*sigma)**2+(.025*sigma)**2
        result-=.5*((obs[i]-sigma)**2/variance+math.log(2*math.pi*variance))
    if beta_prior: result-=.5*((b-.18)/.13)**2
    return result

def diagnostics(chains):
    n=chains.shape[1]//2; x=np.concatenate([chains[:,:n],chains[:,-n:]],axis=0)
    w=x.var(axis=1,ddof=1).mean(axis=0); b=n*x.mean(axis=1).var(axis=0,ddof=1)
    r=np.sqrt(((n-1)*w+b)/n/w)
    tau=[]
    for chain in chains: tau.append(integrated_time(chain[:,None,:],tol=0).tolist())
    ess=chains.shape[0]*chains.shape[1]/np.max(tau,axis=0)
    return {'split_rhat':r.tolist(),'tau':tau,'conservative_ess':ess.tolist()}

def run(case='base',steps=100000,nchains=4):
    model=Model(case); n=model.n; mu0=np.r_[model.ml,model.ms]
    cov=np.block([[model.cl,model.cls],[model.cls.T,model.cs]]); chol=np.linalg.cholesky(cov+np.eye(2*n)*1e-13)
    args=(model.zl,model.zs,model.te,model.ta,model.delta,model.obs,model.err,np.array(model.bounds),case=='lens_noise',case=='beta_prior')
    burn=steps//3; allchains=[]; allmu=[]; rates=[]; start=time.perf_counter()
    for c in range(nchains):
        rng=np.random.default_rng(94122+c*551); t=np.array([.1 if case=='cmbcut' else -.2,2.07,-.25,-.15,.1])+rng.normal(size=5)*[.05 if case=='base' else .02,.01,.01,.01,.005]
        mu=mu0.copy(); ll=loglike(t,mu,*args)
        proposal=np.diag(np.array([.04,.018,.035,.035,.006])**2); history=[]; chain=np.empty((steps,5)); mus=[]; accepted=0; attempted=0; estries=0
        for it in range(steps):
            for _ in range(3):
                tt=t+rng.multivariate_normal(np.zeros(5),proposal); val=loglike(tt,mu,*args); attempted+=1
                if np.log(rng.random())<val-ll: t=tt; ll=val; accepted+=1
            nu=chol@rng.normal(size=2*n); level=ll+np.log(rng.random()); angle=rng.uniform(0,2*np.pi); lo=angle-2*np.pi; hi=angle
            shifted=mu-mu0
            for j in range(1000):
                proposed=mu0+shifted*np.cos(angle)+nu*np.sin(angle); val=loglike(t,proposed,*args); estries+=1
                if val>level: mu=proposed; ll=val; break
                if angle<0: lo=angle
                else: hi=angle
                angle=rng.uniform(lo,hi)
            else: raise RuntimeError('elliptical slice failed to bracket accepted state')
            chain[it]=t
            if it<burn:
                history.append(t.copy())
                if it>1000 and (it+1)%500==0:
                    proposal=2.38**2/5*(np.cov(np.array(history).T)+np.diag([1e-5,1e-6,1e-6,1e-6,1e-8]))
            if it>=burn and it%100==0: mus.append(mu.copy())
            if (it+1)%20000==0: print('latent',case,'chain',c+1,'step',it+1,'seconds',round(time.perf_counter()-start,1),flush=True)
        allchains.append(chain); allmu.append(mus); rates.append({'metropolis_acceptance':accepted/attempted,'elliptical_evaluations_per_step':estries/steps})
        np.save(OUT/f'latent_{case}_chains.npy',np.array(allchains)); np.save(OUT/f'latent_{case}_mu.npy',np.array(allmu))
    kept=np.array(allchains)[:,burn:]; flat=kept.reshape(-1,5)
    result={'method':'elliptical slice distances / adaptive Metropolis parameters','case':case,'N':n,'labels':model.names,'bounds':model.bounds,'steps':steps,'chains':nchains,'burn':burn,'adaptation_stops_at_burn':True,'quantiles':np.percentile(flat,[15.865,50,84.135],axis=0).tolist(),'chain_medians':np.median(kept,axis=1).tolist(),'rates':rates,'diagnostics':diagnostics(kept),'seconds':time.perf_counter()-start,'corr':np.corrcoef(flat.T).tolist(),'p_q_ge_zero':float(np.mean(flat[:,0]>=0))}
    (OUT/f'latent_{case}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--case',default='base'); p.add_argument('--steps',type=int,default=100000); p.add_argument('--chains',type=int,default=4); a=p.parse_args()
    for case in a.case.split(','): run(case,a.steps,a.chains)
