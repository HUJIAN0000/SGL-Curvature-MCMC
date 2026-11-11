"""New v1.4.3 assignments and nonlinear sampling under fixed physical priors."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,json,copy,importlib.util,argparse,hashlib
import numpy as np
import pandas as pd
from supplement_sampler import sample_joint

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reproduction'))
from physical_prior_revision import PhysicalModel,LOWER,C
from inference import distance

CASES={'quality025':(.25,.05),'quality035':(.35,.05),
       'tol03':(.30,.03),'tol07':(.30,.07),'reference':(.30,.05),
       'upper5':(.30,.05)}

def make_assignment(case):
    cutoff,tolerance=CASES[case]
    folder=OUT/'sensitivity'/case; folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'codeanddata'/'DES-Dovekie_HD.csv')
    cov=np.load(ROOT/'reproduction'/'des_covariance.npy',mmap_mode='r')
    sigma=np.sqrt(cov.diagonal())
    selected=data.loc[sigma<=cutoff].copy()
    catalogue=folder/'selected_supernovae.csv'; selected.to_csv(catalogue,index=False)
    config=json.loads((ROOT/'reproduction'/'match_quality.json').read_text())
    config['file_paths'].update(tracer=str(catalogue),obj=str(ROOT/'codeanddata'/'130sgls2.CSV'),
                                covariance=str(ROOT/'reproduction'/'des_covariance.npy'))
    config['params'].update(tol=tolerance,wm=.3,output_dir=str(folder/'matching'))
    (folder/'matching_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('cm143_supplement',ROOT/'addcode'/'CosmoMatcherv1.4.3.py')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    result=module.run_config(config)
    meta={'case':case,'sigma_mu_max':cutoff,'relative_distance_tolerance':tolerance,
          'Omega_m_fiducial':.3,'H0_fixed':70.,'Omega_lower':LOWER,
          'gamma_z_bounds':[-.5,.5],'Omega_upper':5. if case=='upper5' else 2.,'selected_sn_rows':len(selected),
          'matched_lenses':result.summary['matched_targets'],
          'software_version':result.summary['version']}
    old=pd.read_csv(ROOT/'reproduction'/'match_quality'/'matched_sgls_milp_dd_des.csv')
    new=pd.read_csv(folder/'matching'/'matched_sgls_milp_dd_des.csv')
    common=sorted(set(old['Lens name']) & set(new['Lens name']))
    a=old.set_index('Lens name').loc[common]; b=new.set_index('Lens name').loc[common]
    same=(a.sn_lens_idx.to_numpy()==b.sn_lens_idx.to_numpy())&(a.sn_source_idx.to_numpy()==b.sn_source_idx.to_numpy())
    meta.update(common_lenses_with_reference=len(common),identical_triplets_with_reference=int(same.sum()),
                lost_lenses=sorted(set(old['Lens name'])-set(new['Lens name'])),
                gained_lenses=sorted(set(new['Lens name'])-set(old['Lens name'])))
    if case=='reference':
        assert len(common)==len(old)==len(new) and same.all(),'Reference reassignment differs from frozen mapping'
    (folder/'design.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('ASSIGNMENT',json.dumps(meta),flush=True)
    return folder

class SupplementModel(PhysicalModel):
    def __init__(self,folder):
        super().__init__('base')
        df=pd.read_csv(Path(folder)/'matching'/'matched_sgls_milp_dd_des.csv')
        self.df=df; self.n=len(df)
        il=df.sn_lens_idx.to_numpy(int); iss=df.sn_source_idx.to_numpy(int)
        assert len(set(np.r_[il,iss]))==2*self.n
        par=np.load(ROOT/'reproduction'/'des_covariance.npy',mmap_mode='r')
        self.cl=par[np.ix_(il,il)].copy(); self.cs=par[np.ix_(iss,iss)].copy(); self.cls=par[np.ix_(il,iss)].copy()
        self.ml=df.sn_lens_MU.to_numpy(); self.ms=df.sn_source_MU.to_numpy()
        self.zl=df.zl.to_numpy(); self.zs=df.zs.to_numpy()
        self.dm_l=np.asarray(5*np.log10((1+df.zl)*distance(df.zl)/((1+df.sn_lens_zHD)*distance(df.sn_lens_zHD))))
        self.dm_s=np.asarray(5*np.log10((1+df.zs)*distance(df.zs)/((1+df.sn_source_zHD)*distance(df.sn_source_zHD))))
        self.cl+=np.diag(self.dm_l**2); self.cs+=np.diag(self.dm_s**2)
        self.delta=df.delta.to_numpy(); self.te=df.thetaE.to_numpy(); self.ta=df.thetaap.to_numpy()
        self.obs=df.sigma_ap.to_numpy(); self.err=df.dsigma_ap.to_numpy()
        self.xl=70/C*10**((self.ml-25)/5)/(1+self.zl)
        self.xs=70/C*10**((self.ms-25)/5)/(1+self.zs)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cases',default='quality025,quality035,tol03,tol07')
    p.add_argument('--steps',type=int,default=100000); p.add_argument('--chains',type=int,default=4)
    p.add_argument('--assignment-only',action='store_true'); a=p.parse_args()
    for case in a.cases.split(','):
        folder=make_assignment(case)
        if a.assignment_only: continue
        prefix=folder/'nonlinear'
        if Path(str(prefix)+'_summary.json').exists():
            print('PRESERVING COMPLETED',case,flush=True); continue
        seed=2026102101+list(CASES).index(case)*100003
        model=SupplementModel(folder)
        if case=='upper5': model.bounds[0]=(LOWER,5.); model.qbound=(LOWER,5.)
        sample_joint(model,prefix,steps=a.steps,nchains=a.chains,seed=seed)

if __name__=='__main__': main()
