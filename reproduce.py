"""Portable command-line entry point for the SGL-Curvature-MCMC reproducibility package.

All sampling/postprocessing commands operate on a fresh workspace in outputs/.
Frozen scientific modules and published numerical reference files are preserved.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, shutil, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parent

def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()
def registry():
    with (ROOT/'provenance/result_registry.csv').open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))
def check(condition, message):
    if not condition: raise RuntimeError(message)

def prepare(root=ROOT):
    import numpy as np
    import pandas as pd
    from scipy.linalg import cho_factor, cho_solve
    raw=np.load(root/'codeanddata/covtot_inv_000.npz')
    n=int(raw['nsn'][0]); inv=np.zeros((n,n),dtype=np.float64)
    inv[np.triu_indices(n)]=raw['cov']
    inv=inv+inv.T-np.diag(inv.diagonal())
    covariance=cho_solve(cho_factor(inv),np.eye(n))
    covariance=(covariance+covariance.T)/2
    path=root/'reproduction/des_covariance.npy'
    if path.exists():
        check(np.allclose(np.load(path),covariance,atol=1e-10,rtol=1e-11),'Existing full covariance differs from inverse reconstruction')
    else: np.save(path,covariance)
    sn=pd.read_csv(root/'codeanddata/DES-Dovekie_HD.csv')
    selected=sn.loc[np.sqrt(covariance.diagonal())<=.30]
    check(n==len(sn)==1820 and len(selected)==1641,'Unexpected DES row selection')
    np.linalg.cholesky(covariance)
    report={'DES_rows':n,'eligible_rows':len(selected),'dtype':str(covariance.dtype),
        'inverse_reconstruction_inf_residual':float(np.linalg.norm(inv@covariance-np.eye(n),ord=np.inf)),
        'covariance_file':'reproduction/des_covariance.npy','method':'Invert full precision matrix, then select covariance rows; never invert a selected precision block.'}
    write_json(root/'outputs/preparation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report

def restore(archive):
    records=load(ROOT/'provenance/chain_files.json')['files']
    wanted={r['path']:r for r in records}
    # Validate every member and every digest before changing the repository.
    with zipfile.ZipFile(archive) as z:
        check(len(z.namelist())==len(set(z.namelist())),'Duplicate archive members')
        check(set(z.namelist())==set(wanted)|{'provenance/chain_files.json'},'Unexpected/missing archive members')
        for name,r in wanted.items():
            p=PurePosixPath(name)
            check(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'Unsafe archive path')
            check(z.getinfo(name).file_size==r['bytes'],'Archive size mismatch: '+name)
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(4<<20),b''): h.update(b)
            check(h.hexdigest()==r['sha256'],'Archive digest mismatch: '+name)
            dest=ROOT/name
            check(not dest.exists() or sha(dest)==r['sha256'],'Refusing to overwrite different existing array: '+name)
        check(load(ROOT/'provenance/chain_files.json')==json.loads(z.read('provenance/chain_files.json')),'Archive manifest mismatch')
        for name in wanted:
            dest=ROOT/name
            if dest.exists(): continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=dest.with_suffix(dest.suffix+'.partial')
            with z.open(name) as source,temp.open('wb') as target: shutil.copyfileobj(source,target)
            temp.replace(dest)
    print('Restored/verified',len(wanted),'saved arrays; existing identical files preserved.',flush=True)

def science_checks():
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT/'reproduction'))
    from physical_prior_revision import PhysicalModel,LOWER,ROOT as model_root
    from latent_validation import loglike
    check(model_root.resolve()==ROOT.resolve(),'Model resolved outside this repository')
    model=PhysicalModel('base')
    check(model.n==88 and model.gzbound==(-.5,.5),'Primary model selection or prior differs')
    check(abs(LOWER+.09405370006986057)<1e-14,'CMB geometric boundary differs')
    ids=np.r_[model.df.sn_lens_idx,model.df.sn_source_idx].astype(int)
    check(len(np.unique(ids))==176,'SN covariance rows reused')
    parent=np.load(ROOT/'reproduction/des_covariance.npy',mmap_mode='r')
    cov=np.block([[model.cl,model.cls],[model.cls.T,model.cs]])
    expected=parent[np.ix_(ids,ids)]+np.diag(np.r_[model.dm_l**2,model.dm_s**2])
    check(np.allclose(cov,expected,atol=1e-12,rtol=0),'Joint SN factor differs from full covariance plus mismatch envelope')
    np.linalg.cholesky(cov)
    theta=np.array([.1,2.07,-.25,-.15,.1]); mu=np.r_[model.ml,model.ms]
    ll=loglike(theta,mu,model.zl,model.zs,model.te,model.ta,model.delta,model.obs,model.err,np.asarray(model.bounds),False,False)
    check(np.isfinite(ll) and np.isfinite(model(theta)[0]),'Nonfinite likelihood at reference point')
    sn=pd.read_csv(ROOT/'codeanddata/DES-Dovekie_HD.csv')
    selected=pd.read_csv(ROOT/'reproduction/des_quality_catalogue.csv')
    check(np.array_equal(selected.Index.to_numpy(),sn.loc[np.sqrt(parent.diagonal())<=.3,'Index'].to_numpy()),'Frozen quality catalogue differs')
    return {'primary_lenses':88,'distinct_SN_rows':176,'H0_fixed':70,'Omega_lower':LOWER,
        'gamma_z_bounds':[-.5,.5],'joint_loglike_at_reference_point':float(ll),'model_resolves_inside_repository':True}

def verify(with_chains=False):
    import numpy as np
    manifest=load(ROOT/'provenance/repository_manifest.json')['files']
    for row in manifest:
        path=ROOT/row['path']
        check(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Tracked file missing/changed: '+row['path'])
    check((ROOT/'reproduction/des_covariance.npy').is_file(),'Run python reproduce.py prepare first')
    result={'repository_files_verified':len(manifest),'scientific_model':science_checks(),'saved_runs':[]}
    rows=registry(); check(len(rows)==28,'Expected 28 current saved fits')
    for row in rows:
        s=load(ROOT/row['summary']); quant=np.asarray(s['quantiles'])
        check(quant.shape==(3,5) and np.isfinite(quant).all(),'Invalid quantiles: '+row['summary'])
        check(abs(quant[1,0]-float(row['Omega_median']))<1e-12,'Registry median mismatch')
        record={'summary':row['summary'],'N':s['N'],'Omega_quantiles':quant[:,0].tolist(),
            'gamma_z_quantiles':quant[:,2].tolist(),'chains_verified':False}
        if with_chains:
            path=ROOT/row['parameter_chain']; check(path.exists(),'Restore chain archive first: '+row['parameter_chain'])
            chain=np.load(path,mmap_mode='r'); check(np.isfinite(chain).all(),'Nonfinite chain')
            flat=chain[:,s['burn']:].reshape(-1,5)
            measured=np.percentile(flat,[15.865,50,84.135],axis=0)
            check(np.allclose(measured,quant,atol=2e-6,rtol=0),'Stored quantiles disagree with chain: '+row['summary'])
            record.update(chains_verified=True,shape=list(chain.shape))
            if row['latent_chain']:
                mu=np.load(ROOT/row['latent_chain'],mmap_mode='r')
                indices=np.asarray(s.get('mu_saved_indices',np.arange(((s['burn']+99)//100)*100,s['steps'],100)),dtype=int)
                check(mu.shape==(chain.shape[0],len(indices),2*s['N']) and np.isfinite(mu).all(),'Latent state shape/alignment differs')
                check(np.all(indices>=s['burn']) and np.all(indices<chain.shape[1]) and np.all(np.diff(indices)==100),'Invalid aligned latent indices')
                record['aligned_latent_states']=int(mu.shape[0]*mu.shape[1])
        result['saved_runs'].append(record)
    if with_chains:
        for row in load(ROOT/'provenance/chain_files.json')['files']:
            path=ROOT/row['path']
            check(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Array digest differs: '+row['path'])
        sys.path.insert(0,str(ROOT/'reproduction'))
        from physical_prior_revision import PhysicalModel
        from latent_validation import loglike
        m=PhysicalModel(); s=load(ROOT/'reproduction_physical/nonlinear_base_summary.json')
        chain=np.load(ROOT/'reproduction_physical/nonlinear_base_chains.npy',mmap_mode='r')
        mu=np.load(ROOT/'reproduction_physical/nonlinear_base_mu.npy',mmap_mode='r')
        indices=np.arange(((s['burn']+99)//100)*100,s['steps'],100)
        for c in range(4):
            for j in (0,111,333,665):
                ll=loglike(np.asarray(chain[c,indices[j]]),np.asarray(mu[c,j]),m.zl,m.zs,m.te,m.ta,m.delta,m.obs,m.err,np.asarray(m.bounds),False,False)
                check(np.isfinite(ll),'Nonfinite correctly aligned primary joint state')
        result['primary_aligned_joint_likelihood_checks']=16
        result['chain_arrays_verified']=len(load(ROOT/'provenance/chain_files.json')['files'])
    result['chain_check_status']='all 28 saved fits verified' if with_chains else 'not requested; summaries checked only'
    write_json(ROOT/'outputs'/('verification_chains.json' if with_chains else 'verification_quick.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='saved_runs'},indent=2),flush=True)

def workspace(label, all_results=False, primary_results=False):
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest=ROOT/'outputs'/(label+'_'+stamp)
    dest.mkdir(parents=True,exist_ok=False)
    for name in ['codeanddata','addcode','reproduction']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    supp=dest/'reproduction_supplement'; supp.mkdir()
    for p in (ROOT/'reproduction_supplement').glob('*.py'): shutil.copy2(p,supp/p.name)
    (dest/'reproduction_physical').mkdir()
    if all_results or primary_results:
        for p in (ROOT/'reproduction_physical').glob('*'):
            if p.is_file() and p.suffix in {'.json','.csv','.npy'}: shutil.copy2(p,dest/'reproduction_physical'/p.name)
        check((dest/'reproduction_physical/nonlinear_base_mu.npy').exists(),'Restore chain archive before this operation')
    if all_results:
        for name in ['diagnostics','sensitivity','simulations','exact_redshift']:
            shutil.copytree(ROOT/'reproduction_supplement'/name,supp/name)
    print('WORKSPACE='+str(dest),flush=True)
    return dest

def execute_script(root,name,args=()):
    subprocess.run([sys.executable,str(root/'reproduction_supplement'/name),*map(str,args)],cwd=root,check=True)

def matching():
    import numpy as np
    import pandas as pd
    dest=workspace('matching')
    cfg=load(ROOT/'reproduction/match_quality.json')
    cfg['file_paths']={k:str(dest/v) for k,v in cfg['file_paths'].items()}
    cfg['params']['output_dir']=str(dest/'generated_matching')
    spec=importlib.util.spec_from_file_location('cm143',ROOT/'addcode/CosmoMatcherv1.4.3.py')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    result=module.run_config(cfg)
    old=pd.read_csv(ROOT/'reproduction/match_quality/matched_sgls_milp_dd_des.csv').set_index('Lens name').sort_index()
    new=pd.read_csv(dest/'generated_matching/matched_sgls_milp_dd_des.csv').set_index('Lens name').sort_index()
    same=old.index.equals(new.index) and np.array_equal(old[['sn_lens_idx','sn_source_idx']],new[['sn_lens_idx','sn_source_idx']])
    check(same,'Assignment differs from frozen reference; inspect solver/version and ties before changing data')
    report={'matched_lenses':len(new),'identical_complete_triplets':len(new),'version':result.summary['version']}
    write_json(dest/'matching_verification.json',report); print(json.dumps(report),flush=True)

def sample(args):
    dest=workspace('sample_'+args.kind)
    check(args.steps>=600 and args.chains>=2,'Use >=600 steps and >=2 chains for an execution check; use published lengths for inference')
    program=('import sys; sys.path.insert(0,sys.argv[1]); '
        'from physical_prior_revision import run_nonlinear,run_linear; '
        'kind,cases,steps,chains,walkers=sys.argv[2:]; '
        '[(run_nonlinear(c,int(steps),int(chains)) if kind=="nonlinear" else run_linear(c,int(steps),int(walkers))) for c in cases.split(",")]')
    subprocess.run([sys.executable,'-c',program,str(dest/'reproduction'),args.kind,args.cases,
        str(args.steps),str(args.chains),str(args.walkers)],cwd=dest,check=True)
    write_json(dest/'run_request.json',vars(args)|{'execution_check_only':args.steps<12000})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare'); p=sub.add_parser('verify'); p.add_argument('--chains',action='store_true')
    p=sub.add_parser('restore'); p.add_argument('archive',type=Path)
    sub.add_parser('matching')
    p=sub.add_parser('sample'); p.add_argument('--kind',choices=['nonlinear','linear'],default='nonlinear')
    p.add_argument('--cases',default='base'); p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--chains',type=int,default=4); p.add_argument('--walkers',type=int,default=40)
    for name,default,steps in [('sensitivity','quality025,quality035,tol03,tol07,upper5',100000),
                               ('exact-redshift','signed_envelope,signed_no_envelope,gp_log,gp_fid',100000),
                               ('simulate','constant,evolving',60000)]:
        p=sub.add_parser(name); p.add_argument('--cases',default=default)
        p.add_argument('--steps',type=int,default=steps); p.add_argument('--chains',type=int,default=4)
    sub.add_parser('diagnostics'); sub.add_parser('figures')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='restore': restore(args.archive)
    elif args.command=='verify': verify(args.chains)
    elif args.command=='matching': matching()
    elif args.command=='sample': sample(args)
    elif args.command in ['sensitivity','simulate','exact-redshift']:
        dest=workspace(args.command,primary_results=args.command=='exact-redshift')
        script={'sensitivity':'run_sensitivity.py','simulate':'simulation_checks.py','exact-redshift':'exact_redshift_checks.py'}[args.command]
        execute_script(dest,script,['--cases',args.cases,'--steps',args.steps,'--chains',args.chains])
    elif args.command=='diagnostics':
        dest=workspace('diagnostics',all_results=True)
        for script in ['supplement_diagnostics.py','analytic_source_trend_checks.py','sensitivity_predictive_checks.py']:
            execute_script(dest,script)
        execute_script(dest,'simulation_checks.py',['--source-trend-only'])
        execute_script(dest,'validate_release.py'); execute_script(dest,'validate_exact_redshift.py')
    elif args.command=='figures':
        subprocess.run([sys.executable,str(ROOT/'render_results.py')],cwd=ROOT,check=True)

if __name__=='__main__': main()
