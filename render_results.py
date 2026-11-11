"""Regenerate six scientific figures and an all-fit numerical table.

Uses saved draws/diagnostic tables; never reruns inference or edits paper/.
The rendering code is separated from historical manuscript-editing scripts.
"""
from pathlib import Path
import csv,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/figures'; OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'reproduction'),str(ROOT/'reproduction_supplement'),str(ROOT/'addcode')]
from physical_prior_revision import LOWER,PhysicalModel
from inference import distance
from supplement_diagnostics import predict_joint

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sample(prefix):
    s=load(Path(str(prefix)+'_summary.json'))
    c=np.load(str(prefix)+'_chains.npy',mmap_mode='r')
    return np.asarray(c[:,s['burn']:]).reshape(-1,5)
def saved(fig,name):
    fig.tight_layout(); fig.savefig(OUT/(name+'.pdf')); fig.savefig(OUT/(name+'.png'),dpi=150); plt.close(fig)
def physical():
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for kind,case,label,color in [('nonlinear','base','Primary: central CMB bound','#285d9b'),
        ('nonlinear','soft07','CMB distance error included','#c45e36'),('linear','union_same','Union3 GP, same lenses','#388567')]:
        s=sample(ROOT/'reproduction_physical'/f'{kind}_{case}')
        for ax,i in zip(axes,[0,2]):
            x=np.sort(s[:,i]); ax.plot(x,np.arange(1,len(x)+1)/len(x),color=color,lw=1.5,label=label)
    axes[0].axvline(LOWER,color='.4',ls='--',lw=1); axes[0].axvline(0,color='.6',ls=':',lw=1)
    axes[0].set(xlabel=r'$\Omega_k$ (fixed $H_0=70$ convention)',ylabel='Marginal posterior CDF',xlim=(-.15,1.6))
    axes[1].set(xlabel=r'$\gamma_z$',ylabel='Marginal posterior CDF',xlim=(-.5,.5))
    axes[0].legend(frameon=False,fontsize=8); saved(fig,'fig_physical_posteriors')
    import cosmo_tools
    from getdist import MCSamples
    bounds=load(ROOT/'reproduction_physical/nonlinear_base_summary.json')['bounds']
    def factory(s,labels,label=None,**kwargs):
        return MCSamples(samples=s,names=[f'p{i}' for i in range(5)],labels=labels,label=label,
            ranges={f'p{i}':b for i,b in enumerate(bounds)},settings={'smooth_scale_1D':.5,'smooth_scale_2D':.7})
    cosmo_tools._make_mcsamples=factory
    cosmo_tools.plot_getdist_comparison([sample(ROOT/'reproduction_physical/nonlinear_base'),sample(ROOT/'reproduction_physical/linear_union_same')[::4]],
        [r'\Omega_k',r'\gamma_0',r'\gamma_z',r'\beta',r'f_{\rm int}'],legend_labels=['DES: physical priors','Union3 GP: same physical priors'],
        colors=['#285d9b','#c45e36'],filename=str(OUT/'fig_triangle.pdf'))
    plt.close('all')
def matching():
    old=pd.read_csv(ROOT/'codeanddata/matched_sgls_milp_dd.csv')
    new=pd.read_csv(ROOT/'reproduction/match_quality/matched_sgls_milp_dd_des.csv')
    allnew=pd.read_csv(ROOT/'reproduction/match_all/matched_sgls_milp_dd_des.csv')
    parent=pd.read_csv(ROOT/'codeanddata/130sgls2.CSV')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for df,label,color in [(old,'Original (92)','#8f6b40'),(allnew,'Updated, all (92)','#c45e36'),(new,'Updated, cut (88)','#285d9b')]:
        vals=[]
        for z,zn in [('zl','sn_lens_zHD'),('zs','sn_source_zHD')]: vals.extend(abs(distance(df[zn])-distance(df[z]))/distance(df[z]))
        x=np.sort(vals); axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,lw=1.7)
    axes[0].set(xlabel='Absolute relative radial-distance mismatch',ylabel='Endpoint empirical CDF',xlim=(0,.051),ylim=(0,1.02))
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].scatter(parent.zl,parent.zs,s=24,facecolors='none',edgecolors='.65',label='Parent (130)')
    axes[1].scatter(new.zl,new.zs,s=14,color='#285d9b',label='Selected (88)')
    axes[1].set(xlabel='Lens redshift',ylabel='Source redshift'); axes[1].legend(frameon=False,fontsize=9)
    saved(fig,'fig_matching')
def source():
    table=pd.read_csv(ROOT/'reproduction_supplement/diagnostics/source_redshift_binned_residuals.csv')
    z=table.z_mean.to_numpy(); med=table.observed_mean_r_q50.to_numpy()
    fig,ax=plt.subplots(figsize=(6.8,3.2))
    ax.fill_between(z,table['replicated_mean_r_q2.275'],table['replicated_mean_r_q97.725'],color='#ccdce8',label='replicated 95.45%')
    ax.errorbar(z,med,yerr=[med-table['observed_mean_r_q15.865'],table['observed_mean_r_q84.135']-med],fmt='o-',color='#93412e',capsize=3,label='observed posterior median and 68.27%')
    ax.axhline(0,color='.4',lw=.8); ax.set(xlabel=r'source redshift $z_s$ (bin mean)',ylabel='mean standardized velocity residual',xlim=(.32,1.10),ylim=(-.75,.75))
    ax.legend(fontsize=8,loc='upper right'); saved(fig,'fig_source_trend')
def mocks():
    rows=load(ROOT/'reproduction_supplement/simulations/aggregate_summary.json')['results']
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.8))
    for ax,p in zip(axes,[0,2]):
        for j,r in enumerate(rows):
            q=np.array(r['quantiles_95_68'])[:,p]; color='#2166ac' if r['name'].startswith('constant') else '#b35806'
            ax.plot(q[[0,4]],[j,j],color=color,lw=1.2); ax.plot(q[[1,3]],[j,j],color=color,lw=4)
            ax.plot(q[2],j,'o',color=color,ms=5); ax.plot(r['truth'][p],j,marker='|',color='black',ms=15,mew=1.6)
        ax.set_yticks(np.arange(len(rows)),[r['name'].replace('_',' ') for r in rows]); ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2); ax.set_xlabel(r'$\Omega_k$' if p==0 else r'$\gamma_z$'); ax.axvline(LOWER if p==0 else -.5,color='.5',ls=':',lw=1)
    axes[0].set_title('Curvature'); axes[1].set_title('Slope evolution')
    fig.suptitle('Fixed-design simulation check: 68.27% / 95.45% intervals; black marks = truth',fontsize=10)
    saved(fig,'fig_mock_recovery')
def exact():
    model=PhysicalModel('base')
    cases=['baseline','signed_envelope','gp_log','gp_fid']
    labels=['paired baseline','signed transfer','DES GP: log mean','DES GP: fiducial mean']
    colors=['#333333','#bc5637','#2677a8','#557747']
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.65))
    for j,(case,label,color) in enumerate(zip(cases,labels,colors)):
        prefix=ROOT/('reproduction_physical/nonlinear_base' if case=='baseline' else f'reproduction_supplement/exact_redshift/{case}/joint')
        s=load(Path(str(prefix)+'_summary.json')); chain=np.load(str(prefix)+'_chains.npy',mmap_mode='r')
        latent=np.load(str(prefix)+'_mu.npy',mmap_mode='r')
        idx=np.asarray(s.get('mu_saved_indices',np.arange(((s['burn']+99)//100)*100,s['steps'],100)),int)
        theta=np.asarray(chain[:,idx,:]).reshape(-1,5); latent=np.asarray(latent).reshape(-1,176)
        values=np.sort(np.asarray(chain[:,s['burn']:,0]).ravel())
        axes[0].plot(values[::80],np.arange(1,len(values)+1)[::80]/len(values),color=color,lw=1.8,label=label)
        pred,var=predict_joint(model,theta,latent); residual=(model.obs-pred)/np.sqrt(var)
        groups=np.array_split(np.argsort(model.zs),4); z=np.array([model.zs[g].mean() for g in groups])
        means=np.column_stack([residual[:,g].mean(axis=1) for g in groups]); q=np.percentile(means,[15.865,50,84.135],axis=0)
        axes[1].errorbar(z+(j-1.5)*.008,q[1],yerr=[q[1]-q[0],q[2]-q[1]],fmt='o-',color=color,lw=1.25,ms=4,capsize=2.5,label=label)
    axes[0].axvline(0,color='.55',ls='--',lw=.8); axes[0].axvline(LOWER,color='.55',ls=':',lw=.8)
    axes[0].set(xlim=(-.10,.85),ylim=(0,1),xlabel=r'Curvature $\Omega_k$',ylabel='Marginal posterior CDF')
    axes[1].axhline(0,color='.55',ls='--',lw=.8); axes[1].set(xlim=(.35,1.1),ylim=(-.7,.7),xlabel=r'Source redshift $z_s$ (quartile mean)',ylabel='Mean standardized velocity residual')
    axes[0].legend(loc='lower right',fontsize=7.2,frameon=False); axes[1].legend(loc='upper right',fontsize=7.2,frameon=False)
    for ax in axes: ax.grid(alpha=.14)
    saved(fig,'fig_exact_redshift')
def tables():
    rows=[]
    with (ROOT/'provenance/result_registry.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            s=load(ROOT/r['summary']); q=np.asarray(s['quantiles'])
            rows.append({'case':r['case'],'role':r['role'],'N':s['N'],'summary':r['summary'],
                **{f'{p}_{t}':q[k,i] for p,i in [('Omega_k',0),('gamma_z',2)] for k,t in enumerate(['q15p865','median','q84p135'])}})
    pd.DataFrame(rows).to_csv(OUT/'all_28_fit_intervals.csv',index=False)
if __name__=='__main__':
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    matching(); source(); mocks(); exact(); tables(); physical()
    print('Figures and interval table written to',OUT,flush=True)
