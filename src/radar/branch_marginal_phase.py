"""Energy-ratio marginal over independent equiprobable shortest antipodal branches.

The branch prior is explicit and artificial. This ratio of expected quadratic
energies is neither the mean normalized gain nor a cardiac probability.
"""
import numpy as np


def branch_marginal_phase(iq,time_s,marks,frequencies):
    z=np.asarray(iq);t=np.asarray(time_s,dtype=float);mark=np.asarray(marks);f=np.asarray(frequencies,dtype=float)
    if z.ndim!=1 or not np.iscomplexobj(z) or len(z)<2 or not np.isfinite(z).all():raise ValueError('finite 1D complex IQ required')
    if t.shape!=z.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('strictly increasing finite original times required')
    if mark.shape!=(len(z)-1,) or mark.dtype!=np.bool_:raise ValueError('boolean adjacent-edge ambiguity marks required')
    if f.ndim!=1 or not len(f) or not np.isfinite(f).all() or np.any(f<=0) or np.any(np.diff(f)<=0):raise ValueError('positive increasing frequencies required')
    mask=z!=0
    if np.any(mark & ~(mask[:-1]&mask[1:])):raise ValueError('ambiguity marks must join observed endpoints')
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    mean=np.zeros(len(z));columns=[];design=np.stack((np.cos(2*np.pi*f[:,None]*(t-t[0])),np.sin(2*np.pi*f[:,None]*(t-t[0]))),-1)
    for a,b in zip(starts,ends):
        phase=np.unwrap(np.angle(z[a:b]));delta=np.diff(phase);delta[mark[a:b-1]]=0.;m=phase[0]+np.r_[0,np.cumsum(delta)];mean[a:b]=m-m.mean()
        for edge in np.flatnonzero(mark[a:b-1])+a:
            col=np.zeros(len(z));col[edge+1:b]=1;col[a:b]-=col[a:b].mean();columns.append(col)
        design[:,a:b]-=design[:,a:b].mean(1,keepdims=True)
    design[:,~mask]=0;columns=np.array(columns).reshape((-1,len(z)));meanenergy=float(mean@mean);priorenergy=float(np.sum(columns**2)*np.pi**2);gram=np.einsum('fnk,fnl->fkl',design,design);eigen=np.linalg.eigvalsh(gram);dof=int(mask.sum())-len(starts);rank=(eigen[:,0]>1e-10*eigen[:,1])&(dof>=3)
    safe=np.where(rank[:,None,None],gram,np.eye(2));rhs=np.einsum('fnk,n->fk',design,mean);beta=np.linalg.solve(safe,rhs[...,None])[...,0];meanexplained=np.sum(rhs*beta,axis=-1)
    priorrhs=np.einsum('fnk,mn->mfk',design,columns);priorbeta=np.linalg.solve(safe[None],priorrhs[...,None])[...,0];priorexplained=np.sum(priorrhs*priorbeta,axis=(0,2))*np.pi**2
    def read(numerator,energy):
        power=np.zeros(len(f)) if energy<=1e-20 else np.where(rank,numerator/energy,0.)
        assert power.min()>=-1e-10 and power.max()<=1+1e-10
        power=np.clip(power,0,1);maximum=float(power.max());ties=int(np.sum(abs(power-maximum)<=1e-10*(1+abs(maximum))))
        status='insufficient_support' if dof<3 else 'constant_phase' if energy<=1e-20 else 'rank_deficient' if not rank.all() else 'no_evidence' if maximum<=1e-10 else 'ambiguous_peak' if ties!=1 else 'accepted'
        return dict(power=power,status=status,candidate_bpm=float(f[power.argmax()]*60) if status=='accepted' else None)
    return dict(version='branch_marginal_phase_v1',ensemble=read(meanexplained+priorexplained,meanenergy+priorenergy),prior=read(priorexplained,priorenergy),mean=read(meanexplained,meanenergy),mean_phase=mean,centered_step_columns=columns,mean_energy=meanenergy,prior_energy=priorenergy,rank_valid=rank,cardiac_validated=False,ratio_of_expectations=True)
