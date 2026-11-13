"""Fixed output smoothing with physical elapsed time, reset by caller per record."""
import numpy as np

MODES=('raw','ewma025','prior_half','ewma025_prior_half','first_hold')


def causal_readout(values,start_seconds,source_center,mode):
    values=np.asarray(values,dtype=np.float64);starts=np.asarray(start_seconds,dtype=np.float64)
    if values.ndim!=1 or starts.shape!=values.shape or len(values)==0 or not np.isfinite(values).all() or not np.isfinite(starts).all() or not np.all(np.diff(starts)>0):
        raise ValueError('One physical record with finite predictions and strictly increasing actual window starts required')
    if mode not in MODES or not np.isfinite(source_center):raise ValueError('Known fixed readout and finite fit-source center required')
    if mode=='first_hold':return np.full_like(values,values[0])
    result=values.copy()
    if mode in ('ewma025','ewma025_prior_half'):
        for j in range(1,len(result)):
            decay=.75**((starts[j]-starts[j-1])/6.4)
            result[j]=decay*result[j-1]+(1-decay)*values[j]
    if mode in ('prior_half','ewma025_prior_half'):result=.5*result+.5*float(source_center)
    return result
