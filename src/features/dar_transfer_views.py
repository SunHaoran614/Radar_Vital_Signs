"""Native phase spectra and source-equivalent offline waveform views across rates."""
from fractions import Fraction

import numpy as np
from scipy.signal import butter,resample_poly,sosfiltfilt


GRID_BPM=np.arange(10,45,dtype=np.float64)*4.6875


def native_power(cells,sample_rate_hz):
    """[cells, native samples] complex -> safe 35-bin log relative HR power."""

    if not np.iscomplexobj(cells):raise ValueError('Complex radar cells required, not a real power spectrum')
    z=np.asarray(cells,dtype=np.complex128);fs=float(sample_rate_hz)
    if z.ndim!=2 or z.shape[1]!=round(12.8*fs) or not np.isfinite(z).all():raise ValueError('Finite native 12.8-second complex cells required')
    phase=np.concatenate((np.zeros((len(z),1)),np.angle(z[:,1:]*z[:,:-1].conj())),axis=-1);phase-=phase.mean(-1,keepdims=True)
    ff=np.fft.rfft(phase*np.hanning(z.shape[-1]));freq=np.fft.rfftfreq(z.shape[-1],1/fs);ix=np.flatnonzero((freq>=.75)&(freq<=3.5))
    if len(ix)!=35 or not np.allclose(freq[ix]*60,GRID_BPM,atol=1e-10):raise ValueError('Physical HR grid differs')
    power=abs(ff[:,ix])**2*2;total=power.sum(-1,keepdims=True);relative=np.divide(power,total,out=np.zeros_like(power),where=total>0)
    return np.log1p(35*relative).astype(np.float32),int((total[:,0]==0).sum())


def views(stream,sample_rate_hz,starts):
    """[native frames,cells] -> raw spectra, normalized 20Hz and 10Hz windows.

    Whole-record zero-phase filtering is offline. Window starts retain their
    native indices; tail windows may be irregular. No labels or fitted target
    statistics are inputs. Cells with no HR-band energy remain zero.
    """
    if not np.iscomplexobj(stream):raise ValueError('Complex radar stream required')
    fs=float(sample_rate_hz);z=np.asarray(stream,dtype=np.complex128);starts=np.asarray(starts,dtype=np.int64)
    if fs not in (10.,20.,30.) or z.ndim!=2 or len(starts)<1 or starts[0]!=0 or np.any(np.diff(starts)<=0):raise ValueError('Explicit native geometry required')
    if not np.isfinite(z).all() or z.shape[1]<1:raise ValueError('Finite nonempty complex radar required')
    native_n=round(12.8*fs);end=int(starts[-1])+native_n
    if len(z)<end:raise ValueError('Incomplete last native window')
    z=z[:end].T
    native_phase=np.concatenate((np.zeros((len(z),1)),np.angle(z[:,1:]*z[:,:-1].conj())),axis=-1)
    if fs==20.:phase20=native_phase
    else:
        ratio=Fraction(20,int(fs));phase20=resample_poly(native_phase,ratio.numerator,ratio.denominator,axis=-1,window=('kaiser',8.6),padtype='constant')*(fs/20.)
    start20=starts*20./fs
    if np.max(abs(start20-np.rint(start20)))>1e-9:raise ValueError('Native starts do not land on the 20Hz grid')
    start20=np.rint(start20).astype(np.int64)
    if phase20.shape[1]!=int(start20[-1])+256:raise ValueError('Retained native span and 20Hz support differ')
    filtered=sosfiltfilt(butter(4,6.,fs=20.,output='sos'),phase20,axis=-1)
    wave20=[];power=[];zero=0
    for native_start,source_start in zip(starts,start20):
        w=filtered[:,source_start:source_start+256];w=w-w.mean(-1,keepdims=True);w=w/np.maximum(np.sqrt((w*w).mean(-1,keepdims=True)),1e-6);wave20.append(w.astype(np.float32))
        p,nzero=native_power(z[:,native_start:native_start+native_n],fs);power.append(p);zero+=nzero
    wave20=np.stack(wave20);wave10=resample_poly(wave20,1,2,axis=-1,window=('kaiser',8.6),padtype='constant').astype(np.float32);power=np.stack(power)
    if wave20.shape!=(len(starts),len(z),256) or wave10.shape!=(len(starts),len(z),128) or power.shape!=(len(starts),len(z),35):raise AssertionError('Transfer-view geometry')
    if not all(np.isfinite(x).all() for x in (wave20,wave10,power)):raise ValueError('Nonfinite transfer view')
    return dict(power35=power,wave20=wave20,wave10=wave10,zero_hr_cells=zero,retained_native_frames=end)
