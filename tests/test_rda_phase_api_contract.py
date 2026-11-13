"""Regression tests for phase-offset fidelity and combined pipeline outputs."""
import numpy as np
import pytest

from src.radar import RDAConfig, apply_rda_pipeline
from src.radar.phase import _unwrap_phase, phase_single_bin


def example():
    t=np.arange(256)/20.
    phase=1.1+.15*np.sin(2*np.pi*1.25*t)
    return np.exp(1j*phase).astype(np.complex64)[:,None,None,None]


def test_unwrap_preserves_initial_offset_through_multiple_wraps():
    phase=1.1+np.arange(128)*.27
    principal=np.angle(np.exp(1j*phase))
    np.testing.assert_allclose(_unwrap_phase(principal),phase,atol=2e-6,rtol=0)
    assert _unwrap_phase(np.array([],dtype=float)).shape==(0,)


@pytest.mark.parametrize('rotation',[.6,2.6,-2.6])
@pytest.mark.parametrize('normalize',[False,True])
def test_detrended_phase_has_no_initial_phase_impulse(rotation,normalize):
    cube=example()
    a=phase_single_bin(cube,normalize=normalize)
    b=phase_single_bin(cube*np.exp(1j*rotation),normalize=normalize)
    np.testing.assert_allclose(a,b,atol=3e-6,rtol=0)


@pytest.mark.parametrize('method',['p0_single','p1_diff','p2_fusion'])
def test_phase_and_metadata_requests_return_the_same_trace(method):
    cube=example();config=RDAConfig(phase=method,phase_kwargs={'normalize':False})
    if method=='p2_fusion':config.phase_kwargs['top_bins']=1
    phase=apply_rda_pipeline(cube,config,return_phase=True)
    combined,metadata=apply_rda_pipeline(cube,config,return_phase=True,record_meta=True)
    assert combined.shape==(256,)
    np.testing.assert_array_equal(combined,phase)
    assert metadata['phase']['method']==method
    plain,meta=apply_rda_pipeline(cube,config,record_meta=True)
    np.testing.assert_array_equal(plain,cube)
    assert meta==metadata


def test_no_phase_selection_returns_cube_with_or_without_metadata():
    cube=example()
    np.testing.assert_array_equal(apply_rda_pipeline(cube,RDAConfig(),return_phase=True),cube)
    actual,meta=apply_rda_pipeline(cube,RDAConfig(),return_phase=True,record_meta=True)
    np.testing.assert_array_equal(actual,cube)
    assert meta=={}
