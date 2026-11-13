"""Analytic complex-response and interference tests for explicit RX operators."""
import unittest
import numpy as np
from src.radar.rx_beamforming import bartlett_weights, mvdr_weights, apply_weights


class RXBeamformingTests(unittest.TestCase):
    def test_fft_grid_and_target_phase(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(256,4))+1j*rng.normal(size=(256,4))
        bins = np.arange(-8,8)
        a = np.exp(2j*np.pi*bins[:,None]*np.arange(4)[None,:]/16)
        np.testing.assert_allclose(apply_weights(x,bartlett_weights(a)),
                                   np.fft.fftshift(np.fft.fft(x,n=16,axis=1),axes=1)/4,
                                   atol=1e-14)
        phase = np.exp(1j*(0.1*np.arange(256)+0.3*np.sin(np.arange(256))))
        for weights in (bartlett_weights(a[[11]]), mvdr_weights(x,a[[11]])):
            np.testing.assert_allclose(apply_weights(phase[:,None]*a[11],weights)[:,0],
                                       phase,atol=1e-13)

    def test_known_covariance_interference_suppression(self):
        # Exact ensemble covariance via deterministic factor, avoiding Monte Carlo.
        a = np.exp(2j*np.pi*0.13*np.arange(4))
        nuisance = np.exp(-2j*np.pi*0.19*np.arange(4))
        covariance = np.eye(4)+100*np.outer(nuisance,nuisance.conj())
        x = 2*np.linalg.cholesky(covariance).T
        w = mvdr_weights(x,a[None],diagonal_loading=0)[0]
        np.testing.assert_allclose(np.vdot(w,a),1,atol=1e-13)
        classic = bartlett_weights(a[None])[0]
        self.assertLess(abs(np.vdot(w,nuisance)),abs(np.vdot(classic,nuisance))/50)

    def test_invalid_and_singular_inputs(self):
        with self.assertRaises(ValueError):
            bartlett_weights(np.zeros((1,4)))
        with self.assertRaises(ValueError):
            mvdr_weights(np.zeros((4,4)),np.ones((1,4)))
        with self.assertRaises(ValueError):
            apply_weights(np.ones((3,4)),np.ones((2,3)))
        with self.assertRaises(ValueError):
            mvdr_weights(np.eye(4),np.ones((1,4)),diagonal_loading=-1)
        with self.assertRaises(np.linalg.LinAlgError):
            mvdr_weights(np.ones((8,4)),np.ones((1,4)),diagonal_loading=0)
        w = mvdr_weights(np.ones((8,4)),np.ones((1,4)))
        np.testing.assert_allclose(apply_weights(np.ones((8,4)),w),1,atol=1e-13)


if __name__ == '__main__':
    unittest.main()
