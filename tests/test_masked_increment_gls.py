"""Numerical observation/abstention contract, not a physiological HR test."""
import unittest
import numpy as np
from scipy.signal import lombscargle
from src.radar.masked_increment_gls import masked_increment_gls,peak_status

class MaskedIncrementGLSTest(unittest.TestCase):
    def setUp(self):self.t=np.arange(128)/10;self.freq=np.arange(11,45)*10/128
    def wave(self,f):return np.exp(1j*.3*np.sin(2*np.pi*f*self.t))
    def run_gls(self,z,t=None):return masked_increment_gls(z,self.t if t is None else t,self.freq)
    def test_clean_and_gapped_bin_exact(self):
        for f in (1.25,2.5):
            for gap in (False,True):
                z=self.wave(f)
                if gap:z[np.r_[10:21,40:51,80:91]]=0
                result=self.run_gls(z);self.assertEqual(int(result['status']),0);self.assertEqual(float(result['candidate_bpm']),60*f);self.assertGreater(result['power'].max(),.99)
    def test_no_phase_observations(self):
        for z in (np.zeros(128,complex),np.eye(1,128,30,dtype=complex).ravel()):
            r=self.run_gls(z);self.assertEqual(int(r['status']),1);self.assertTrue(np.isnan(r['candidate_bpm']));self.assertEqual(int(r['pair_count']),0)
    def test_two_isolated_samples_not_phase_pulses(self):
        z=np.zeros(128,complex);z[20]=1j;z[90]=-1j;r=self.run_gls(z);self.assertEqual(int(r['status']),1);self.assertEqual(int(r['pair_count']),0);self.assertTrue(np.all(r['power']==0))
    def test_constant_phase_or_constant_increment_abstains(self):
        for z in (np.ones(128,complex),np.exp(.2j)*np.ones(128),np.exp(.1j*np.arange(128))):
            self.assertEqual(int(self.run_gls(z)['status']),2)
    def test_original_grid_no_compression(self):
        z=self.wave(1.25);z[np.random.default_rng(371).choice(128,40,replace=False)]=0;r=self.run_gls(z);self.assertEqual(float(r['candidate_bpm']),75.)
        mask=(z[1:]!=0)&(z[:-1]!=0);v=np.angle(z[1:][mask]*z[:-1][mask].conj());wrong=lombscargle(np.arange(len(v))/10,v,2*np.pi*self.freq,floating_mean=True,normalize=True)
        self.assertLess(wrong[np.argmin(abs(self.freq-1.25))],.99)
    def test_periodic_gaps_can_destroy_design_rank(self):
        z=self.wave(1.25);z[::4]=0;r=self.run_gls(z);self.assertEqual(int(r['status']),3);self.assertTrue(np.isnan(r['candidate_bpm']))
    def test_gain_offset_and_conjugation(self):
        z=self.wave(1.25);z[30:45]=0;base=self.run_gls(z)
        for q in (z*1e100*np.exp(2j),z*1e-100*np.exp(-2j),z.conj()):
            r=self.run_gls(q);np.testing.assert_allclose(r['power'],base['power'],atol=1e-12);self.assertEqual(float(r['candidate_bpm']),75.)
    def test_tied_or_zero_evidence_abstains(self):
        self.assertEqual(int(peak_status(np.array([.4,.4]),np.array(10),np.array(1.),np.array([True,True]))),5)
        self.assertEqual(int(peak_status(np.zeros(2),np.array(10),np.array(1.),np.ones(2,bool))),4)
    def test_scipy_and_qr_power(self):
        z=self.wave(1.25);z[10:20]=0;result=self.run_gls(z);mask=(z[1:]!=0)&(z[:-1]!=0);t=((self.t[1:]+self.t[:-1])/2)[mask];v=np.angle(z[1:][mask]*z[:-1][mask].conj());ref=lombscargle(t,v,2*np.pi*self.freq,floating_mean=True,normalize=True);np.testing.assert_allclose(result['power'],ref,atol=1e-8,rtol=1e-6)
        energy=((v-v.mean())**2).sum();qr=[]
        for f in self.freq:
            design=np.stack((np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones(len(t))),axis=1);beta=np.linalg.lstsq(design,v,rcond=None)[0];qr.append(1-((v-design@beta)**2).sum()/energy)
        np.testing.assert_allclose(result['power'],qr,atol=1e-8,rtol=1e-6)
    def test_batch_shape(self):
        z=np.stack((self.wave(1.25),self.wave(2.5))).reshape(1,2,128);r=self.run_gls(z);self.assertEqual(r['power'].shape,(1,2,34));np.testing.assert_array_equal(r['candidate_bpm'],[[75.,150.]])
    def test_bad_inputs(self):
        for z,t,f in ((np.ones(128),self.t,self.freq),(self.wave(1.25),self.t[::-1],self.freq),(self.wave(1.25),self.t,[0,1]),(self.wave(1.25),self.t,[1,np.nan])):
            with self.assertRaises(ValueError):masked_increment_gls(z,t,f)
if __name__=='__main__':unittest.main()
