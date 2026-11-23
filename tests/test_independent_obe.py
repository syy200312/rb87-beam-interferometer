"""Physical and numerical invariants for the independent, explicit D2 OBEs."""
import unittest
from types import SimpleNamespace

import numpy as np

from rb87_beam.independent_obe import atomic_tensors, propagate_ensemble
from rb87_beam.obe_slow_atoms import phase_quadrature


def sample(vx=0.01):
    return SimpleNamespace(r_m=np.zeros((1,3)),v_m_s=np.array([[vx,0.,15.]]),
                           reference_plane_z_m=0., ground_density=np.eye(8)[None,:]/8)


class OBEPhysicsTests(unittest.TestCase):
    def test_independent_wigner_strengths_and_sign_interference(self):
        d=atomic_tensors()
        self.assertAlmostEqual(d[2,-1,-1]**2,1.)
        np.testing.assert_allclose(np.sum(d*d,axis=(0,2)),1.,atol=1e-14)
        self.assertAlmostEqual(np.max(d[:,1:4,3:]**2),.1)
        self.assertAlmostEqual(np.max(d[:,0,:3]**2),1/3)
        np.testing.assert_allclose(np.sum(d[:,1:4,:3]**2,axis=(0,2)),5/6)
        # The q=+1 and q=-1 virtual F'=1 paths for the m=0 clock pair
        # cancel for identical linear light; magnitude-only d's would fail.
        self.assertAlmostEqual(np.sum(d[:,1:4,1]*d[:,1:4,5]),0.)

    def test_dark_input_and_zero_drive(self):
        out=propagate_ensemble(sample(),transfer_uW=0.,dark_uW=0.,length_mm=4.,
                               detuning_MHz=1.5,slices=8)
        np.testing.assert_allclose(out["ground_populations"],1/8,atol=1e-13)
        self.assertEqual(out["scattered_photons"][0],0.)

    def test_integrated_counter_and_optical_coherence_when_linear(self):
        # In the weak-field limit first optical harmonics are sufficient;
        # explicit independent standing-wave trajectories integrated over
        # the input phase give the same mean photon number.
        kw=dict(transfer_uW=.003,dark_uW=.001,length_mm=4.,detuning_MHz=1.5,slices=48)
        e=sample(0.)
        h=propagate_ensemble(e,grating_order=0,**kw)
        q=phase_quadrature(e,phase_nodes=4,**kw)
        self.assertGreater(q["scattered_photons"][0],1e-5)
        np.testing.assert_allclose(h["scattered_photons"],q["scattered_photons"],rtol=.004)
        self.assertLess(float(h["trace_error"][0]),1e-12)
        self.assertLess(float(q["trace_error"][0]),1e-12)


if __name__=="__main__":
    unittest.main()
