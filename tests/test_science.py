"""Small numerical checks independent of the stored posterior summaries."""
from pathlib import Path
import sys,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from inference import structural,A,C
from physical_prior_revision import PhysicalModel,LOWER,H0,ZSTAR,DASTAR

class ScientificChecks(unittest.TestCase):
    def test_isothermal_isotropic_limit(self):
        self.assertAlmostEqual(float(structural(2.,2.,0.)),1/(2*np.sqrt(np.pi)),places=14)

    def test_distance_jacobian(self):
        model=PhysicalModel(); theta=np.array([.1,2.07,-.25,-.15,.1])
        sigma,jl,js,valid=model.predict(theta,True); self.assertTrue(valid[0])
        for role,jac in [('xl',jl),('xs',js)]:
            original=getattr(model,role).copy(); eps=1e-5
            setattr(model,role,original*np.exp(A*eps)); plus=model.predict(theta)[0]
            setattr(model,role,original*np.exp(-A*eps)); minus=model.predict(theta)[0]
            setattr(model,role,original)
            np.testing.assert_allclose((plus-minus)/(2*eps),jac,rtol=2e-7,atol=2e-7)

    def test_common_distance_rescaling(self):
        model=PhysicalModel(); theta=np.array([.1,2.07,-.25,-.15,.1]); original=model.predict(theta)[0]
        scale=1.17; model.xl*=scale; model.xs*=scale; theta[0]/=scale**2
        np.testing.assert_allclose(model.predict(theta)[0],original,rtol=2e-14,atol=1e-12)

    def test_cmb_geometric_bound_and_soft_weight(self):
        self.assertAlmostEqual(1+LOWER*((1+ZSTAR)*H0/C*DASTAR)**2,0.,places=14)
        model=PhysicalModel('soft07')
        self.assertAlmostEqual(float(model.geometric_weight(np.array([LOWER]))[0]),-np.log(2),places=12)
        self.assertEqual(float(model.geometric_weight(np.array([0.1]))[0]),0.)
        self.assertLess(float(model.geometric_weight(np.array([LOWER-.02]))[0]),-10)

if __name__=='__main__': unittest.main()
