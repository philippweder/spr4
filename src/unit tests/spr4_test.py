import unittest
import numpy as np
from spr4.base import Spr4
from spr4.geom_utils import vec4D_to_bivec4D, vec6D_to_bivec4D

tau1 = (1 / np.sqrt(6)) * np.array([-2, 1, 1, 0])
tau2 = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0])
tau3 = (1 / (2 * np.sqrt(3))) * np.array([1, 1, 1, -3])
tau4 = (1 / 2) * np.array([1, 1, 1, 1])
tau_basis = (tau1, tau2, tau3, tau4)


class SimpleDecompositionsTest(unittest.TestCase):
    def test_translation(self):
        swimmer = Spr4(0.01, 100)
        dp = np.array([1, 0, 0, 0, 0, 0])
        w = swimmer.Linv @ dp
        w_bv = vec6D_to_bivec4D(w, basis=tau_basis)
        _, _, coeffs = swimmer.optimal_curve(dp, full=True)

        u1_bv, v1_bv, u2_bv, v2_bv = tuple(vec4D_to_bivec4D(u) for u in coeffs)
        self.assertAlmostEqual(abs(w_bv - ((u1_bv ^ v1_bv) + (u2_bv ^ v2_bv))), 0)


    def test_rotation(self):
        swimmer = Spr4(0.01, 100)
        dp = np.array([0, 0, 0, 1, 0, 0])
        w = swimmer.Linv @ dp
        w_bv = vec6D_to_bivec4D(w, basis=tau_basis)
        _, _, coeffs = swimmer.optimal_curve(dp, full=True)
        u1_bv, v1_bv, u2_bv, v2_bv = tuple(vec4D_to_bivec4D(u) for u in coeffs)
        self.assertAlmostEqual(abs(w_bv - ((u1_bv ^ v1_bv) + (u2_bv ^ v2_bv))), 0)


class NonSimpleDecompositionsTest(unittest.TestCase):
    def test_screw(self):
        swimmer = Spr4(0.01, 100)
        dp = np.array([1, 0, 0, 1, 0, 0])
        w = swimmer.Linv @ dp
        w_bv = vec6D_to_bivec4D(w, basis=tau_basis)
        _, _, coeffs = swimmer.optimal_curve(dp, full=True)

        u1_bv, v1_bv, u2_bv, v2_bv = tuple(vec4D_to_bivec4D(u) for u in coeffs)
        self.assertAlmostEqual(abs(w_bv - ((u1_bv ^ v1_bv) + (u2_bv ^ v2_bv))), 0)

    def test_full(self):
        swimmer = Spr4(0.01, 100)
        dp = np.array([1, 1, 1, 1, 1, 1])
        w = swimmer.Linv @ dp
        w_bv = vec6D_to_bivec4D(w, basis=tau_basis)
        _, _, coeffs = swimmer.optimal_curve(dp, full=True)

        u1_bv, v1_bv, u2_bv, v2_bv = tuple(vec4D_to_bivec4D(u) for u in coeffs)
        self.assertAlmostEqual(abs(w_bv - ((u1_bv ^ v1_bv) + (u2_bv ^ v2_bv))), 0)

if __name__ == '__main__':
    unittest.main()
