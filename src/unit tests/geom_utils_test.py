#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 2022

@author:

    Philipp Weder
    philipp.weder@epfl.ch
    Rue Louis-Favre 4
    CH-1024 Ecublens VD

"""

import unittest
from spr4.geom_utils import *
import clifford as cf
from spr4.geom_utils import r4_basis

class TestIsSimple(unittest.TestCase):
    def test_simple(self):
        coeffs = np.array([1, 0, 0, 0, 0, 0])
        self.assertEqual(is_simple(coeffs), True)  # add assertion here

    def test_not_simple(self):
        coeffs = np.array([1, 0, 0, 1, 0, 0])
        self.assertEqual(is_simple(coeffs), False)

    def test_tol_not_simple(self):
        coeffs = np.array([1, 0, 0, 1, 0, 0])*10E-18
        self.assertEqual(is_simple(coeffs), True)

class TestDecompose4D(unittest.TestCase):

    def test_decomposition(self):
        coeffs = np.array([1, 0, 0, 2, 0, 0])
        comp_vecs = decompose4d(coeffs, r4_basis)

        u1_exp = np.array([1, 0, 0, 0])
        v1_exp = np.array([0, 0, 0, 1])
        u2_exp = np.array([0, 0, -2, 0])
        v2_exp = np.array([0, 1, 0, 0])
        exp_vecs = (u1_exp, v1_exp, u2_exp, v2_exp)

        for exp_vec, comp_vec in zip(exp_vecs, comp_vecs):
            self.assertTrue(np.allclose(exp_vec, comp_vec))

    def test_simple_error(self):
        coeffs = np.array([1, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            decompose4d(coeffs, r4_basis)

class TestIsLinDep(unittest.TestCase):
    def test_lin_dep_true(self):
        self.assertTrue(is_lin_dep(e1, e1))

    def test_lin_dep_false(self):
        self.assertFalse(is_lin_dep(e1, e2))

    def test_lin_dep_true_tol(self):
        self.assertTrue(is_lin_dep(e1*10E-18, e2*10E-18))

class TestCheckDecomposition(unittest.TestCase):
    def test_correct_decomposition(self):
        w = np.array([1, 0, 0, 0, 0, 0])
        self.assertTrue(check_decomposition(e1, e4, np.zeros(4), np.zeros(4), w))

    def test_wrong_decomposition(self):
        w = np.array([1, 0, 0, 0, 0, 0])
        self.assertTrue(check_decomposition(e1, e4, e3, np.zeros(4), w))


class TestDecompose4DSimple(unittest.TestCase):
    def test_decomposition(self):
        coeffs = np.array([1, 0, 0, 0, 0, 0])
        u, v = decompose4d_simple(coeffs, r4_basis)
        self.assertTrue(np.allclose(u, e1))
        self.assertTrue(np.allclose(v, e4))

    def test_nonsimple_error(self):
        coeffs = np.array([1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            decompose4d_simple(coeffs, r4_basis)

class TestPostProcessors(unittest.TestCase):
    def test_simple_pp(self):
        G = np.eye(4)

        def Gdot(u, v):
            return np.dot(G.dot(u), v)

        def Gnorm(u):
            return np.sqrt(Gdot(u, u))

        u = np.random.randn(4)
        v = np.random.randn(4)
        u_new, v_new = post_processor_simple(u, v, G)

        self.assertAlmostEqual(Gdot(u_new, v_new).squeeze(), 0)
        self.assertAlmostEqual(Gnorm(u_new), Gnorm(v_new))

    def test_double_pp(self):
        G = np.eye(4)

        def Gdot(u, v):
            return np.dot(G.dot(u), v)

        def Gnorm(u):
            return np.sqrt(Gdot(u, u))

        u1 = np.random.randn(4)
        v1 = np.random.randn(4)
        u2 = np.random.randn(4)
        v2 = np.random.randn(4)
        u1_new, v1_new, u2_new, v2_new = post_processor_double(u1, v1, u2, v2, G)
        self.assertAlmostEqual(Gnorm(u1_new), Gnorm(v1_new))
        self.assertAlmostEqual(Gnorm(u2_new), Gnorm(v2_new))

        self.assertAlmostEqual(Gdot(u1_new, v1_new).squeeze(), 0)
        self.assertAlmostEqual(Gdot(u2_new, v2_new).squeeze(), 0)
        self.assertAlmostEqual(Gdot(u1_new, u2_new).squeeze(), 0)
        self.assertAlmostEqual(Gdot(u1_new, v2_new).squeeze(), 0)
        self.assertAlmostEqual(Gdot(v1_new, u2_new).squeeze(), 0)
        self.assertAlmostEqual(Gdot(v1_new, v2_new).squeeze(), 0)


        with self.assertRaises(ValueError):
            post_processor_double(e1, e1, e2, e3, G)
            post_processor_double(e2, e3, e1, e1, G)


class TestConversions(unittest.TestCase):
    def test_vec4D_to_bivec4D(self):
        layout, blades = cf.Cl(4)

        ee1 = blades['e1']
        ee2 = blades['e2']
        ee3 = blades['e3']
        ee4 = blades['e4']
        r4_basis_bv = (ee1, ee2, ee3, ee4)

        for ei, eei in zip(r4_basis, r4_basis_bv):
            self.assertEqual(vec4D_to_bivec4D(ei), eei)

    def test_vec6D_to_bivec4D(self):
        layout, blades = cf.Cl(4)

        ee1 = blades['e1']
        ee2 = blades['e2']
        ee3 = blades['e3']
        ee4 = blades['e4']
        r4_basis_bv = (ee1, ee2, ee3, ee4)

        e14 = np.array([1, 0, 0, 0, 0, 0])
        e24 = np.array([0, 1, 0, 0, 0, 0])
        e34 = np.array([0, 0, 1, 0, 0, 0])
        e23 = np.array([0, 0, 0, 1, 0, 0])
        e31 = np.array([0, 0, 0, 0, 1, 0])
        e12 = np.array([0, 0, 0, 0, 0, 1])

        comp_vecs = [e14, e24, e34, e23, e31, e12]
        comp_vecs_bv = [vec6D_to_bivec4D(eij, basis=r4_basis) for eij in comp_vecs]
        exp_vecs_bv = [(ee1 ^ ee4), (ee2 ^ ee4), (ee3 ^ ee4), (ee2 ^ ee3), (ee3 ^ ee1), (ee1 ^ ee2)]

        for comp_vec_bv, exp_vec_bv in zip(comp_vecs_bv, exp_vecs_bv):
            self.assertEqual(comp_vec_bv, exp_vec_bv)

if __name__ == '__main__':
    unittest.main()
