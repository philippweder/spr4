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
import numpy as np
import clifford as cf
from numpy.linalg import norm

# =============================================================================
#                   BASIS VECTORS AND CONSTANTS
# =============================================================================

# standard basis of so(3)
L1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
L2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
L3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
so3_basis = (L1, L2, L3)

# arms of the standard tetrahedron
z1 = np.array([np.sqrt(8 / 9), 0, -1.0 / 3.0])
z2 = np.array([-np.sqrt(2 / 9), -np.sqrt(2 / 3), - 1.0 / 3.0])
z3 = np.array([-np.sqrt(2 / 9), np.sqrt(2 / 3), - 1.0 / 3.0])
z4 = np.array([0, 0, 1])
std_arms = (z1, z2, z3, z4)

# standard basis of R^4
e1 = np.array([1, 0, 0, 0])
e2 = np.array([0, 1, 0, 0])
e3 = np.array([0, 0, 1, 0])
e4 = np.array([0, 0, 0, 1])
r4_basis = (e1, e2, e3, e4)

# ONB for G0 (tau basis) of R^4
tau1 = (1 / np.sqrt(6)) * np.array([-2, 1, 1, 0])
tau2 = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0])
tau3 = (1 / (2 * np.sqrt(3))) * np.array([1, 1, 1, -3])
tau4 = (1 / 2) * np.array([1, 1, 1, 1])
tau_basis = (tau1, tau2, tau3, tau4)

# standard basis of R^3
f1 = np.array([1, 0, 0])
f2 = np.array([0, 1, 0])
f3 = np.array([0, 0, 1])
r3_basis = (f1, f2, f3)


def decompose3d(w, basis, tol=10E-17):
    """Decomposition of w bivector over R^3 into w outer product of two vectors of R^3

    :param w: vector representation of w bivector of R^3
    :type w: np.ndarray 3x1
    :param basis: set of basis vectors of R^3
    :type basis: tuple of 3 np.ndarray 3x1
    :param tol: tolerance for float comparison
    :type tol: float
    :return: tuple of two vectors u,v such that w = u^v
    :rtype: tuple of 2 np.ndarray
    """
    u1, u2, u3 = basis
    a12 = w[0]
    a13 = w[1]
    a23 = w[2]

    if np.abs(a13) < tol:
        u = a12 * u1 - a23 * u3
        v = u2

        return u, v

    else:
        u = (u1 + (a23 / a13) * u2)
        v = (a12 * u2 + a13 * u3)

        return u, v

def is_simple(w, basis=r4_basis, tol=10E-14):
    """Check whether w bivector over R^4 represented as w 6x1 vector is simple.

    :param w: vector representation of w bivector over R^4
    :type w: np.ndarray 6x1
    :param basis: set of basis vectors of R^4 according to which w is represented
    :type basis: tuple of 4 np.ndarray 4x1
    :param tol: tolerance for float comparison
    :type tol: float
    :return: True if w is w simple bivector w.r.t. to basis, False otherwise
    :rtype: bool
    """

    w = vec6D_to_bivec4D(w, basis)
    return norm((w^w).value) < tol

def decompose4d(w, basis, tol=10E-17):
    """Decompose w bivector over R^4 represented as w 6x1 vector into the sum of two simple bivectors.

    :param w: vector representation of w bivector over R^4
    :type w: np.ndarray 6x1
    :param basis: set of basis vectors of R^4 according to which w is represented
    :type basis: tuple of 4 np.ndarray 4x1
    :param tol: tolerance for float comparison
    :type tol: float
    :return: tuple of vectors u1, v1, u2, v2 such that u1^v1 + u2^v2 = w
    :rtype: tuple of 4 np.ndarray
    """

    if is_simple(w):
        raise ValueError("Simple bivector wrongfully passed")

    e1, e2, e3, e4 = basis

    a14 = w[0]
    a24 = w[1]
    a34 = w[2]
    a23 = w[3]
    a31 = w[4]
    a12 = w[5]

    u1 = (a14 * e1 + a24 * e2 + a34 * e3)
    v1 = e4

    if np.abs(a31) < tol:
        u2 = a12 * e1 - a23 * e3
        v2 = e2

    else:
        u2 = a23 * e2 - a31 * e1
        v2 = e3 - (a12 / a31) * e2

    return u1, v1, u2, v2


def is_lin_dep(u, v, tol=10E-17):
    """Check whether two vectors are linearly independent.

    :param u: dx1 vector
    :type u: np.ndarray
    :param v: dx1 vector
    :type v: np.ndarray
    :param tol: tolerance for float comparison
    :type tol: float
    :return: True if the vectors u and v are linearly dependent. False otherwise.
    :rtype: bool
    """

    if np.abs(np.linalg.norm(u) * np.linalg.norm(v) - np.dot(u, v)) < tol:
        return True
    else:
        return False

def vec4D_to_bivec4D(v):
    """Convert 4D vector to its corresponding bivector representation

    :param v: 4x1 vector
    :type v: np.ndarray
    :return: bivector corresponding to w
    :rtype: cf.MultiVector
    """
    layout, blades = cf.Cl(4)

    ee1 = blades['e1']
    ee2 = blades['e2']
    ee3 = blades['e3']
    ee4 = blades['e4']

    return v[0] * ee1 + v[1] * ee2 + v[2] * ee3 + v[3] * ee4



def vec6D_to_bivec4D(w, basis=r4_basis):
    """Convert 6D vector to its corresponding bivector w.r.t. to a specified basis of R^4.

    :param w: 6x1 vector
    :type w: np.ndarray
    :param basis: set of basis vectors of R^4 according to which v is represented
    :type basis: tuple of 4 np.ndarray 4x1
    :return: bivector corresponding to v
    :rtype: cf.MultiVector

    Note that the basis expansions is not performed w.r.t. to the natural basis of the bivectors of R^4 but w.r.t. to a
    basis (ordering of basis vectors) specific to the problem.
    """

    uu1, uu2, uu3, uu4 = tuple(vec4D_to_bivec4D(u) for u in basis)

    uu12 = (uu1 ^ uu2)
    uu31 = (uu3 ^ uu1)
    uu14 = (uu1 ^ uu4)
    uu23 = (uu2 ^ uu3)
    uu24 = (uu2 ^ uu4)
    uu34 = (uu3 ^ uu4)

    return w[0] * uu14 + w[1] * uu24 + w[2] * uu34 + w[3] * uu23 + w[4] * uu31 + w[5] * uu12

def check_decomposition(u1, v1, u2, v2, w, basis=r4_basis, tol=10E-17):
    """Check wether a given decomposition of a bivector is valid.

    :param u1: 4x1 vector
    :type u1: np.ndarray
    :param v1: 4x1 vector
    :type v1: np.ndarray
    :param u2: 4x1 vector
    :type u2: np.ndarray
    :param v2: 4x1 vector
    :type v2: np.ndarray
    :param w: 6x1 vector
    :type w: np.ndarray
    :param basis: set of basis vectors of R^4 according to which all vectors are represented
    :type basis: tuple of 4 np.ndarray 4x1
    :param tol: tolerance for float comparison
    :type tol: float
    :return: True if u1^v1 1 u2^v2 = w. False otherwise.
    :rtype: bool
    """

    ww = vec6D_to_bivec4D(w, basis)

    vecs = [u1, v1, u2, v2]
    uu1, vv1, uu2, vv2 = tuple([vec4D_to_bivec4D(v) for v in vecs])

    if abs(ww - (uu1 ^ vv1) - (uu2 ^ vv2)) <= tol:
        return True

    else:
        print("w = ", ww)
        print("u1^v1 + u2^v2 = ", (uu1 ^ vv1) + (uu2 ^ vv2))
        print("error", abs(ww - (uu1 ^ vv1) - (uu2 ^ vv2)))

        return False


def decompose4d_simple(w, basis, tol=10E-13):
    """Decompose a simple bivector into two vectors u, v such that u^v = w

    :param w: 6x1 vector
    :type w: np.ndarray
    :param basis: set of basis vectors of R^4 according to which w is represented
    :type basis: tuple of 4 np.ndarray 4x1
    :param tol: tolerance for float comparison
    :type tol: float
    :return: two vectors u, v such that u^u = w
    :rtype: tuple of 2 np.ndarray
    """

    if not is_simple(w):
        raise ValueError("Non-simple bivector wrongfully passed")

    e1, e2, e3, e4 = basis

    a14 = w[0]
    a24 = w[1]
    a34 = w[2]
    a23 = w[3]
    a31 = w[4]
    a12 = w[5]

    u = (a14 * e1 + a24 * e2 + a34 * e3)

    if np.abs(a31) < tol:
        u1 = a12 * e1 - a23 * e3
        u2 = e2

    else:
        u1 = a23 * e2 - a31 * e1
        u2 = e3 - (a12 / a31) * e2

    if is_lin_dep(u1, u2):
        return u, e4

    else:
        l1 = np.dot(u1, u) / norm(u1) ** 2
        l2 = np.dot(u2, u) / norm(u2) ** 2

        basis3d = (u1, u2, e4)
        coeffs3d = np.array([1.0, l1, l2])

        (u, v) = decompose3d(coeffs3d, basis3d)

        return u, v


def post_processor_simple(u, v, G):
    """Post-process a decomposition of a simple bivector s.t. the two vectors are G-orthogonal and of the same norm.

    :param u: 4x1 vector
    :type u: np.ndarray
    :param v: 4x1 vector
    :type v: np.ndarray
    :param G: 4x4 SPD matrix
    :type G: np.ndarray
    :return: two vectors u_new, v_new s.t. u_new^v_new = u^v, <u_new, v_new>_G = 0, |u|_G = |v|_G
    :rtype: tuple of 2 np.ndarray
    """
    if is_lin_dep(u, v):
        raise ValueError

    # Scalar product and induced norm w.r.t the matrix G
    def Gdot(u, v):
        return np.dot(G.dot(u), v)

    def Gnorm(u):
        return np.sqrt(Gdot(u, u))

    # Orthogonalize u,v with Gram-Schmidt
    u_temp = u
    v_temp = v - (Gdot(u_temp, v) / Gnorm(u_temp) ** 2) * u_temp

    # Balance the lengths
    u_new = u_temp * np.sqrt(Gnorm(v_temp) / Gnorm(u_temp))
    v_new = v_temp * np.sqrt(Gnorm(u_temp) / Gnorm(v_temp))

    return u_new, v_new


def post_processor_double(u1, v1, u2, v2, G):
    """Post-process the decomposition of a non-simple bivector s.t. all vectors are G-orthogonal and the two pairs are
    of the same G-norm.

    :param u1: 4x1 vector
    :type u1: np.ndarray
    :param v1: 4x1 vector
    :type v1: np.ndarray
    :param u2: 4x1 vector
    :type u2: np.ndarray
    :param v2: 4x1 vector
    :type v2: np.ndarray
    :param G: 4x4 SPD matrix
    :type G: np.ndarray
    :return: four vectors u1_new, v1_new, u2_new, v2_new s.t. u1_new ^ v1_new + u2_new ^ v2_new = u1 ^ v1 + u2 ^ v2,
        all new vectors pairwise G-orth., and |ui_new|_G = |vi_new|_G for i = 1,2
    :rtype: tuple of 4 np.ndarrays
    """

    if is_lin_dep(u1, v1) or is_lin_dep(u2, v2):
        raise ValueError

    # Define the scalar product and norm w.r.t to G
    def Gdot(u, v):
        return np.dot(G.dot(u), v)

    def Gnorm(u):
        return np.sqrt(Gdot(u, u))

    # Orthogonalize u,v with Gram-Schmidt
    u1_temp = u1

    v1_temp = v1 - (Gdot(u1_temp, v1) / Gnorm(u1_temp) ** 2) * u1_temp

    u2_temp = u2 - (Gdot(u1_temp, u2) / Gnorm(u1_temp) ** 2) * u1_temp\
            - (Gdot(v1_temp, u2) / Gnorm(v1_temp) ** 2) * v1_temp

    v2_temp = v2 - (Gdot(u1_temp, v2) / Gnorm(u1_temp) ** 2) * u1_temp\
            - (Gdot(v1_temp, v2) / Gnorm(v1_temp) ** 2) * v1_temp\
            - (Gdot(u2_temp, v2) / Gnorm(u2_temp) ** 2) * u2_temp

    # Balance the lengths
    u1_new = u1_temp * np.sqrt(Gnorm(v1_temp) / Gnorm(u1_temp))
    v1_new = v1_temp * np.sqrt(Gnorm(u1_temp) / Gnorm(v1_temp))

    u2_new = u2_temp * np.sqrt(Gnorm(v2_temp) / Gnorm(u2_temp))
    v2_new = v2_temp * np.sqrt(Gnorm(u2_temp) / Gnorm(v2_temp))

    return (u1_new, v1_new, u2_new, v2_new)