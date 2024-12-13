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
from numpy.linalg import norm, det
from scipy.integrate import quad, solve_ivp

from rich.console import Console
from rich.table import Table
from spr4.geom_utils import is_simple, decompose4d, decompose4d_simple, post_processor_simple, post_processor_double, \
    r3_basis, so3_basis, std_arms, tau_basis
from spr4.plot_utils import rescale_coordinates, amp_angle


class Spr4:

    def get_radius(self):
        return self.a

    def set_radius(self, a):
        self.a = a

    def get_init_length(self):
        return self.xi0

    def set_init_length(self, xi0):
        self.xi0 = xi0

    def build_params(self):
        """Builds control and energy parameters
        """

        # Parameters of energy functional
        self.rho = 3 / 4 + (9 / 16) * np.sqrt(3 / 2) * (self.a / self.xi0)
        self.h = 1 / 12 + (3 / 16) * np.sqrt(3 / 2) * (self.a / self.xi0)
        self.gc = 2 / 3 + (3 / 8) * np.sqrt(3 / 2) * (self.a / self.xi0)
        self.gt = 1 + np.sqrt(3 / 2) * (9 / 8) * (self.a / self.xi0)

        # Control parameters
        self.alpha0 = (-1 / 4 + 3 / 32 * np.sqrt(3 / 2) * self.a / self.xi0) * self.damp_alpha0

        self.alpha_minus = 1 / self.xi0 * (np.sqrt(3) / 256 * self.a / self.xi0)

        self.alpha_plus = 1 / self.xi0 * (-3 / 128 * np.sqrt(3 / 2) * self.a / self.xi0)

        self.alpha_bar_plus = 1 / self.xi0 * (-9 / 128 * np.sqrt(3 / 2) * self.a / self.xi0)

        self.beta_minus = 1 / self.xi0 ** 2 * (-1 / (16 * np.sqrt(6)) + 9 / 512 * self.a / self.xi0)

        self.hc = -2 * np.sqrt(6) * self.alpha_minus
        self.ht = -2 * np.sqrt(6) * self.beta_minus

        self.energy_params = (self.rho, self.h, self.gc, self.gt, self.hc, self.ht)
        self.control_params = (self.alpha0, self.alpha_minus, self.alpha_plus, self.alpha_bar_plus, self.beta_minus)

    def build_energy_matrices(self):
        """Assembles the energy matrix as well as its diagonalization
        """

        self.G0 = np.array([
            [self.rho, self.h, self.h, self.h],
            [self.h, self.rho, self.h, self.h],
            [self.h, self.h, self.rho, self.h],
            [self.h, self.h, self.h, self.rho]])

        tau1, tau2, tau3, tau4 = tau_basis
        self.U = np.array([tau1, tau2, tau3, tau4]).T

        self.L0 = np.diag([self.gc, self.gc, self.gc, self.gt])

        self.L012 = np.diag([np.sqrt(self.gc), np.sqrt(self.gc), np.sqrt(self.gc), np.sqrt(self.gt)])

        self.G012 = self.U @ self.L012 @ self.U.T

        self.L = -2 * np.sqrt(6) * np.diag([
            self.alpha_minus, self.alpha_minus, self.alpha_minus,
            self.beta_minus, self.beta_minus, self.beta_minus
        ])

        self.Linv = -1 / (2 * np.sqrt(6)) * np.diag([
            1 / self.alpha_minus, 1 / self.alpha_minus, 1 / self.alpha_minus,
            1 / self.beta_minus, 1 / self.beta_minus, 1 / self.beta_minus
        ])

    def build_control_matrices(self):
        """Assembles the control matrices
        """

        # zeroth order terms
        tau1, tau2, tau3, _ = tau_basis
        self.A0 = -3 * np.sqrt(3) * self.alpha0 * np.array([tau1, tau2, tau3])
        self.B0 = np.zeros((3, 4))
        self.F0 = np.vstack((self.A0, self.B0))

        # first order terms in space - skew symmetric parts
        self.A1_minus = self.alpha_minus * np.array([
            [0, 3, 3, 2],
            [-3, 0, 0, -1],
            [-3, 0, 0, -1],
            [-2, 1, 1, 0]
        ])

        self.A2_minus = np.sqrt(3) * self.alpha_minus * np.array([
            [0, 1, -1, 0],
            [-1, 0, -2, -1],
            [1, 2, 0, 1],
            [0, 1, -1, 0]
        ])

        self.A3_minus = 2 * np.sqrt(2) * self.alpha_minus * np.array([
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [1, 1, 1, 0]
        ])

        # first order terms in space - symmetric parts
        self.A1_plus = np.sqrt(2) / 3 * self.alpha_bar_plus * np.array([
            [2, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 0]
        ]) + \
                       np.sqrt(2) / 3 * self.alpha_plus * np.array([
            [0, -1, -1, -2],
            [-1, 0, 2, 1],
            [-1, 2, 0, 1],
            [-2, 1, 1, 0]
        ])

        self.A2_plus = np.sqrt(2 / 3) * self.alpha_bar_plus * np.array([
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ]) + \
                       np.sqrt(2 / 3) * self.alpha_plus * np.array([
            [0, 1, -1, 0],
            [1, 0, 0, 1],
            [-1, 0, 0, -1],
            [0, 1, -1, 0]
        ])

        self.A3_plus = 1 / 3 * self.alpha_bar_plus * np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 3]
        ]) + \
                       2 / 3 * self.alpha_plus * np.array([
            [0, 1, 1, -1],
            [1, 0, 1, -1],
            [1, 1, 0, -1],
            [-1, -1, -1, 0]
        ])

        # first-order terms in rotation - skew-symmetric parts
        self.B1_minus = self.beta_minus * np.array([
            [0, 1, -1, 0],
            [-1, 0, -2, 3],
            [1, 2, 0, -3],
            [0, -3, 3, 0]
        ])

        self.B2_minus = np.sqrt(3) * self.beta_minus * np.array([
            [0, -1, -1, 2],
            [1, 0, 0, -1],
            [1, 0, 0, -1],
            [-2, 1, 1, 0]
        ])

        self.B3_minus = 2 * np.sqrt(2) * self.beta_minus * np.array([
            [0, 1, -1, 0],
            [-1, 0, 1, 0],
            [1, -1, 0, 0],
            [0, 0, 0, 0]
        ])

        self.A1 = self.A1_plus + self.A1_minus
        self.A2 = self.A2_plus + self.A2_minus
        self.A3 = self.A3_plus + self.A3_minus

        self.B1 = self.B1_minus
        self.B2 = self.B2_minus
        self.B3 = self.B3_minus

    def __init__(self, a, xi0, damp_alpha0=1):
        self.a = a
        self.xi0 = xi0
        self.damp_alpha0 = damp_alpha0

        self.build_params()
        self.build_energy_matrices()
        self.build_control_matrices()

    def net_displacement(self, xi, dxidt, split=False):
        """Compute the net displacement due to a control curve

        :param xi: control curve [0, 2pi] -> R^4
        :type xi: Callable[[float], np.ndarray]
        :param dxidt: time derivative of xi
        :type dxidt: Callable[[float], np.ndarray]
        :param split: if split, then the spatial net displacement and angular net displacement are returned separately
        :type split: bool
        :return: net displacement in space and angle
        :rtype: if split=True then tuple of 2 np.ndarray 3x1 otherwise 1 np.ndarray 6x1
        """

        # TODO: Explain why this works
        tau1, tau2, tau3, tau4 = tau_basis
        integrand1 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), tau2, tau3]))
        integrand2 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), tau3, tau1]))
        integrand3 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), tau1, tau2]))

        integrand4 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), tau1, tau4]))
        integrand5 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), tau2, tau4]))
        integrand6 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), tau3, tau4]))

        position_integrands = [integrand1, integrand2, integrand3]
        orientation_integrands = [integrand4, integrand5, integrand6]
        integrands = [integrand1, integrand2, integrand3, integrand4, integrand5, integrand6]

        if split:
            dc = [quad(integrand, 0, 2 * np.pi)[0] for integrand in position_integrands]
            dR = [quad(integrand, 0, 2 * np.pi)[0] for integrand in orientation_integrands]
            return np.array(dc), np.array(dR)

        else:
            dp = [quad(integrand, 0, 2 * np.pi)[0] for integrand in integrands]
            return np.array(dp)

    def optimal_curve(self, dp, full=False):
        """

        :param dp: prescribed net displacement as a vector 6x1
        :type dp: np.ndarray
        :return: optimal control curve, time derivative, tuple of Fourier coefficients
        :rtype: tuple
        """

        # rescaling of the net displacement
        w = self.Linv.dot(dp)

        # distinction of simple and general case
        if is_simple(dp, basis=tau_basis):
            uraw, vraw = decompose4d_simple(w, basis=tau_basis)
            u, v = post_processor_simple(uraw, vraw, self.G0)

            xi = lambda t: 1 / np.sqrt(2 * np.pi) * (np.sin(t) * u + np.cos(t) * v)
            coeffs = (u, v, np.zeros((4,)), np.zeros((4,)))
            dxidt = lambda t: 1 / np.sqrt(2 * np.pi) * (np.cos(t) * u - np.sin(t) * v)

        else:
            u1raw, v1raw, u2raw, v2raw = decompose4d(w, basis=tau_basis)
            u1, v1, u2, v2 = post_processor_double(u1raw, v1raw, u2raw, v2raw, self.G0)

            # Swap vector pairs if needed
            if norm(self.G012.dot(u1)) * norm(self.G012.dot(v1)) < norm(self.G012.dot(u2)) * norm(self.G012.dot(v2)):
                u1, u2 = u2, u1
                v1, v2 = v2, v1

            xi = lambda t: 1 / np.sqrt(2 * np.pi) * (np.sin(t) * u1 + np.cos(t) * v1 \
                                                     + 1 / np.sqrt(2) * np.sin(2 * t) * u2 + 1 / np.sqrt(2) * np.cos(
                        2 * t) * v2)

            dxidt = lambda t: 1 / np.sqrt(2 * np.pi) * (np.cos(t) * u1 - np.sin(t) * v1 \
                                                        + np.sqrt(2) * np.cos(2 * t) * u2 - np.sqrt(2) * np.sin(
                        2 * t) * v2)
            coeffs = (u1, v1, u2, v2)

        return xi, dxidt, coeffs


    def energy(self, dxidt):
        """Compute energy cosumption due to a control curve xi

        :param dxidt: time derivative of a control curve
        :type dxidt: Callable[[float], np.ndarray]
        :return: energy
        :rtype: float
        """
        integrand = lambda t: np.dot(self.G0.dot(dxidt(t)), dxidt(t))

        return quad(integrand, 0, 2 * np.pi)[0]


    def rhs(self, t, u, xi, dxidt):
        """Linearized RHS of control equations.

        :param t: time parameter
        :type t: float
        :param u: vector 12x1 containing position and orientation (orientation flattened)
        :type u: np.ndarray
        :param xi: control curve
        :type xi: Callable[[float], np.ndarray]
        :param dxidt: time derivative of control curve
        :type dxidt: Callable[[float], np.ndarray]
        :return: dudt (time derivative of variable u)
        :rtype: np.ndarray
        """
        f1, f2, f3 = r3_basis
        L1, L2, L3 = so3_basis

        R = u[3:].reshape(3, 3)

        dcdt = np.dot(R, (np.dot(self.A0, dxidt(t)) + np.dot(np.dot(self.A1, xi(t)), dxidt(t)) * f1 + np.dot(
            np.dot(self.A2, xi(t)), dxidt(t)) * f2 + np.dot(np.dot(self.A3, xi(t)),
                                                            dxidt(t)) * f3))

        dRdt = np.dot(R, (np.dot(np.dot(self.B1, xi(t)), dxidt(t)) * L1 + np.dot(np.dot(self.B2, xi(t)), dxidt(t)) * L2 \
                          + np.dot(np.dot(self.B3, xi(t)), dxidt(t)) * L3))

        dudt = np.hstack((dcdt, dRdt[0, :], dRdt[1, :], dRdt[2, :]))

        return dudt

    def build_ic(self, c0, R0):
        """Assemble initial condition

        :param c0: initial position in R^3
        :type c0: np.ndarray
        :param R0: initial orientation in R^3, i.e. rotation matrix 3x3
        :type R0: np.ndarray
        :return: vector 12x1 containing the concatenated arrays with R0 split row-wise
        :rtype: np.ndarray
        """
        return np.hstack((c0, R0[0, :], R0[1, :], R0[2, :]))

    def optimal_curve_trajectory(self, problem_params,plot_params, full=False, verbose=False):
        """

        :param problem_params: tuple of problem parameters (dp, c0, R0, n_strokes, n_res) with
            dp : prescribed net displacement 6x1
            c0 : initial position 3x1
            R0 : initial orientation 3x3
            n_strokes : number of swimming strokes
            n_res : number of timesteps per swimming stroke
        :type problem_params: tuple
        :param plot_params: tuple of plotting parameters, i.e. important for visibility of curve
            n_positions : int, number of highlighted positions
            amp_rot : float, amplification of rotation
            damp_arms : float, damping of std arm length
            damp_curve: float, damping of control curve
            aspect_ratio : tuple(int, int, int), scaling of coordinates
        :type plot_params: tuple
        :param full: If True, then full output is returned, i.e. tuple of
            sol_c : solution vector of position 3x(n_res*n_strokes)
            sol_R : solution vector of orientation (n_res*n_strokes)x3x3
            sphere_hists : time histories of sphere center points 3x(n_res*n_strokes)
            pos_hists : vector of highlighted positions 3xn_positions
            diff : difference in position between t=0 and t=T
            dp_int : integrated net displacement from computed optimal curve
            T : Endpoint of time interval
            Otherwise, only sol_c, diff, dp_int are returned.
        :type full: bool
        :param verbose: True for verbose output
        :type verbose: bool
        :return: cf. full
        :rtype: tuple
        """

        dp, c0, R0, n_strokes, n_res = problem_params
        n_positions, amp_rot, damp_arms, damp_curve, aspect_ratio, scale_arms = plot_params

        xi, dxidt, coeffs = self.optimal_curve(dp, full=True)
        T = np.linspace(0, n_strokes * 2 * np.pi, n_res * n_strokes, endpoint=True)
        t_span = [0, 2 * np.pi * n_strokes]
        p0 = self.build_ic(c0, R0)

        sol = solve_ivp(self.rhs, t_span, p0, t_eval=T, atol=10E-16, rtol=10E-16, method="DOP853", args=(xi, dxidt))
        sol_c = sol.y[:3, :]
        diff = sol_c[:, -1] - sol_c[:, 0]
        sol_R_raw = sol.y[3:, :]
        sol_R = []
        for k in range(len(T)):
            u = sol_R_raw[:, k]
            new_R = u.reshape(3, 3)
            sol_R.append(new_R)

        sol_R = np.array(sol_R)
        dp_int = self.net_displacement(xi, dxidt)

        if verbose:
            table = Table(title="Simulation result", style="bold", expand=True)

            table.add_column("Vector", justify="left", style="bold")
            table.add_column("Value", justify="left", style="bold")

            np.set_printoptions(precision=3)
            table.add_row("Prescr. Net Disp.", f"{n_strokes * dp}")
            table.add_row("Traj. Net Disp.", f"{diff}")
            table.add_row("Int. Net Disp.", f"{n_strokes * dp_int}")

            console = Console()
            console.print(table)

        if full:
            # computation of sphere positions
            sphere_hists = [[] for _ in range(5)]

            for k in range(len(T)):
                Rk = sol_R[k, :, :]
                Rk = amp_angle(Rk, amp_rot)
                ck = sol_c[:, k]

                for i, hist in enumerate(sphere_hists):
                    if i == 4:  # center sphere
                        hist.append(ck)

                    else:  # outer spheres
                        if scale_arms:
                            hist.append(ck + rescale_coordinates(
                                np.dot(Rk, (self.xi0 / damp_arms + xi(T[k])[i] / damp_curve) * std_arms[i]), aspect_ratio))
                        else:  # no scaling of the arms according to control curve
                            hist.append(ck + rescale_coordinates(np.dot(Rk, self.xi0 * std_arms[i] / damp_arms),
                                                                 aspect_ratio))

            sphere_hists = tuple([np.array(hist) for hist in sphere_hists])

            # computation of regularly spaced positions to be highlighted in the plot
            interval_len = int(len(sphere_hists[0]) / n_positions)
            pos_hists = [[] for _ in range(5)]

            for k in range(n_positions):
                for sphere_hist, pos_hist in zip(sphere_hists, pos_hists):
                    pos_hist.append(sphere_hist[k * interval_len, :])
                    pos_hist.append(sphere_hist[-1, :])

            pos_hists = tuple([np.array(hist) for hist in pos_hists])

        if full:
            return sol_c, sol_R, sphere_hists, pos_hists, diff, dp_int, T
        else:
            return sol_c, diff, dp_int
