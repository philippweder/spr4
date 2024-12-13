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
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from spr4.base import Spr4
from spr4.geom_utils import so3_basis
from scipy.integrate import solve_ivp
from pathlib import Path


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)
FIG_DIR = os.path.join(ROOT_DIR, "spr/case studies/figures")

L1, L2, L3 = so3_basis

def convergence_experiment_spr4(n_eps, FIG_NAME=None, FIG_DIR=FIG_DIR):
    # Setting the geometric parameters of the swimmer
    a = 0.01
    xi0 = 100

    # Initialization of w Spr4 object, cf. Spr4.py for details on the functionality
    swimmer = Spr4(a, xi0)

    # Definition of the net displacements which will be considered in the
    # convergence experiment. They will later be scaled by w factor epsilon
    dp_simple_spat = np.array([1, 0, 0, 0, 0, 0])
    dp_simple_rot = np.array([0, 0, 0, 0, 0, 1])
    dp_simple_spat_orth_rot = np.array([1, 1, 0, 0, 0, 1])
    dp_nonsimple = np.array([0, 0, 1, 0, 0, 1])

    dps = [dp_simple_spat, dp_simple_rot, dp_simple_spat_orth_rot, dp_nonsimple]

    label_simple_spat = "$\delta p_1 \sim (0, 0, 1, 0, 0, 0)$"
    label_simple_rot = "$\delta p_2 \sim (0, 0, 0, 0, 0, 1)$"
    label_simple_spat_orth_rot = "$\delta p_3 \sim (1, 1, 0, 0, 0, 1)$"
    label_nonsimple = "$\delta p_4 \sim (0, 0, 1, 0, 0, 1)$"

    labels = [label_simple_spat, label_simple_rot, label_simple_spat_orth_rot, label_nonsimple]
    markers = ["s-", "d-", "p-", "h-"]

    # Initialization of empty containers for the different error arrays for each
    # net displacement from the list above
    rel_errors = []
    abs_errors = []

    rel_errors_R = []
    abs_errors_R = []

    # array of scaling factors for each of which we will compute the difference
    # between the integrated system of ODEs and the theoretical net displacement
    # calculated in our paper
    epsilons = np.logspace(-6, 0, n_eps)

    # Iteration over all net displacements
    for dp in dps:

        new_rel_errs = []
        new_abs_errs = []

        new_rel_errs_R = []
        new_abs_errs_R = []

        # Iteration over all epsilons
        for eps in epsilons:
            # Calculating the optimal curve for dp
            xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)

            # Rescaling the curve and its Fourier coefficients by epsilon
            xi_eps = lambda t: eps * xi(t)
            dxidt_eps = lambda t: eps * dxidt(t)
            a1, b1, a2, b2 = coeffs
            coeffs_eps = (eps * a1, eps * b1, eps * a2, eps * b2)

            # Calculating the theoretical displacement due to the RESCALED
            # optimal curve
            dp_exp = swimmer.net_displacement(xi_eps, dxidt_eps)
            dR_exp = dp_exp[3:]


            # Solving the system of ODEs of order 3 for the rescaled curve
            n_strokes = 1
            T = np.linspace(0, n_strokes * 2 * np.pi, 600 * n_strokes, endpoint=True)

            t_span = [0, 2 * np.pi * n_strokes]
            R0 = np.eye(3)
            c0 = np.array([0, 0, 0])
            p0 = np.hstack((c0, R0[0, :], R0[1, :], R0[2, :]))

            sol = solve_ivp(swimmer.rhs, t_span, p0, t_eval=T, method="DOP853", atol=10E-16, rtol=10E-16,
                            args=(xi_eps, dxidt_eps))
            sol_c = sol.y[:3, :]
            sol_R = sol.y[3:, :]

            # Calculating Delta_c and Delta_R
            diff = sol_c[:, -1] - sol_c[:, 0]
            diff_R = sol_R[:, -1] - sol_R[:, 0]

            R1 = diff_R[:3]
            R2 = diff_R[3:6]
            R3 = diff_R[6:]

            diff_R = np.vstack((R1, R2, R3))
            dR_exp = dp_exp[3] * L1 + dp_exp[4] * L2 + dp_exp[5] * L3

            # Calculating the different errors
            new_rel_errs.append(norm(diff - dp_exp[:3]) / norm(dp_exp[:3]))
            new_abs_errs.append(norm(diff - dp_exp[:3]))

            new_abs_errs_R.append(norm(diff_R - dR_exp))
            new_rel_errs_R.append(norm(diff_R - dR_exp) / norm(dR_exp))

        new_rel_errs = np.array(new_rel_errs)
        new_abs_errs = np.array(new_abs_errs)

        new_rel_errs_R = np.array(new_rel_errs_R)
        new_abs_errs_R = np.array(new_abs_errs_R)

        rel_errors.append(new_rel_errs)
        abs_errors.append(new_abs_errs)

        abs_errors_R.append(new_abs_errs_R)
        rel_errors_R.append(new_rel_errs_R)

    fz = 16  # fontsize
    fz_ticks = 14  # fontsize for ticks

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7), sharey="row", sharex="row", \
                             gridspec_kw={'height_ratios': [4, 1]})
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,7), sharey=True, sharex=True)
    axs = axes[0, :]
    non_axs = axes[1, :]

    for errors, label, marker in zip(abs_errors, labels, markers):
        axs[0].loglog(epsilons, errors, marker, label=label)

    axs[0].loglog(epsilons, (1.0 * epsilons) ** 3, "k-.", label="$\mathcal{O}(\epsilon^3)$")
    axs[0].set_xlabel("$\epsilon$", fontsize=fz)
    axs[0].set_ylabel("$||\Delta c_\epsilon - \delta c_\epsilon ||$", fontsize=fz)
    axs[0].tick_params(axis="x", labelsize=fz_ticks)
    axs[0].tick_params(axis="y", labelsize=fz_ticks)
    handles1, labels1 = axs[0].get_legend_handles_labels()
    for errors, label, marker in zip(abs_errors_R, labels, markers):
        axs[1].loglog(epsilons, errors, marker, label=label)

    axs[1].loglog(epsilons, (1.0 * epsilons) ** 4, "k--", label="$\mathcal{O}(\epsilon^4)$")
    axs[1].set_xlabel("$\epsilon$", fontsize=fz)
    axs[1].set_ylabel("$||\Delta R_\epsilon - \delta R_\epsilon ||$", fontsize=fz)
    axs[1].tick_params(axis="x", labelsize=fz_ticks)
    axs[1].tick_params(axis="y", labelsize=fz_ticks)

    handles2, labels2 = axs[1].get_legend_handles_labels()
    handles2.append(handles1[-1])
    labels2.append(labels1[-1])

    fig.legend(handles2, labels2, loc='lower center', fontsize=fz, ncol=3)
    for ax in non_axs:
        ax.set_axis_off()
    # Turn back on the axis labels for the shared axes
    for ax in axs.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
    plt.tight_layout()
    if FIG_NAME is not None:
        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH)
    plt.show()

if __name__ == "__main__":
    n_eps = 20

    FIG_NAME="convergence_spr4.png"
    convergence_experiment_spr4(n_eps, FIG_NAME=FIG_NAME)
