#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:07:37 2020

@author: philipp
"""
import os
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from spr4.base import Spr4
from spr4.geom_utils import tau_basis
from spr4.plot_utils import plot_3D_components, set_ax_options

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)
FIG_DIR = os.path.join(ROOT_DIR, "spr/case studies/figures")


def trajectories_plot_non_unique(sols, fcts, markers, highlights=False, sphere_hists=None, FIG_DIR=FIG_DIR):
    frontview = (0, 0)
    sideview = (0, 90)
    topview = (90, 0)
    views = [frontview, sideview, topview]
    FIG_NAMES = ["non_unique_front.png", "non_unique_side.png", "non_unique_top.png"]

    for view, FIG_NAME in zip(views, FIG_NAMES):
        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        max_ranges = []
        means_x = []
        means_y = []
        means_z = []

        for sol_c, marker in zip(sols, markers):
            x = sol_c[0, :]
            y = sol_c[1, :]
            z = sol_c[2, :]
            max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                                  z.max() - z.min()]).max() / 2.0
            max_ranges.append(max_range)

            mean_x = x.mean()
            mean_y = y.mean()
            mean_z = z.mean()

            means_x.append(mean_x)
            means_y.append(mean_y)
            means_z.append(mean_z)

            plot_3D_components(ax, sol_c, marker)

        if highlights:
            start_x = np.array([x[0]])
            start_y = np.array([y[0]])
            start_z = np.array([z[0]])

            end_x = np.array([x[-1]])
            end_y = np.array([y[-1]])
            end_z = np.array([z[-1]])

            ax.plot3D(start_x, start_y, start_z, "h", color="#28D709")
            ax.plot3D(end_x, end_y, end_z, "^", color="#CA13EB")

        if sphere_hists is not None:
            for sphere_hist in sphere_hists:
                for hist in sphere_hist[:-1]:
                    plot_3D_components(ax, hist.T)

        mean_x = np.array(means_x).mean()
        mean_y = np.array(means_y).mean()
        mean_z = np.array(means_z).mean()
        means = (mean_x, mean_y, mean_z)
        max_range = np.array(max_ranges).max()

        view_id = FIG_NAME.split(".")[0].split("_")[-1]
        x_fct, y_fct, z_fct = fcts
        set_ax_options(ax, means, max_range, x_fct, y_fct, z_fct)
        ax.tick_params(which='major', pad=3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if view_id == "front":
            ax.set_xticks([])
        elif view_id == "side":
            ax.set_yticks([])
        elif view_id == "top":
            ax.set_zticks([])
        ax.view_init(view[0], view[1])
        fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.0)

        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH, bbox_inches='tight')
        plt.show()


def sim(xi, dxidt, p0, n_res, n_strokes):
    T = np.linspace(0, n_strokes * 2 * np.pi, n_res * n_strokes, endpoint=True)
    t_span = [0, 2 * np.pi * n_strokes]

    sol = solve_ivp(swimmer.rhs, t_span, p0, t_eval=T, method="DOP853", atol=10E-16, rtol=10E-16, args=(xi, dxidt))
    sol_c = sol.y[:3, :]
    return sol_c

a = 0.01
xi0 = 100
damp_alpha0 = 1 / 100000
damp_arms = 1 / 1000000

swimmer = Spr4(a, xi0, damp_alpha0=damp_alpha0)
w = np.array([0, 0, 1, 0, 0, 1])
dp = swimmer.L @ w

tau1, tau2, tau3, tau4 = tau_basis
# construction of the first curve
u1 = tau1
v1 = tau2

u2 = tau3
v2 = tau4

xi1 = lambda t:  1 / np.sqrt(2 * np.pi) * (np.sin(t) * u1 + np.cos(t) * v1 + 1 / np.sqrt(2) * np.sin(2 * t) * u2 +
                                           1 / np.sqrt(2) * np.cos(2 * t) * v2)

dxi1dt = lambda t: 1 / np.sqrt(2 * np.pi) * (np.cos(t) * u1 - np.sin(t) * v1 + np.sqrt(2) * np.cos(2 * t) * u2
                                            - np.sqrt(2) * np.sin(2 * t) * v2)
coeffs1 = (u1, v1, u2, v2)

# construction of the second curve
x1 = 1/np.sqrt(2)*(tau1 + tau3)
y1 = 1/np.sqrt(2)*(tau2 + tau4)

x2 = 1/np.sqrt(2)*(tau1 - tau3)
y2 = 1/np.sqrt(2)*(tau2 - tau4)

xi2 = lambda t:  1 / np.sqrt(2 * np.pi) * (np.sin(t) * x1 + np.cos(t) * y1 + 1 / np.sqrt(2) * np.sin(2 * t) * x2 +
                                           1 / np.sqrt(2) * np.cos(2 * t) * y2)

dxi2dt = lambda t: 1 / np.sqrt(2 * np.pi) * (np.cos(t) * x1 - np.sin(t) * y1 + np.sqrt(2) * np.cos(2 * t) * x2
                                            - np.sqrt(2) * np.sin(2 * t) * y2)
coeffs2 = (x1, y1, x2, y2)


n_strokes = 1
n_res = 1000
T = np.linspace(0, n_strokes * 2 * np.pi, n_res * n_strokes, endpoint=True)

eps = 1000
dp_eps = eps * dp

xi1_eps = lambda t: np.sqrt(eps) * xi1(t)
dxi1dt_eps = lambda t: np.sqrt(eps) * dxi1dt(t)

xi2_eps = lambda t: np.sqrt(eps) * xi2(t)
dxi2dt_eps = lambda t: np.sqrt(eps) * dxi2dt(t)


t_span = [0, 2 * np.pi * n_strokes]
R0 = np.eye(3)
c0 = np.array([0, 0, 0])
p0 = swimmer.build_ic(c0, R0)

sol1_c = sim(xi1_eps, dxi1dt_eps, p0, n_res, n_strokes)
sol2_c = sim(xi2_eps, dxi2dt_eps, p0, n_res, n_strokes)

sols = [sol1_c, sol2_c]

# 3D Plotting
aspect_ratio = (1 / 2, 1 / 2, 1 / 2)
markers = ["k", "k--"]
trajectories_plot_non_unique(sols, aspect_ratio, markers, highlights=True)






