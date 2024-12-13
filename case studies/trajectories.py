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
from pathlib import Path
from spr4.base import Spr4
import numpy as np
import matplotlib.pyplot as plt
from matplot_fmt_pi.ticker import MultiplePi

from spr4.plot_utils import plot_3D_components, set_ax_options

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(str(Path(__file__).absolute())))
)
FIG_DIR = os.path.join(ROOT_DIR, "spr/case studies/figures")



def Rz(t):
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ])


def translation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME=None, FIG_DIR=FIG_DIR):
    swimmer = Spr4(a, xi0, damp_alpha0)

    dp = np.array([0, 0, 1, 0, 0, 0]) * eps
    c0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    n_positions = 1
    problem_params = (dp, c0, R0, n_strokes, n_res)

    amp_rot = 1
    damp_arms = 1
    damp_curve = 1
    aspect_ratio = (1, 1, 1)
    scale_arms = False
    plot_params = (n_positions, amp_rot, damp_arms, damp_curve, aspect_ratio, scale_arms)

    sol_c, sol_R, sphere_hist, pos_hist, diff, dp_exp, T = \
        swimmer.optimal_curve_trajectory(problem_params, plot_params, full=True, verbose=True)

    lower_bound = np.zeros(len(T))
    upper_bound = np.full(len(T), eps * n_strokes)

    sol_c_z = sol_c[2, :]

    pi_controller = MultiplePi(2)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 4))

    ax.plot(T, sol_c_z, "k")
    ax.plot(T, lower_bound, "-.", color="#28D709", label="starting point")
    ax.plot(T, upper_bound, "--", color="#CA13EB", label="end point")
    ax.legend(loc="lower right")
    ax.set_xlabel("time")
    ax.set_ylabel("z")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.xaxis.set_major_locator(pi_controller.locator())
    ax.xaxis.set_major_formatter(pi_controller.formatter())
    fig.tight_layout()
    if FIG_NAME is not None:
        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH)
    plt.show()


def rotation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME=None, FIG_DIR=FIG_DIR):
    swimmer = Spr4(a, xi0, damp_alpha0)

    dp = np.array([0, 0, 0, 0, 0, 1]) * eps
    c0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    n_positions = 5
    problem_params = (dp, c0, R0, n_strokes, n_res)

    amp_rot = 500
    damp_arms = 1E7
    damp_curve = 1E6
    aspect_ratio = (1/1.5, 1/1.5, 1/1.5)
    scale_arms = True
    plot_params = (n_positions, amp_rot, damp_arms, damp_curve, aspect_ratio, scale_arms)

    sol_c, sol_R, sphere_hist, pos_hist, diff, dp_exp, T = \
        swimmer.optimal_curve_trajectory(problem_params, plot_params, full=True, verbose=False)

    # 3D Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y', labelpad=15)
    ax.set_zlabel('z', labelpad=1)

    x = sol_c[0, :]
    y = sol_c[1, :]
    z = sol_c[2, :]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                          z.max() - z.min()]).max() / 2.0

    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()


    marker = "k"
    plot_3D_components(ax, sol_c, marker)

    pos_hist1, pos_hist2, pos_hist3, pos_hist4, pos_histc = pos_hist

    xs = []
    ys = []
    zs = []

    xc = pos_histc[:, 0]
    yc = pos_histc[:, 1]
    zc = pos_histc[:, 2]

    colors = ["#e31c23", "#1eb53a", "#00a3dd", "#fcd116"]
    for pos_histx, color in zip(pos_hist[:-1], colors):
        x = pos_histx[:, 0]
        y = pos_histx[:, 1]
        z = pos_histx[:, 2]

        ax.scatter(x, y, z, color=color, alpha=0.7)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # plotting gray arms at positions
    for k in range(len(xs[0, :]) - 1):
        for i in range(4):
            segment_x = np.array([xc[k], xs[i, k]])
            segment_y = np.array([yc[k], ys[i, k]])
            segment_z = np.array([zc[k], zs[i, k]])

            ax.plot3D(segment_x, segment_y, segment_z, "k", alpha=0.3)

    # plotting the final position with neon green arms
    for i in range(4):
        segment_x = np.array([xc[-1], xs[i, -1]])
        segment_y = np.array([yc[-1], ys[i, -1]])
        segment_z = np.array([zc[-1], zs[i, -1]])

        ax.plot3D(segment_x, segment_y, segment_z, "#CA13EB", alpha=0.7)

    # plotting the first position with purple arms
    for i in range(4):
        segment_x = np.array([xc[0], xs[i, 0]])
        segment_y = np.array([yc[0], ys[i, 0]])
        segment_z = np.array([zc[0], zs[i, 0]])

        ax.plot3D(segment_x, segment_y, segment_z, "#28D709", alpha=0.7)

    means = (mean_x, mean_y, mean_z)
    x_fct, y_fct, z_fct = aspect_ratio
    set_ax_options(ax, means, max_range, x_fct, y_fct, z_fct)
    ax.tick_params(axis="y", which='major', pad=10)
    ax.set_xlim([-5E-5, 5E-5])
    ax.set_ylim([-9E-5, 1E-5])
    ax.view_init(25, 47)
    fig.tight_layout()
    if FIG_NAME is not None:
        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH)
    fig.show()


def screw(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME=None, FIG_DIR=FIG_DIR):
    swimmer = Spr4(a, xi0, damp_alpha0)

    dp = np.array([0, 0, 1, 0, 0, 1]) * eps
    c0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    n_positions = 8
    problem_params = (dp, c0, R0, n_strokes, n_res)

    amp_rot = 2500
    damp_arms = 5E5
    damp_curve = 5E5
    aspect_ratio = (15, 15, 1)
    scale_arms = True
    plot_params = (n_positions, amp_rot, damp_arms, damp_curve, aspect_ratio, scale_arms)

    sol_c, sol_R, sphere_hist, pos_hist, diff, dp_exp, T = \
        swimmer.optimal_curve_trajectory(problem_params, plot_params, full=True, verbose=False)

    # 3D Plotting
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y', labelpad=6)
    ax.set_zlabel('z')

    x = sol_c[0, :]
    y = sol_c[1, :]
    z = sol_c[2, :]

    max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                          z.max() - z.min()]).max() / 2.0
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    means = (mean_x, mean_y, mean_z)

    marker = "k"
    plot_3D_components(ax, sol_c, marker)

    pos_hist1, pos_hist2, pos_hist3, pos_hist4, pos_histc = pos_hist

    xs = []
    ys = []
    zs = []

    xc = pos_histc[:, 0]
    yc = pos_histc[:, 1]
    zc = pos_histc[:, 2]

    # plotting the spheres
    colors = ["#e31c23", "#1eb53a", "#00a3dd", "#fcd116"]
    for pos_histx, color in zip(pos_hist[:-1], colors):
        x = pos_histx[:, 0]
        y = pos_histx[:, 1]
        z = pos_histx[:, 2]

        ax.scatter(x, y, z, color=color, alpha=0.7)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # plotting gray arms at positions
    for k in range(1, len(xs[0, :]) - 1):
        for i in range(4):
            segment_x = np.array([xc[k], xs[i, k]])
            segment_y = np.array([yc[k], ys[i, k]])
            segment_z = np.array([zc[k], zs[i, k]])

            ax.plot3D(segment_x, segment_y, segment_z, "k", alpha=0.3)

    # plotting the final position with neon green arms
    for i in range(4):
        segment_x = np.array([xc[-1], xs[i, -1]])
        segment_y = np.array([yc[-1], ys[i, -1]])
        segment_z = np.array([zc[-1], zs[i, -1]])

        ax.plot3D(segment_x, segment_y, segment_z, "#CA13EB", alpha=0.7)

    # plotting the first position with purple arms
    for i in range(4):
        segment_x = np.array([xc[0], xs[i, 0]])
        segment_y = np.array([yc[0], ys[i, 0]])
        segment_z = np.array([zc[0], zs[i, 0]])

        ax.plot3D(segment_x, segment_y, segment_z, "#28D709", alpha=0.7)

    x_fct, y_fct, z_fct = aspect_ratio
    set_ax_options(ax, means, max_range, x_fct, y_fct, z_fct)
    ax.set_xlim([-7E-5, 7E-5])
    ax.view_init(8, -65)
    fig.tight_layout()
    if FIG_NAME is not None:
        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH)
    fig.show()


def translation_orth_rotation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME=None, FIG_DIR=FIG_DIR):
    swimmer = Spr4(a, xi0, damp_alpha0)
    dp = np.array([2, 2, 0, 0, 0, 1]) * eps

    c0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    n_positions = 4
    problem_params = (dp, c0, R0, n_strokes, n_res)

    amp_rot = 2500
    damp_arms = 4E5
    damp_curve = 4E5
    aspect_ratio = (1/1.2, 1/1.2, 1)
    scale_arms = True
    plot_params = (n_positions, amp_rot, damp_arms, damp_curve, aspect_ratio, scale_arms)

    sol_c, sol_R, sphere_hist, pos_hist, diff, dp_exp, T = \
        swimmer.optimal_curve_trajectory(problem_params, plot_params, full=True, verbose=False)

    # 3D Plotting
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y', labelpad=6)
    ax.set_zlabel('z')

    x = sol_c[0, :]
    y = sol_c[1, :]
    z = sol_c[2, :]

    max_range = np.array([x.max() - x.min(), y.max() - y.min(),
                          z.max() - z.min()]).max() / 2.0
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    means = (mean_x, mean_y, mean_z)

    marker = "k"
    plot_3D_components(ax, sol_c, marker)

    pos_hist1, pos_hist2, pos_hist3, pos_hist4, pos_histc = pos_hist

    xs = []
    ys = []
    zs = []

    xc = pos_histc[:, 0]
    yc = pos_histc[:, 1]
    zc = pos_histc[:, 2]

    # plotting the spheres
    colors = ["#e31c23", "#1eb53a", "#00a3dd", "#fcd116"]
    for pos_histx, color in zip(pos_hist[:-1], colors):
        x = pos_histx[:, 0]
        y = pos_histx[:, 1]
        z = pos_histx[:, 2]

        ax.scatter(x, y, z, color=color, alpha=0.7)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # plotting gray arms at positions
    for k in range(1, len(xs[0, :]) - 1):
        for i in range(4):
            segment_x = np.array([xc[k], xs[i, k]])
            segment_y = np.array([yc[k], ys[i, k]])
            segment_z = np.array([zc[k], zs[i, k]])

            ax.plot3D(segment_x, segment_y, segment_z, "k", alpha=0.3)

    # plotting the final position with neon green arms
    for i in range(4):
        segment_x = np.array([xc[-1], xs[i, -1]])
        segment_y = np.array([yc[-1], ys[i, -1]])
        segment_z = np.array([zc[-1], zs[i, -1]])

        ax.plot3D(segment_x, segment_y, segment_z, "#CA13EB", alpha=0.7)

    # plotting the first position with purple arms
    for i in range(4):
        segment_x = np.array([xc[0], xs[i, 0]])
        segment_y = np.array([yc[0], ys[i, 0]])
        segment_z = np.array([zc[0], zs[i, 0]])

        ax.plot3D(segment_x, segment_y, segment_z, "#28D709", alpha=0.7)

    x_fct, y_fct, z_fct = aspect_ratio
    set_ax_options(ax, means, max_range, x_fct, y_fct, z_fct)
    ax.view_init(15, -65)
    fig.tight_layout()
    if FIG_NAME is not None:
        FIG_PATH = os.path.join(FIG_DIR, FIG_NAME)
        plt.savefig(FIG_PATH)
    fig.show()

if __name__ == "__main__":
    a = 0.01
    xi0 = 100
    damp_alpha0 = 1 / 100000
    eps = 1 / 1000

    n_res = 500
    n_strokes = 1

    FIG_NAME = "translation_spr4.png"
    translation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME)
    FIG_NAME = "rotation_spr4.png"
    rotation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME)
    FIG_NAME = "screw_spr4.png"
    screw(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME)
    FIG_NAME = "translation_orth_rotation_spr4.png"
    translation_orth_rotation(a, xi0, eps, n_res, n_strokes, damp_alpha0, FIG_NAME=FIG_NAME)


