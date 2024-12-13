import numpy as np
from scipy.spatial.transform import Rotation as Rot

def plot_3D_components(ax, sol, marker=None, opacity=1):
    x = sol[0, :]
    y = sol[1, :]
    z = sol[2, :]
    if marker is not None:
        ax.plot3D(x, y, z, marker, alpha=opacity)
    else:
        ax.plot3D(x, y, z, alpha=opacity)


def set_ax_options(ax, means, max_range, x_fct=1, y_fct=1, z_fct=1):
    mean_x, mean_y, mean_z = means

    ax.set_xlim(mean_x - max_range / x_fct, mean_x + max_range / x_fct)
    ax.set_ylim(mean_y - max_range / y_fct, mean_y + max_range / y_fct)
    ax.set_zlim(mean_z - max_range / z_fct, mean_z + max_range / z_fct)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
    ax.set_box_aspect((1, 1, 1))


def rescale_coordinates(u, fcts):
    fctx, fcty, fctz = fcts
    scalex = 1 / fctx
    scaley = 1 / fcty
    scalez = 1 / fctz

    resx = scalex * u[0]
    resy = scaley * u[1]
    resz = scalez * u[2]

    return np.array([resx, resy, resz])


def amp_angle(R_matrix, amp):
    R = Rot.from_matrix(R_matrix)
    rotvec = R.as_rotvec()
    amp_rotvec = amp * rotvec

    amp_R = Rot.from_rotvec(amp_rotvec)
    return amp_R.as_matrix()

