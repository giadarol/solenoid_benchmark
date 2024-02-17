import numpy as np
from pathlib import Path

from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt

ctx = xo.ContextCpu()

boris_knl_description = xo.Kernel(
    c_name='boris_step',
    args=[
        xo.Arg(xo.Int64,   name='N_sub_steps'),
        xo.Arg(xo.Float64, name='Dtt'),
        xo.Arg(xo.Float64, name='B_field', pointer=True),
        xo.Arg(xo.Float64, name='B_skew', pointer=True),
        xo.Arg(xo.Float64, name='xn1', pointer=True),
        xo.Arg(xo.Float64, name='yn1', pointer=True),
        xo.Arg(xo.Float64, name='zn1', pointer=True),
        xo.Arg(xo.Float64, name='vxn1', pointer=True),
        xo.Arg(xo.Float64, name='vyn1', pointer=True),
        xo.Arg(xo.Float64, name='vzn1', pointer=True),
        xo.Arg(xo.Float64, name='Ex_n', pointer=True),
        xo.Arg(xo.Float64, name='Ey_n', pointer=True),
        xo.Arg(xo.Float64, name='Bx_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='By_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='Bz_n_custom', pointer=True),
        xo.Arg(xo.Int64,   name='custom_B'),
        xo.Arg(xo.Int64,   name='N_mp'),
        xo.Arg(xo.Int64,   name='N_multipoles'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, name='mass'),
    ],
)

ctx.add_kernels(
    kernels={'boris': boris_knl_description},
    sources=[Path('./boris.h')],
)

p = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=1e9,
                 x=[-1e-3, 1e-3], px=[1e-3, -1e-3], py=[2e-3, -2e-3])

x = p.x.copy()
y = p.y.copy()
z = p.s.copy()

gamma = p.energy / p.mass0
mass0_kg = p.mass0 * qe / clight**2

p0c_J = p.p0c * qe

Pxc_J = p.px * p0c_J
Pyc_J = p.py * p0c_J
Pzc_J = np.sqrt((p0c_J)**2 - Pxc_J**2 - Pyc_J**2)

vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
vz = Pzc_J / clight / (gamma * mass0_kg) # m/s
