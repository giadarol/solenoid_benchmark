import numpy as np
from pathlib import Path

from scipy.constants import c as clight
from scipy.constants import e as qe

from solenoid_field import SolenoidField

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
        xo.Arg(xo.Float64, name='mass', pointer=True),
    ],
)

ctx.add_kernels(
    kernels={'boris': boris_knl_description},
    sources=[Path('./boris.h')],
)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                 kinetic_energy0=20e9,
                #  kinetic_energy0=1e8, # to see the spiral
                 x=[-1e-3, 1e-3], px=[1e-6, -1e-6], py=[2e-3, -2e-3])

p = p0.copy()

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

dt = 1e-10
n_steps = 1000

x_log = []
y_log = []
z_log = []
px_log = []
py_log = []
pz_log = []


for ii in range(n_steps):

    x = p.x.copy()
    y = p.y.copy()
    z = p.s.copy()

    Bx, By, Bz = sf.get_field(x, y, z)

    gamma = p.energy / p.mass0
    mass0_kg = p.mass0 * qe / clight**2
    charge0_coulomb = p.q0 * qe

    p0c_J = p.p0c * qe

    Pxc_J = p.px * p0c_J
    Pyc_J = p.py * p0c_J
    Pzc_J = np.sqrt((p0c_J)**2 - Pxc_J**2 - Pyc_J**2)

    vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
    vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
    vz = Pzc_J / clight / (gamma * mass0_kg) # m/s

    ctx.kernels.boris(
            N_sub_steps=1,
            Dtt=dt,
            B_field=np.array([0.]),
            B_skew=np.array([0.]),
            xn1=x,
            yn1=y,
            zn1=z,
            vxn1=vx,
            vyn1=vy,
            vzn1=vz,
            Ex_n=0 * x,
            Ey_n=0 * x,
            Bx_n_custom=Bx,
            By_n_custom=By,
            Bz_n_custom=Bz,
            custom_B=1,
            N_mp=len(x),
            N_multipoles=0,
            charge=charge0_coulomb,
            mass=mass0_kg * gamma,
    )

    p.x = x
    p.y = y
    p.s = z
    p.px = mass0_kg * gamma * vx * clight / p0c_J
    p.py = mass0_kg * gamma * vy * clight / p0c_J

    x_log.append(p.x.copy())
    y_log.append(p.y.copy())
    z_log.append(p.s.copy())
    px_log.append(p.px.copy())
    py_log.append(p.py.copy())

x_log = np.array(x_log)
y_log = np.array(y_log)
z_log = np.array(z_log)
px_log = np.array(px_log)
py_log = np.array(py_log)

z_axis = np.linspace(0, 30, 300)
Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]

P0_J = p.p0c[0] * qe / clight
brho = P0_J / qe / p.q0

ks = 0.5 * (Bz_axis[:-1] + Bz_axis[1:]) / brho

line = xt.Line(elements=[xt.Solenoid(length=z_axis[1]-z_axis[0], ks=ks[ii])
                            for ii in range(len(z_axis)-1)])
line.build_tracker()

p_xt = p0.copy()
line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
ax1 = plt.subplot(2, 1, 1)
plt.plot(z_log, x_log, label='Boris')
plt.plot(mon.s.T, mon.x.T, '.', label='xsuite')
plt.ylabel('x [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(z_axis, Bz_axis)
plt.ylabel(r'$B_{z}$ [T]')
plt.xlabel('z [m]')

plt.show()