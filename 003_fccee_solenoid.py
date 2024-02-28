import numpy as np
from pathlib import Path

from scipy.constants import c as clight
from scipy.constants import e as qe

from solenoid_field import SolenoidField, Multifield

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
                 kinetic_energy0=45.5e9, # Z pole
                #  kinetic_energy0=1e9, # Z pole
                 s=-4., x=-4*0.015, px=0.015)


p = p0.copy()

sol = SolenoidField(L=2, a=0.05, B0=2., z0=0)
asol_left = SolenoidField(L=1, a=0.05, B0=-3, z0=-1.5)
asol_right = SolenoidField(L=1, a=0.05, B0=-3, z0=1.5)

sf = Multifield([sol, asol_left, asol_right])

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

z_axis = np.linspace(p0.s[0], -p0.s[0], 300)
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

# Wolsky Eq. 3.114
Ax = -0.5 * Bz_axis * mon.y
Ay =  0.5 * Bz_axis * mon.x

# Wolsky Eq. 2.74
ax = Ax * p0.q0 * qe / P0_J
ay = Ay * p0.q0 * qe / P0_J

px_mech = mon.px - ax
py_mech = mon.py - ay

pz_mech = np.sqrt((1 + mon.delta)**2 - px_mech**2 - py_mech**2)

xp = px_mech / pz_mech
yp = py_mech / pz_mech

dx_ds = np.diff(mon.x, axis=1) / np.diff(mon.s, axis=1)
dx_ds_boris = np.diff(x_log, axis=0) / np.diff(z_log, axis=0)

dy_ds = np.diff(mon.y, axis=1) / np.diff(mon.s, axis=1)
dy_ds_boris = np.diff(y_log, axis=0) / np.diff(z_log, axis=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(z_log, x_log, label='Boris')
plt.plot(mon.s.T, mon.x.T, '.', label='xsuite')
plt.ylabel('x [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(z_axis, Bz_axis)
plt.ylabel(r'$B_{z}$ [T]')
plt.xlabel('z [m]')

plt.figure(2)

plt.plot(mon.s.T, xp.T, label="x'", color='C0', linestyle='-')
plt.plot(mon.s[:, :-1].T, dx_ds.T, '.', label=r"$\Delta x / \Delta s$", color='C1')
plt.plot(z_log[:-1, :], dx_ds_boris, 'x', label=r"$\Delta x / \Delta s$ Boris", color='C2')
plt.plot(mon.s.T, mon.px.T, '--', label=r"$p_x$", color='C2')
plt.legend()

plt.figure(3)

plt.plot(mon.s.T, yp.T, label="y'", color='C0', linestyle='-')
plt.plot(mon.s[:, :-1].T, dy_ds.T, '.', label=r"$\Delta y / \Delta s$", color='C1')
plt.plot(z_log[:-1, :], dy_ds_boris, 'x', label=r"$\Delta y / \Delta s$ Boris", color='C2')
plt.plot(mon.s.T, mon.py.T, '--', label=r"$p_y$", color='C2')
plt.legend()


plt.show()