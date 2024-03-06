import xtrack as xt
from scipy.constants import c as clight
from scipy.constants import e as qe
import numpy as np

p = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                     energy0=45.5e9)

Bz = 10. # T
P0_J = p.p0c[0] * qe / clight
brho = P0_J / qe / p.q0

ks = Bz / brho

line = xt.Line(elements=[xt.Solenoid(length=0.1, ks=ks)])
line.particle_ref = p
line.build_tracker()
line.configure_radiation(model='mean')

particle_on_co = line.particle_ref.copy()
R = line.compute_one_turn_matrix_finite_differences(particle_on_co=particle_on_co)

particle_on_co.x = 1e-3

p_co_before = particle_on_co.copy()
line.track(particle_on_co)
p_co_after = particle_on_co.copy()

from xtrack.twiss import _compute_eneloss_and_damping_rates

damp = _compute_eneloss_and_damping_rates(
    particle_on_co=particle_on_co,
    R_matrix=R['R_matrix'],
    W_matrix=None,
    px_co=None, py_co=None,
    ptau_co=np.array([p_co_before.ptau[0], p_co_after.ptau[0]]),
    T_rev0=1000000, line=line, radiation_method='mean')