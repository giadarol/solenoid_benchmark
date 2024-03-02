import xtrack as xt
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

line = xt.Line.from_json('fccee_z_thick.json')
line.cycle('ip.4', inplace=True)

tw = line.twiss(method='4d')

tt = line.get_table()
bz_data_file = './z_fieldmaps/Koratsinos_Bz_closed_before_quads.dat'

import pandas as pd
bz_df = pd.read_csv(bz_data_file, sep='\s+', skiprows=1, names=['z', 'Bz'])

s_sol_slices = np.linspace(-2.2, 2.2, 80)
bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)

P0_J = line.particle_ref.p0c[0] * qe / clight
brho = P0_J / qe / line.particle_ref.q0
ks = 0.5 * (bz_sol_slices[:-1] + bz_sol_slices[1:]) / brho
l_sol_slices = np.diff(s_sol_slices)
s_sol_slices_entry = s_sol_slices[:-1]

sol_slices = []
for ii in range(len(s_sol_slices_entry)):
    sol_slices.append(
        xt.Solenoid(length=l_sol_slices[ii]))


# plot
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(bz_df.z, bz_df.Bz, label='Bz')
plt.plot(s_sol_slices, bz_sol_slices, '.-', label='slices')
plt.xlabel('z [m]')
plt.ylabel('Bz [T]')
plt.grid()

plt.show()