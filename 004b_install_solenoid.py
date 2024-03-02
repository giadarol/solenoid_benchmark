import xtrack as xt
import numpy as np

line = xt.Line.from_json('fccee_z_thick.json')
line.cycle('ip.4', inplace=True)

tw = line.twiss(method='4d')

tt = line.get_table()
bz_data_file = './z_fieldmaps/Koratsinos_Bz_closed_before_quads.dat'

import pandas as pd
bz_df = pd.read_csv(bz_data_file, sep='\s+', skiprows=1, names=['z', 'Bz'])

s_sol_slices = np.linspace(-2.2, 2.2, 80)
bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)

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