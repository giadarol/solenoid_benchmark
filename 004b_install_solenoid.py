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

l_solenoid = 4.4
ds_sol_start = -2.2
ds_sol_end = 2.2
ip_sol = 'ip.1'

theta_tilt = 15e-3 # rad

s_sol_slices = np.linspace(-2.2, 2.2, 81)
bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)

P0_J = line.particle_ref.p0c[0] * qe / clight
brho = P0_J / qe / line.particle_ref.q0
ks = 0.5 * (bz_sol_slices[:-1] + bz_sol_slices[1:]) / brho
l_sol_slices = np.diff(s_sol_slices)
s_sol_slices_entry = s_sol_slices[:-1]

sol_slices = []
for ii in range(len(s_sol_slices_entry)):
    sol_slices.append(xt.Solenoid(length=l_sol_slices[ii], ks=0)) # Off for now

s_ip = tw['s', ip_sol]

line.discard_tracker()
line.insert_element(name='sol_start_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_start)
line.insert_element(name='sol_end_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_end)

sol_start_tilt = xt.YRotation(angle=-theta_tilt * 180 / np.pi)
sol_end_tilt = xt.YRotation(angle=+theta_tilt * 180 / np.pi)
sol_start_shift = xt.XYShift(dx=l_solenoid/2 * np.sin(theta_tilt))
sol_end_shift = xt.XYShift(dx=l_solenoid/2 * np.sin(theta_tilt))

line.element_dict['sol_start_tilt_'+ip_sol] = sol_start_tilt
line.element_dict['sol_end_tilt_'+ip_sol] = sol_end_tilt
line.element_dict['sol_start_shift_'+ip_sol] = sol_start_shift
line.element_dict['sol_end_shift_'+ip_sol] = sol_end_shift


sol_slice_names = []
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_dict[nn] = sol_slices[ii]
    sol_slice_names.append(nn)

tt = line.get_table()
names_upstream = list(tt.rows[:'sol_start_'+ip_sol].name)
names_downstream = list(tt.rows['sol_end_'+ip_sol:].name[:-1]) # -1 to exclude '_end_point' added by the table

element_names = (names_upstream
                 + ['sol_start_tilt_'+ip_sol, 'sol_start_shift_'+ip_sol]
                 + sol_slice_names
                 + ['sol_end_shift_'+ip_sol, 'sol_end_tilt_'+ip_sol]
                 + names_downstream)

line.element_names = element_names

# re-insert the ip
line.element_dict.pop(ip_sol)
line.insert_element(name=ip_sol, element=xt.Marker(), at_s=s_ip)
line.build_tracker()

# Set strength
line.vars['on_sol_'+ip_sol] = 0
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_refs[nn].ks = ks[ii] * line.vars['on_sol_'+ip_sol]

tt = line.get_table()

tt.rows['sol_start_ip.1':'sol_end_ip.1'].show()

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