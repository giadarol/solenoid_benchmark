import numpy as np
import scipy
import matplotlib.pyplot as plt

from solenoid_field import SolenoidField

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=1)

z = np.linspace(-sf.L*1.3, sf.L*1.3, 1000)
x = 0 * z +0.01
y = 0 * z +0.02

Bx, By, Bz = sf.get_field(x, y, z)

import matplotlib.pyplot as plt
plt.close('all')

ax1 = plt.subplot(2, 1, 1)
plt.plot(z, Bz)
plt.ylabel('Bz [T]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(z, Bx, label='Bx')
plt.plot(z, By, label='By')
plt.ylabel('Br [T]')
plt.xlabel('z [m]')
plt.legend()

plt.show()