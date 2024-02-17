import numpy as np
import scipy
import matplotlib.pyplot as plt

# https://par.nsf.gov/servlets/purl/10220882

def ellipp(n, m):
    """
    Elliptic integral of the third kind
    """

from scipy.special import elliprf, elliprj

def ellipp(n, m):
    assert (m <= 1).all()
    y = 1 - m
    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3

L = 4
a = 0.3
B0 = 1.5

# r = np.linspace(-0.5 * a, 0.5 * a, 100)
z = np.linspace(-L*1.3, L*1.3, 1000)
r = 0 * z +0.01

u = 4 * a * r / (a + r)**2
zeta_plus = z + L / 2
zeta_minus = z - L / 2
m_plus = 4 * a * r / ((a + r)**2 + zeta_plus**2)
m_minus = 4 * a * r / ((a + r)**2 + zeta_minus**2)

# sqrt_plus = sqrt(m_plus / (a * r))
sqrt_plus = np.sqrt(4 / ((a + r)**2 + zeta_plus**2))

# sqrt_minus = sqrt(m_minus / (a * r))
sqrt_minus = np.sqrt(4 / ((a + r)**2 + zeta_minus**2))

KK = scipy.special.ellipk
EE = scipy.special.ellipe
PP = ellipp

Bz_plus = B0 * zeta_plus / (4 * np.pi) * sqrt_plus * (
                (KK(m_plus) + (a - r)/(a + r) * PP(u, m_plus)))

Bz_minus = B0 * zeta_minus / (4 * np.pi) * sqrt_minus * (
                (KK(m_minus) + (a - r)/(a + r) * PP(u, m_minus)))

Bz = Bz_plus - Bz_minus

sqrt_r_plus = 0 * sqrt_plus
sqrt_r_minus = 0 * sqrt_minus

mask_r_nonzero = r > 1e-11
sqrt_r_plus[mask_r_nonzero] = np.sqrt(a / (r[mask_r_nonzero] * m_plus[mask_r_nonzero]))
sqrt_r_minus[mask_r_nonzero] = np.sqrt(a / (r[mask_r_nonzero] * m_minus[mask_r_nonzero]))

Br_plus = B0 / np.pi * sqrt_r_plus * (EE(m_plus) - (1 - m_plus / 2) * KK(m_plus))
Br_minus = B0 / np.pi * sqrt_r_minus * (EE(m_minus) - (1 - m_minus / 2) * KK(m_minus))

Br = Br_plus - Br_minus


import matplotlib.pyplot as plt
plt.close('all')

ax1 = plt.subplot(2, 1, 1)
plt.plot(z, Bz)
plt.ylabel('Bz [T]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(z, Br)
plt.ylabel('Br [T]')

plt.show()