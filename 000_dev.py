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

L = 10
a = 0.3
B0 = 1.5

# r = np.linspace(-0.5 * a, 0.5 * a, 100)
r = 0.001
z = np.linspace(-L*1.3, L*1.3, 100)

u = 4 * a * r / (a + r)**2
zeta_plus = z + L / 2
zeta_minus = z - L / 2
m_plus = 4 * a * r / ((a + r)**2 + zeta_plus**2)
m_minus = 4 * a * r / ((a + r)**2 + zeta_minus**2)

KK = scipy.special.ellipk
EE = scipy.special.ellipe
PP = ellipp

Bz_plus = B0 * zeta_plus / (4 * np.pi) * np.sqrt(m_plus / (a * r)) * (
                (KK(m_plus) + (a - r)/(a + r) * PP(u, m_plus)))

Bz_minus = B0 * zeta_minus / (4 * np.pi) * np.sqrt(m_minus / (a * r)) * (
                (KK(m_minus) + (a - r)/(a + r) * PP(u, m_minus)))

Bz = Bz_plus - Bz_minus