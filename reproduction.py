import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sim import *

## reproduction 
def test_prl():
    # DOI: 10.1103/PhysRevLett.112.016401
    # shows decoupling of polaritons into pure matter / pure light excitations when light-matter coupling is strong enough, bc A^2 term dominates there and this minimizes it
    omega_c = 1/1.7
    omega = 1.    
    scales = jnp.logspace(-2, 1, 100) # 0.01 => 10

    def _get_matter_content(s):
        # corresponds to single matter and cavity mode, reproduce with completely chiral mode
        k = get_kernel(omega_c, omega_c, omega, g = 1, scale = s, diamagnetic = True, anti_res = True)

        x = get_bogoliubov(k)["trafo"]

        # + only couples to + mode
        matter_idxs = [2, 6]
        polariton_idx = 0
        matter_content = jnp.abs(x[matter_idxs, polariton_idx]**2).sum() / jnp.linalg.norm(x[:, polariton_idx])**2

        return matter_content

    # content = jax.vmap(get_matter_content)(scales)
    content = []
    for s in scales:
        content.append(_get_matter_content(s))

    plt.plot(scales, content)
    plt.xscale('log')
    plt.show()


def test_jpcl():
    # J. Phys. Chem. Lett. 2023, 14, 3777−3784
    # shows difference between polaritonic ground and excited state energies for single-mode chiral cavity with singly chiral molecules inside when number of molecules is increased

    omega_c = 1.
    omega = 1.
    g = 1e-2

    # ~ sqrt(number) of molecules
    scales = jnp.logspace(-2, 1, 100)
    # linear scaling of gs discrimination for small number of molecules jnp.logspace(0, 1, 100)

    def get_energy_deltas(s):
        # corresponds to different enantiomers placed separately into perfectly chiral cavity at resonance

        # "reproduce" by zeroing out one mode, setting negative coupling to this mode for other chirality    
        k_plus = get_kernel(omega_c, 0, omega, g = g, scale = s, diamagnetic = True, anti_res = True, damping = 0)
        energies_plus = get_bogoliubov(k_plus)["energies"]

        k_minus = get_kernel(omega_c, 0, omega, g = -g, scale = s, diamagnetic = True, anti_res = True, damping = 0)
        energies_minus = get_bogoliubov(k_minus)["energies"]

        # energies come in pairs, one of them ios decoupled => 0
        energies_minus = jnp.sort(energies_minus)[5:]    
        energies_plus = jnp.sort(energies_plus)[5:]
        return energies_plus, energies_minus

    e_p, e_m = [], []
    for s in scales:
        ep, em = get_energy_deltas(s)
        e_p.append(ep)
        e_m.append(em)

    e_p = jnp.array(e_p)
    e_m = jnp.array(e_m)

    plt.plot(scales**2, (e_p - e_m)[:, [0, -1]] )
    delta_vac = e_p[:, 0] + e_p[:, -1] - (e_m[:, 0] + e_m[:, -1])
    plt.plot(scales**2, delta_vac / 2, '--')

    # see individual branches
    # plt.plot(scales**2, e_p[:, 0] + e_p[:, -1], '-')
    # plt.plot(scales**2, e_m[:, 0] + e_m[:, -1], '--')

    # turn off to see sqrt / linear scaling
    plt.xscale('log')
    
    plt.show()
    
test_prl()
# test_jpcl()
