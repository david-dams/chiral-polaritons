import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

# TODO: static case: (damping, fraction_minus (?)) ~ matter content (+-) in lowest polariton mode
# TODO: static case: (damping, fraction_minus (?)) ~ delta_{+-}
# TODO: dynamic analysis: (c_+/-, fraction_minus) ~ energy deposited

# TODO: compare hopfield RWA and full
# TODO: knobs to tune: g, w_+/-, w_b, fraction_minus, c_+/-, damping

def get_matrix_stack(array, n = 2):
    shape = array.shape  
    X = int(jnp.prod(jnp.array(shape[:-n])))    
    return array.reshape(X, *(shape[-i-1] for i in range(n)))

def get_metric(n):
    return jnp.diag(jnp.concatenate([jnp.ones(n), -jnp.ones(n)]))

def get_converted(M):
    return get_metric(M.shape[-1] // 2) @ M

def get_kernel(omega_plus, omega_minus, omega_b, g, scale = 1., fraction_minus = 0, diamagnetic = True, anti_res = False, damping = 1.):
    """
    Constructs the Bogoliubov Kernel from the Hamiltonian for a two-mode (plus, minus) cavity coupled to a matter mode, incorporating chiral 
    paramagnetic and diamagnetic couplings, with options for the rotating wave approximation (RWA).

    The Hamiltonian is structured as:
    XXX

    where:
    - \(i, n\) index the cavity and matter modes.
    - Matter modes are assumed degenerate.
    - \(g_{in}\) and \(D_{ij}\) are chiral paramagnetic and diamagnetic couplings.
    - The coupling strength is influenced by the fraction of negative enantiomers.
    - Diamagnetic interactions contribute to self-interaction terms.

    The Kernel is given by metric @ H.

    Parameters:
    ----------
    omega_plus : float
        Frequency of the plus cavity mode.
    omega_minus : float
        Frequency of the minus cavity mode.
    omega_b : float
        Frequency of the matter mode (assumed degenerate).
    g : float
        Chiral paramagnetic coupling constant.
    scale : float
        scale of the interaction strength (default=1), roughly ~ \sqrt{N_+}
    fraction_minus : float, optional (default=0)
        Fraction of negative enantiomers, scaling the coupling asymmetry, roughly ~ $\sqrt{N_-/ N_+}$
    diamagnetic : bool, optional (default=True)
        If True, includes the diamagnetic coupling contributions.
    anti_res : bool, optional (default=False)
        If False, applies the rotating wave approximation (RWA), neglecting antiresonant terms.
    damping : float, optional (default=1)
        Damping of the negative helicity mode

    Returns:
    -------
    jax.numpy.ndarray
        The Hamiltonian matrix with the Bogoliubov transformation structure, incorporating cavity-matter couplings,
        self-interactions, and optional diamagnetic contributions.
    """

    # magnetic couplings, interaction ~ (g_+, g_- \sqrt{N_- / N_+}) \sqrt{N_+}
    g_matrix = jnp.array( [
        [1 + g, 1 - g ],
        [1 - g, 1 + g ]
        ]
    ) * jnp.array( [1, fraction_minus] ) * scale
    dampings = jnp.array( [1, damping] )    
    g_matrix *= dampings[:, None]
    diamagnetic = 2 * g_matrix.sum(axis = 1) * g_matrix.sum(axis = 1)[:, None]  / omega_b * (diamagnetic == True)
    
    # coupling is (c+ c) M (c c+) with c = (a+ a- b+ b-)
    # leading to block structure
    # M = [ [ A B ], [conj(B) conj(A) ] ]
    # A = [ [h_light_ca h_light_matter_ca], [dagger(h_light_matter_ca) h_matter_ca] ]
    # B = [ [h_light_cc h_light_matter_cc], [dagger(h_light_matter_cc) h_matter_cc] ]
    
    # a^{\dagger}a - term
    h_light_ca = jnp.diag( jnp.array([omega_plus, omega_minus]) ) + diamagnetic
    
    # b^{\dagger}b - term
    h_matter_ca = jnp.eye(2) * omega_b

    # RESONANT b^{\dagger}a - term
    h_light_matter_ca = g_matrix
    
    # together in A-type matrix
    A = jnp.block(  [ [h_light_ca, h_light_matter_ca], [h_light_matter_ca.conj().T, h_matter_ca] ] )
    
    # a^{\dagger}a^{\dagger} - term
    h_light_cc = diamagnetic * (anti_res == True)
    
    # b^{\dagger}b^{\dagger} - term    
    h_matter_cc = jnp.zeros((2,2))

    # OFFRESONANT b^{\dagger}a^{\dagger} - term
    h_light_matter_cc = g_matrix * (anti_res == True)

    # together in B-type matrix
    B = jnp.block(  [ [h_light_cc, h_light_matter_cc], [h_light_matter_cc.conj().T, h_matter_cc] ] )

    # build hamiltonian
    hamiltonian = jnp.block([ [A, B], [B.conj(), A.conj()] ] )

    # apply metric
    kernel = get_converted(hamiltonian)

    return kernel

# bosonic bogoliubov
# G = diag(1, -1)
# diagonalize G H
# from evs T = [x_1, ... J x_1, ...], where J (u v) = (conj(v) conj(u))
# => T^{\dagger} (GH) T diagonalizes, so T : matter, light => polaritons
# so we need T^{-1} : polaritons => matter, light
# construct by taking the positive eigenvectors
def get_bogoliubov(kernel):
    """bogoliubov transformation matrix $T$ for a kernel M, i.e. the matrix that diagonalizes $H = Ma a^{\\dagger}$ via $a' = T a$ obeying $TgT^{\\dagger} = g$

    returns

    trafo: indexed by N_orig x N_polaritons
    inverse: trafo, indexed by N_polaritons x N_orig
    energies :    
    """
    n = kernel.shape[0]//2
    
    # "metric"
    G = get_metric(n)    

    # diagonalize
    energies, vecs = jnp.linalg.eig(kernel)    

    # positive (ev dist symmetric around zero)
    positive = jnp.argsort(energies)[n:]
    
    # x is N x N / 2
    x = vecs[:, positive]
    
    # renorming is crucial (cf bf discussion)
    prod = x.T @ G @ x
    x = x / jnp.sqrt(jnp.diagonal(prod))

    # Jx is N x N / 2
    Jx = jnp.concatenate([x[n:], x[:n]])
    
    # T = [x_1, ..., J x_1, ...]
    T = jnp.concatenate([x, Jx], axis = 1)

    # inverse = G T^{\dagger} G
    inv = G @ T.T @ G

    return {"kernel" : kernel, "trafo" : T, "inverse" : inv, "energies" : energies}

def validate(kernel, trafo, inverse, energies):
    # metric
    G = get_metric(energies.shape[-1] // 2)

    # energies should be real-ish
    frac_imag = energies.imag.max() / energies.real.max()
    if frac_imag > 0.1:
        raise Exception(f"Energies not sufficiently real {frac_imag}")
    
    # pseudo-unitarity
    inv = get_matrix_stack(inverse)
    diff_inv = jnp.linalg.norm(inv @ G @ jnp.transpose(inv, axes = (0, 2, 1)) - G, axis = (-1, -2))
    if jnp.any(diff_inv > 0.9):
        raise Exception(f"Trafo not pseudo-unitary {diff_inv}")

    # diagonalizing
    energies_sorted_positive = jnp.sort(get_matrix_stack(energies, 1), axis = 1)[:, (kernel.shape[-1] // 2):]
    Omega = jnp.concatenate( [energies_sorted_positive, energies_sorted_positive], axis = 1)
    Omega = jax.vmap(jnp.diag)(Omega)
    trafo = get_matrix_stack(trafo)
    kernel = get_matrix_stack(kernel)
    diag = jnp.transpose(trafo, axes = (0, 2, 1)) @ G @ kernel @ trafo
    diff_diag = jnp.linalg.norm(diag - Omega, axis = (-1, -2))
    if jnp.any(diff_inv > 0.9):
        raise Exception(f"Trafo not diagonalizing {diff_diag}")    
    

omega_plus = 1
omega_minus = 1
omega_b = 1
g = 1

get_kernel_vmapped = jax.vmap(jax.vmap( lambda f, d: get_kernel(omega_plus, omega_minus, omega_b, g, fraction_minus = f, damping = d), (None, 0), 0), (0, None), 0)

dampings = jnp.linspace(0, 1, 10)
fractions = jnp.linspace(0, 1, 20)

kernels = get_kernel_vmapped(dampings, fractions)

# i x j x N x N -> i x j x N x N
get_bogoliubov_vmapped = jax.vmap(jax.vmap(get_bogoliubov, in_axes=0, out_axes=0), in_axes=1, out_axes=1)
output = get_bogoliubov_vmapped(kernels)
validate(**output)





