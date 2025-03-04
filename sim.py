import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    Constructs the Bogoliubov Kernel from the Hamiltonian for a two-mode (plus, minus) cavity coupled to two (plus, minus) matter modes, incorporating chiral 
    paramagnetic and diamagnetic couplings, with options for the rotating wave approximation (RWA).

    Parameters:
    ----------
    omega_plus : float
        Frequency of the plus cavity mode.
    omega_minus : float
        Frequency of the minus cavity mode.
    omega_b : float
        Frequency of the matter mode (assumed degenerate).
    g : float
        Chiral paramagnetic coupling constant. Leads to raw coupling of $1 \\pm g$.
    scale : float
        scale of the interaction strength (default=1), roughly ~ $\\sqrt{N_+}$
    fraction_minus : float, optional (default=0)
        Fraction of negative enantiomers, scaling the coupling asymmetry, roughly ~ $\\sqrt{N_-/ N_+}$
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
# in trafo, last axis is polaritons
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

def validate(kernel, trafo, inverse, energies, eps = 1e-4):
    """vectorized validation of output dict, assumes trailing hilbert space axes"""
    
    # metric
    G = get_metric(energies.shape[-1] // 2)

    # energies should be real-ish
    frac_imag = energies.imag.max() / energies.real.max()
    if frac_imag > eps:
        raise Exception(f"Energies not sufficiently real {frac_imag}")
    
    # pseudo-unitarity
    inv = get_matrix_stack(inverse)
    diff_inv = jnp.linalg.norm(inv @ G @ jnp.transpose(inv, axes = (0, 2, 1)) - G, axis = (-1, -2))
    if jnp.any(diff_inv > eps):
        raise Exception(f"Trafo not pseudo-unitary {diff_inv}")

    # diagonalizing
    energies_sorted_positive = jnp.sort(get_matrix_stack(energies, 1), axis = 1)[:, (kernel.shape[-1] // 2):]
    Omega = jnp.concatenate( [energies_sorted_positive, energies_sorted_positive], axis = 1)
    Omega = jax.vmap(jnp.diag)(Omega)
    trafo = get_matrix_stack(trafo)
    kernel = get_matrix_stack(kernel)
    diag = jnp.transpose(trafo, axes = (0, 2, 1)) @ G @ kernel @ trafo
    diff_diag = jnp.linalg.norm(diag - Omega, axis = (-1, -2))
    if jnp.any(diff_inv > eps):
        raise Exception(f"Trafo not diagonalizing {diff_diag}")


def get_matter_content(t, matter, polariton):
    return jnp.sum(jnp.abs(t[..., matter, polariton])**2, axis = -1)  / jnp.linalg.norm(t[..., matter, polariton], axis = -1)

def plot_perfect_cavity_energies():
    """plots (energy, coupling strength) for mildly chiral molecule in perfect cavity"""
    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    g = 1e-2
    scales = jnp.logspace(-2, 1, 100)

    get_kernel_vmapped = lambda g : jax.vmap(
        lambda scale : get_kernel(omega_plus,
                                  omega_minus,
                                  omega_b,
                                  g,
                                  scale=scale,
                                  anti_res = True,
                                  damping=0),
        in_axes = 0, out_axes = 0)

    get_bogoliubov_vmapped = jax.vmap(get_bogoliubov, in_axes=0, out_axes=0)

    energies_plus = get_bogoliubov_vmapped(get_kernel_vmapped(g)(scales))["energies"]
    energies_plus = jnp.sort(energies_plus, axis = 1)

    # ground state differences, similar to reproduction.py
    # energies_minus = get_bogoliubov_vmapped(get_kernel_vmapped(-g)(scales))["energies"]
    # energies_minus = jnp.sort(energies_minus, axis = 1)
    # plot_energies_plus = energies_plus[:, 4] + energies_plus[:, -1]
    # plot_energies_minus = energies_minus[:, 4] + energies_minus[:, -1]    
    # plt.plot(scales**2, (plot_energies_plus - plot_energies_minus) / 2)    
    # plt.plot(scales**2, energies_plus[:, 5] - energies_minus[:, 5])    
    # plt.plot(scales**2, energies_plus[:, -1] - energies_minus[:, -1])    

    # lower polariton
    plt.plot(scales**2, energies_plus[:, 5], label = 'Lower Polariton')
    
    # upper polariton
    plt.plot(scales**2, energies_plus[:, -1], label = 'Upper Polariton')
    
    plt.legend()
    plt.xscale('log')
    plt.show()

def plot_mixture_energies():
    """plots of (energy, fraction negative) annotated with polaritonic + matter fraction for mildly chiral molecule in perfect cavity for strong coupling"""

    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    g = 1e-2
    scale = 1e1
    fractions = jnp.linspace(0, 1, 100)

    get_kernel_vmapped = jax.vmap(
        lambda fraction : get_kernel(omega_plus,
                                     omega_minus,
                                     omega_b,
                                     g,
                                     scale=scale,
                                     fraction_minus = fraction,
                                     anti_res = True,
                                     damping=0),
        in_axes = 0, out_axes = 0)

    get_bogoliubov_vmapped = jax.vmap(get_bogoliubov, in_axes=0, out_axes=0)

    energies = get_bogoliubov_vmapped(get_kernel_vmapped(fractions))["energies"]
    
    # filter out dead mode
    energies = jnp.sort(energies, axis = 1)[:, [4,5,7]]

    plt.plot(fractions, energies)
    
    plt.legend()
    plt.show()

def plot_imperfection_energies():
    """plot of (energy, cavity imperfection) annotated with polaritonic + matter fraction for varying mixtures of mildly chiral molecule"""
    
    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    g = 1e-2
    scale = 1e1
    fraction = 1 # racemic
    dampings = jnp.linspace(0, 1, 100)

    get_kernel_vmapped = jax.vmap(
        lambda d : get_kernel(omega_plus,
                              omega_minus,
                              omega_b,
                              g,
                              scale=scale,
                              fraction_minus = fraction,
                              anti_res = True,
                              damping=d),
        in_axes = 0, out_axes = 0)

    get_bogoliubov_vmapped = jax.vmap(get_bogoliubov, in_axes=0, out_axes=0)

    energies = get_bogoliubov_vmapped(get_kernel_vmapped(dampings))["energies"]

    # no dead mode anymore
    energies = jnp.sort(energies, axis = 1)[:, 4:]

    plt.plot(dampings, energies)
    
    plt.legend()
    plt.show()

def plot_excess_matter_content():
    # Define ranges for g and scale
    g_values = [1e-1, 0.5, 1]  # Example values for g
    scale_values = [1e-1, 1e1]  # Example values for scale

    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    dampings = jnp.linspace(0, 1, 20)
    fractions = jnp.linspace(0., 1, 25)

    # Set up figure
    fig, axes = plt.subplots(len(g_values), len(scale_values), figsize=(15, 10), sharex=True, sharey=True)

    # Iterate over g and scale values to generate panels
    for i, g in enumerate(g_values):
        for j, scale in enumerate(scale_values):
            # Compute kernels for the given g and scale
            get_kernel_vmapped = jax.vmap(
                jax.vmap(
                    lambda f, d: get_kernel(omega_plus, omega_minus, omega_b, g, scale=scale, fraction_minus=f, damping=d),
                    (None, 0), 0),
                (0, None), 0)
            kernels = get_kernel_vmapped(dampings, fractions)

            # Compute Bogoliubov transformation
            get_bogoliubov_vmapped = jax.vmap(jax.vmap(get_bogoliubov, in_axes=0, out_axes=0), in_axes=1, out_axes=1)
            output = get_bogoliubov_vmapped(kernels)
            # validate(**output)            
            trafo = output["trafo"]

            # Extract matter content
            polariton = 0
            matter_plus = [2, 6]
            content_plus = get_matter_content(trafo, matter_plus, polariton)
            matter_minus = [3, 7]
            content_minus = get_matter_content(trafo, matter_minus, polariton)
            excess_content = content_plus #- jnp.nan_to_num(content_minus)
            # excess_content /= content_plus

            # Plot heatmap in the appropriate subplot
            ax = axes[i, j]
            im = ax.imshow(excess_content, 
                           aspect='auto', 
                           cmap="coolwarm", 
                           origin='lower',
                           vmin=0,
                           vmax=1,
                           extent=[fractions.min(), fractions.max(), dampings.min(), dampings.max()])

            # Set labels and titles
            if i == len(g_values) - 1:
                ax.set_xlabel("Fraction Minus")
            if j == 0:
                ax.set_ylabel("Damping")
            ax.set_title(f"g={g}, scale={scale}")

            # Add colorbar to each subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            # cbar.set_label(r"C_+ - C_-", fontsize=20, labelpad=-85)


    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':    
    plot_perfect_cavity_energies() # TODO hübsch, annotate with + matter content
    plot_mixture_energies() # TODO hübsch annotate with (relative) + matter content
    plot_imperfection_energies() # TODO hübsch annotate with (relative) + matter content
