import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

## DEBUG
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

## PROCESSING
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
# so first, second index of
# T => matter, light
# T^{-1} => light, matter
def get_bogoliubov(kernel):
    """performs bogoliubov transformation for kernel. returns dict with keys:

    trafo: indexed by N_orig x N_polaritons
    inverse: trafo, indexed by N_polaritons x N_orig
    energies : QP energies derived from positive spectrum E_pos like [E_pos, -E_pos]
    energies_raw : QP energies from diagonalization routine

    Trafo contains matrix such that K_ij T_jk = E_k T_ik.
    Due to sorting, "bands" are automatically identified.
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
    
    return {"kernel" : kernel, "trafo" : T, "inverse" : inv, "energies" : jnp.concatenate([energies[positive], -energies[positive]]), "energies_raw" : energies}

def asymptotic_occupation(output, coupling):
    """computes the numbers of original bosons in asymptotic out state"""

    # delta peak
    def illumination(x):        
        return 1

    trafo_inv = output["inverse"]
    # 2nd index = matter index
    X = trafo_inv[:4, :4].T
    Y = trafo_inv[4:, :4].T

    energies = output["energies"][:4]
    phi = -1j * illumination(2 * energies) * coupling @ jnp.conj(X + Y) 

    xp = X @ phi
    yp = Y @ phi

    number = xp * yp + yp * jnp.conj(yp) + jnp.conj(xp) * xp + jnp.conj(xp * yp) + jnp.diag(Y @ Y.conj().T)
    number = jnp.abs(xp + yp)**2 + jnp.diag(Y @ Y.conj().T)
    return number
    
def validate(kernel, trafo, inverse, energies, energies_raw, eps = 1e-4):
    """vectorized validation of output dict, assumes trailing hilbert space axes"""    
    G = get_metric(energies.shape[-1] // 2)

    # energies should come in +/- pairs
    diff_e = jnp.linalg.norm(jnp.sort(energies, axis=-1) - jnp.sort(energies_raw, axis=-1))
    if jnp.any(diff_e > eps):
        print(f"Energies not paired {diff_e}")
    
    # energies should be real-ish
    frac_imag = energies.imag.max() / energies.real.max()
    if frac_imag > eps:
        print(f"Energies not sufficiently real {frac_imag}")
    
    # pseudo-unitarity
    inv = get_matrix_stack(inverse)
    diff_inv = jnp.linalg.norm(inv @ G @ jnp.transpose(inv, axes = (0, 2, 1)) - G, axis = (-1, -2))
    if jnp.any(diff_inv > eps):
        print(f"Trafo not pseudo-unitary {diff_inv}")

    # diagonalizing
    Omega = jax.vmap(jnp.diag)(energies)
    trafo = get_matrix_stack(trafo)
    kernel = get_matrix_stack(kernel)
    diag = jnp.transpose(trafo, axes = (0, 2, 1)) @ G @ kernel @ trafo
    diff_diag = jnp.linalg.norm(diag - Omega, axis = (-1, -2))
    if jnp.any(diff_inv > eps):
        print(f"Trafo not diagonalizing {diff_diag}")

def get_content(t, idxs, polariton):
    nom = jnp.sum(jnp.abs(t[..., idxs, polariton]**2), axis = -1)
    denom = jnp.linalg.norm(t[..., polariton], axis = -1)**2
    return nom / denom

## PLOTTING
def add_segment(ax, x, y, colors, mi = 0, mx = 1):        
    points = jnp.array([x, y]).T.reshape(-1, 1, 2)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(mi, mx)
    lc = LineCollection(segments, cmap='magma', norm=norm)
    lc.set_array(colors)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    
    # auto-adjust plot limits by invisible line
    ax.plot(x, y, alpha=0.0)
    return lc

def plot_coupling_energies():
    """plots (energy, coupling strength) for mildly chiral molecule in perfect cavity"""
    omega_plus = 0.5
    omega_minus = 0.5
    omega_b = 1
    g = 1e-2
    scales = jnp.logspace(-1, 0.5, 30)

    get_kernel_vmapped = lambda g : jax.vmap(
        lambda scale : get_kernel(omega_plus,
                                  omega_minus,
                                  omega_b,
                                  g,
                                  scale=scale,
                                  anti_res = True,                            
                                  damping=0),
        in_axes = 0, out_axes = 0)

    get_bogoliubov_vmapped = jax.vmap(
        lambda k : get_bogoliubov(k),
        in_axes=0,
        out_axes=0)

    kernels = get_kernel_vmapped(g)(scales)
    output = get_bogoliubov_vmapped(kernels)
    validate(**output)    
    energies = output["energies"].real

    trafo = output["trafo"]    
    light_idxs = [0, 1, 4, 5]
    matter_idxs = [2, 6]
    polaritons = jnp.arange(8)
    content_plus = jax.vmap(lambda p : get_content(trafo, matter_idxs, p) )(polaritons)
    
    fig, ax = plt.subplots(1, 1)    

    mi, mx = content_plus.min(), content_plus.max()
    
    idx = 0
    line = add_segment(ax, scales, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
    line.set_label('Lower Polariton')
    line.set_linestyle('--')

    idx = 3
    line = add_segment(ax, scales, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
    line.set_label('Upper Polariton')

    ax.set_xlabel(r'Coupling Strength $\sim \sqrt{N}$')
    ax.set_ylabel(r'E / $\omega_b$')
    
    fig.colorbar(line, ax=ax, label="Matter Content")
    
    plt.legend()
    # plt.xscale('log')
    # plt.show()
    plt.savefig("energy_coupling.pdf")

    # fig, axs = plt.subplots(1, 1)
    # ax = axs
    # idxs = [0, 1, 2, 3]
    # mi, mx = content_plus.min(), content_plus.max()
    # for idx in idxs:
    #     ann = content_plus[idx]
    #     line = add_segment(ax, scales, energies[:, idx], ann, mi = mi, mx = mx)        

    # ax.set_xlabel(r'Coupling Strength $\sim \sqrt{N}$')
    # ax.set_ylabel(r'E / $\omega_b$')    
    # fig.colorbar(line, ax=ax, label="Positive Matter Content")    
    # plt.legend()
    # plt.savefig("energy_coupling.pdf")
    
def plot_fraction_energies():
    """plots of (energy, fraction negative) annotated with polaritonic + matter fraction for mildly chiral molecule in perfect cavity for strong coupling"""

    omega_plus = 0.5
    omega_minus = 0.5
    omega_b = 1
    g = 1e-2
    scale = 0.5
    fractions = jnp.linspace(0, 1, 100)

    get_kernel_vmapped = jax.vmap(
        lambda f : get_kernel(omega_plus,
                              omega_minus,
                              omega_b,
                              g,
                              scale=scale,
                              anti_res = True,
                              fraction_minus = f,
                              damping=0),
        in_axes = 0, out_axes = 0)

    get_bogoliubov_vmapped = jax.vmap(
        lambda k : get_bogoliubov(k),
        in_axes=0,
        out_axes=0)

    kernels = get_kernel_vmapped(fractions)
    output = get_bogoliubov_vmapped(kernels)
    validate(**output)    
    energies = output["energies"].real

    trafo = output["trafo"]    
    light_idxs = [0, 1, 4, 5]
    matter_idxs = [2, 6]
    polaritons = jnp.arange(8)
    content_plus = jax.vmap(lambda p : get_content(trafo, matter_idxs, p) )(polaritons)
    
    matter_idxs = [3, 7]
    content_minus = jax.vmap(lambda p : get_content(trafo, matter_idxs, p) )(polaritons)
    
    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    idxs = [0, 1, 2, 3]
    content_plus = content_plus[jnp.array(idxs)]
    mi, mx = content_plus.min(), content_plus.max()
    for idx in idxs:
        ann = content_plus[idx]
        line = add_segment(ax, fractions, energies[:, idx], ann, mi = mi, mx = mx)        

    ax.set_xlabel(r'$\frac{N_+}{N_-}$')
    ax.set_ylabel(r'E / $\omega_b$')    
    fig.colorbar(line, ax=ax, label="Positive Matter Content")    
    
    ax = axs[1]
    idxs = [0, 1, 2, 3]
    content_minus = content_minus[jnp.array(idxs)]
    mi, mx = content_minus.min(), content_minus.max()
    for idx in idxs:
        ann = content_minus[idx]
        line = add_segment(ax, fractions, energies[:, idx], ann, mi = mi, mx = mx)
        
    ax.set_xlabel(r'$\frac{N_+}{N_-}$')
    ax.set_ylabel(r'E / $\omega_b$')    
    fig.colorbar(line, ax=ax, label="Negative Matter Content")    
    plt.legend()
    plt.savefig("energy_fraction.pdf")


def plot_damping_energies():
    """plot of (energy, cavity imperfection) annotated with polaritonic + matter fraction for varying mixtures of mildly chiral molecule"""
    omega_plus = 0.5
    omega_minus = 0.5
    omega_b = 1
    g = 1e-2
    scale = 1
    fraction = 1
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

    get_bogoliubov_vmapped = jax.vmap(
        lambda k : get_bogoliubov(k),
        in_axes=0,
        out_axes=0)

    kernels = get_kernel_vmapped(dampings)
    output = get_bogoliubov_vmapped(kernels)
    # validate(**output)    
    energies = output["energies"].real
    
    trafo = output["trafo"]    
    light_idxs = [0, 1, 4, 5]
    matter_idxs = [2, 6]
    polaritons = jnp.arange(8)
    content_plus = jax.vmap(lambda p : get_content(trafo, matter_idxs, p) )(polaritons)
    
    matter_idxs = [3, 7]
    content_minus = jax.vmap(lambda p : get_content(trafo, matter_idxs, p) )(polaritons)
    
    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    idxs = [0, 1, 2, 3]
    idxs = [3]
    content_plus = content_plus[jnp.array(idxs)]
    mi, mx = content_plus.min(), content_plus.max()
    for idx in idxs:
        ann = content_plus[idx]
        line = add_segment(ax, dampings, energies[:, idx], ann, mi = mi, mx = mx)        

    ax.set_xlabel(r'Damping')
    ax.set_ylabel(r'E / $\omega_b$')    
    fig.colorbar(line, ax=ax, label="Positive Matter Content")    
    plt.legend()

    ax = axs[1]
    idxs = [0, 1, 2, 3]
    idxs = [3]
    content_minus = content_minus[jnp.array(idxs)]
    mi, mx = content_plus.min(), content_plus.max()
    for idx in idxs:
        ann = content_minus[idx]
        line = add_segment(ax, dampings, energies[:, idx], ann, mi = mi, mx = mx)        

    ax.set_xlabel(r'Damping')
    ax.set_ylabel(r'E / $\omega_b$')    
    fig.colorbar(line, ax=ax, label="Negative Matter Content")    
    plt.legend()
    plt.savefig("energy_damping.pdf")

def plot_asymptotic_occupation():
    """plots asymptotic occupation for racemic mixture"""    
    omega_plus = 1
    omega_minus = 1
    omega_b = 1.2
    g = 1e-2
    scale = 0.5
    fraction = 1
    damping = 1
    
    # c_- / c_+
    scale = 1
    coupling_ratios = scale * jnp.linspace(0, 1, 100)

    # reshape to matrix, couples to light only
    one = scale * jnp.ones_like(coupling_ratios)
    zero = 0 * one
    coupling = jnp.stack([one, coupling_ratios, zero, zero])

    kernel = get_kernel(omega_plus,
                        omega_minus,
                        omega_b,
                        g,
                        scale = scale,
                        fraction_minus = fraction,
                        anti_res = True,
                        damping = damping)    
    output = get_bogoliubov(kernel)

    # validate(**output)    
    get_occ_vmapped = jax.vmap(lambda c : asymptotic_occupation(output, coupling = c), in_axes = 1, out_axes = 0)    
    occ = get_occ_vmapped(coupling)
    print(jnp.abs(occ.imag).max())
    occ = occ.real
    
    occ_plus = occ[:, 2]
    occ_minus = occ[:, 3]

    
    fig, axs = plt.subplots(1, 1)

    ax = axs
    # ax.plot(coupling_ratios, occ)
    ax.plot(coupling_ratios, occ_plus, label = r'$\langle n_+ \rangle$')
    ax.plot(coupling_ratios, occ_minus, '--', label = r'$\langle n_- \rangle$')
    # ax.plot(coupling_ratios, occ_plus - occ_minus, '-.')
    ax.set_xlabel(r'$c_- / c_+$')
    ax.set_ylabel(r'$\langle n \rangle$')    
    plt.legend()
    plt.show()
    # plt.savefig("occupation_coupling.pdf")

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
    # plot_coupling_energies() # TODO pub-hübsch
    # plot_fraction_energies() # TODO pub-hübsch 
    # plot_damping_energies() # TODO pub-hübsch
    plot_asymptotic_occupation()  # TODO pub-hübsch, check if correct
