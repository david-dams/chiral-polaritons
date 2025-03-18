import jax
import jax.numpy as jnp

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

def get_kernel(omega_plus, omega_minus, omega_b, g, gamma = 1., fraction_minus = 0, diamagnetic = True, anti_res = False, damping = 1.):
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
    gamma : float
        scale of the positive interaction strength (default=1) ~ $\\sqrt{\frac{N_+ \omega_+}{2 \epsilon_0 V}$
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
    ) * jnp.array( [1, fraction_minus] ) * gamma
    dampings = jnp.array( [1, damping] )    
    g_matrix *= dampings[:, None]
    # g_matrix *= 1j
    diamagnetic = 2 * g_matrix.sum(axis = 1) * g_matrix.conj().sum(axis = 1)[:, None]  / omega_b * (diamagnetic == True)
    
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

def get_bogoliubov_vmapped(kernels):
    return jax.vmap(lambda k : get_bogoliubov(k), in_axes=0, out_axes=0)(kernels)

def get_asymptotic_occupation(output, coupling):
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

def plot_gamma_energies():
    """plots (energy, coupling strength) for mildly chiral molecule in perfect cavity"""
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    g = 1e-2
    gammas = jnp.logspace(-1, 0.5, 30)

    get_kernel_vmapped = lambda g : jax.vmap(
        lambda gamma : get_kernel(omega_plus,
                                  omega_minus,
                                  omega_b,
                                  g,
                                  gamma=gamma,
                                  anti_res = True,                            
                                  damping=0),
        in_axes = 0, out_axes = 0)

    kernels = get_kernel_vmapped(g)(gammas)
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
    line = add_segment(ax, gammas, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
    line.set_label('Lower Polariton')
    line.set_linestyle('--')

    idx = 3
    line = add_segment(ax, gammas, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
    line.set_label('Upper Polariton')

    ax.set_xlabel(r'Coupling Strength $\sim \sqrt{N}$')
    ax.set_ylabel(r'E / $\omega_b$')
    
    fig.colorbar(line, ax=ax, label="Matter Content")
    
    plt.legend()
    # plt.xgamma('log')
    # plt.show()
    plt.savefig("energy_gamma.pdf")

    
def plot_fraction_energies():
    """plots (energy, fraction negative) annotated with polaritonic + matter fraction
    for mildly chiral molecule in perfect cavity for strong coupling, looping over different g"""

    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    g_values = [1e-2]  # Looping over these g values
    gamma = 0.1
    fractions = jnp.linspace(0, 1, 100)

    # Define custom settings for plots
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 10,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig, axs = plt.subplots(1, 1)

        for i, g in enumerate(g_values):
            get_kernel_vmapped = jax.vmap(
                lambda f: get_kernel(omega_plus,
                                     omega_minus,
                                     omega_b,
                                     g=g,
                                     gamma=gamma,
                                     anti_res=True,
                                     fraction_minus=f,
                                     damping=0),
                in_axes=0, out_axes=0)

            kernels = get_kernel_vmapped(fractions)
            output = get_bogoliubov_vmapped(kernels)
            validate(**output)
            energies = output["energies"].real

            trafo = output["trafo"]
            matter_idxs_plus = [2, 6]
            matter_idxs_minus = [3, 7]
            polaritons = jnp.arange(8)
            content_plus = jax.vmap(lambda p: get_content(trafo, matter_idxs_plus, p))(polaritons)
            content_minus = jax.vmap(lambda p: get_content(trafo, matter_idxs_minus, p))(polaritons)

            ax = axs
            idxs = [1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            for idx in idxs:
                ann = delta[idx]
                line = add_segment(ax, fractions, energies[:, idx], ann, mi=mi, mx=mx)

            ax.set_xlabel(r'$\sqrt{N_- / N_+}$')
            if i == 0:
                ax.set_ylabel(r'$\omega / \omega_b$')

            fig.colorbar(line, ax=ax, label=r"$\Delta$")

        plt.tight_layout()
        plt.savefig("energy_fraction.pdf")

def plot_fraction_g_energy():
    """3D surface plot for energies[i] colored by matter fraction vs fractions and gs"""

    fractions = jnp.linspace(0, 1, 60)
    g_values = jnp.linspace(0, 1, 80)
    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    gamma = 0.1
    
    get_kernel_vmapped = jax.vmap(jax.vmap(
        lambda f, g:
        get_kernel(omega_plus,
                   omega_minus,
                   omega_b,
                   g=g,
                   gamma=gamma,
                   anti_res=True,
                   fraction_minus=f,
                   damping=0),
        in_axes=(0, None),
        out_axes=0),
                                  in_axes = (None, 0),
                                  out_axes = 1                                  
                                  )

    kernels = get_kernel_vmapped(fractions, g_values)
    output = get_bogoliubov_vmapped(get_matrix_stack(kernels))

    energies = output["energies"].reshape(fractions.size, g_values.size, 8)
    
    matter_idxs_plus = [2, 6]    
    get_content_plus = jax.vmap(jax.vmap(
        lambda p, t:
        get_content(t,
                    matter_idxs_plus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                in_axes = (None, 0),
                                out_axes = 1)    
    matter_idxs_minus = [3, 7]    
    get_content_minus = jax.vmap(jax.vmap(
        lambda p, t :
        get_content(t,
                    matter_idxs_minus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                 in_axes = (None, 0),
                                 out_axes = 1)

    polaritons = jnp.arange(8)
    cp = get_content_plus(polaritons, output["trafo"])
    cm = get_content_minus(polaritons, output["trafo"])
    delta_raw = cp - cm
    delta = delta_raw.reshape(8, fractions.size, g_values.size)

    X, Y = jnp.meshgrid(g_values, fractions)
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 10,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        idxs = [1, 3]        
        
        for i in idxs:
            Z = energies[..., i]
            colors = delta[i]

            surf = ax.plot_surface(X, Y, Z,
                                   facecolors=plt.cm.plasma((colors - colors.min()) / (colors.max() - colors.min())),
                                   linewidth=0.2,
                                   antialiased=True,
                                   alpha = 0.9,
                                   shade=False)

        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(delta[jnp.array(idxs)])
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=-0.05, aspect=20, label=r'$\Delta$')

        # ax.minorticks_on()
        ax.set_facecolor('white')

        # Remove background grid for a cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.view_init(elev=25, azim=80)

        ax.set_xlabel(r'$g$')

        ax.yaxis.set_rotate_label(False)        
        ax.set_ylabel(r'$\sqrt{N_- / N_+}$', rotation = 0, labelpad = 35)    

        ax.zaxis.set_rotate_label(False)  
        ax.set_zlabel(r'$\omega / \omega_b$', rotation = 0, labelpad = 35)
        ax.tick_params(axis='z', which='major', pad=15)

        plt.tight_layout()
        plt.savefig("energy_fraction_g.pdf")
        

def plot_damping_energies():
    """plots (energy, fraction negative) annotated with polaritonic + matter fraction
    for mildly chiral molecule in perfect cavity for strong coupling, looping over different g"""

    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    g_values = [1e-2]  # Looping over these g values
    gamma = 0.1
    dampings = jnp.linspace(0, 1, 100)
    fraction = 1

    # Define custom settings for plots
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 10,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig, axs = plt.subplots(1, 1)

        for i, g in enumerate(g_values):
            get_kernel_vmapped = jax.vmap(
                lambda d: get_kernel(omega_plus,
                                     omega_minus,
                                     omega_b,
                                     g=g,
                                     gamma=gamma,
                                     anti_res=True,
                                     fraction_minus=fraction,
                                     damping=d),
                in_axes=0, out_axes=0)

            kernels = get_kernel_vmapped(dampings)
            output = get_bogoliubov_vmapped(kernels)
            validate(**output, eps = 1e-2)
            energies = output["energies"].real

            trafo = output["trafo"]
            matter_idxs_plus = [2, 6]
            matter_idxs_minus = [3, 7]
            polaritons = jnp.arange(8)
            content_plus = jax.vmap(lambda p: get_content(trafo, matter_idxs_plus, p))(polaritons)
            content_minus = jax.vmap(lambda p: get_content(trafo, matter_idxs_minus, p))(polaritons)

            ax = axs
            idxs = [0, 1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            for idx in idxs:
                ann = delta[idx]
                line = add_segment(ax, dampings, energies[:, idx], ann, mi=mi, mx=mx)

            ax.set_xlabel(r'$d$')
            if i == 0:
                ax.set_ylabel(r'$\omega / \omega_b$')

            fig.colorbar(line, ax=ax, label=r"$\Delta$")

        plt.tight_layout()
        plt.savefig("energy_damping.pdf")

    
def plot_energies_damping_fraction():
    """grid plot: energies for different combinations of damping and fraction"""
    
    fractions = jnp.linspace(0, 1, 60)
    dampings = jnp.linspace(0, 1, 80)
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    gamma = 0.1
    g = 1e-2
    
    get_kernel_vmapped = jax.vmap(jax.vmap(
        lambda f, d:
        get_kernel(omega_plus,
                   omega_minus,
                   omega_b,
                   g=g,
                   gamma=gamma,
                   anti_res=True,
                   fraction_minus=f,
                   damping=d),
        in_axes=(0, None),
        out_axes=0),
                                  in_axes = (None, 0),
                                  out_axes = 1                                  
                                  )

    kernels = get_kernel_vmapped(fractions, dampings)
    output = get_bogoliubov_vmapped(get_matrix_stack(kernels))

    energies = output["energies"].reshape(fractions.size, dampings.size, 8)
    
    matter_idxs_plus = [2, 6]    
    get_content_plus = jax.vmap(jax.vmap(
        lambda p, t:
        get_content(t,
                    matter_idxs_plus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                in_axes = (None, 0),
                                out_axes = 1)    
    matter_idxs_minus = [3, 7]    
    get_content_minus = jax.vmap(jax.vmap(
        lambda p, t :
        get_content(t,
                    matter_idxs_minus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                 in_axes = (None, 0),
                                 out_axes = 1)

    polaritons = jnp.arange(8)
    cp = get_content_plus(polaritons, output["trafo"])
    cm = get_content_minus(polaritons, output["trafo"])
    delta_raw = cp - cm
    delta = delta_raw.reshape(8, fractions.size, dampings.size)

    X, Y = jnp.meshgrid(dampings, fractions)
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 10,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        idxs = [0, 3]        
        
        for i in idxs:
            Z = energies[..., i]
            colors = delta[i]

            surf = ax.plot_surface(X, Y, Z,
                                   facecolors=plt.cm.plasma((colors - colors.min()) / (colors.max() - colors.min())),
                                   linewidth=0.2,
                                   antialiased=True,
                                   alpha = 0.9,
                                   shade=False)

        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(delta[jnp.array(idxs)])
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=-0.05, aspect=20, label=r'$\Delta$')

        # ax.minorticks_on()
        ax.set_facecolor('white')

        # Remove background grid for a cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.view_init(elev=25, azim=80)

        ax.set_xlabel(r'$d$')

        ax.yaxis.set_rotate_label(False)        
        ax.set_ylabel(r'$\sqrt{N_- / N_+}$', rotation = 0, labelpad = 35)    

        ax.zaxis.set_rotate_label(False)  
        ax.set_zlabel(r'$\omega / \omega_b$', rotation = 0, labelpad = 35)
        ax.tick_params(axis='z', which='major', pad=15)

        plt.tight_layout()
        plt.savefig("energy_damping_fraction.pdf")

def plot_fraction_damping_energy():
    """3D surface plot for energies[i] colored by matter fraction vs fractions and damping
    """

    fractions = jnp.linspace(0, 1, 20)    
    damping = jnp.linspace(0, 1, 40)
    g = 1e-2
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    gamma = 0.1
    
    get_kernel_vmapped = jax.vmap(jax.vmap(
        lambda f, d:
        get_kernel(omega_plus,
                   omega_minus,
                   omega_b,
                   g=g,
                   gamma=gamma,
                   anti_res=True,
                   fraction_minus=f,
                   damping=d),
        in_axes=(0, None),
        out_axes=0),
                                  in_axes = (None, 0),
                                  out_axes = 1                                  
                                  )

    kernels = get_kernel_vmapped(fractions, damping)
    output = get_bogoliubov_vmapped(get_matrix_stack(kernels))

    energies = output["energies"].reshape(fractions.size, damping.size, 8)
    
    matter_idxs_plus = [2, 6]    
    get_content_plus = jax.vmap(jax.vmap(
        lambda p, t:
        get_content(t,
                    matter_idxs_plus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                in_axes = (None, 0),
                                out_axes = 1)    
    matter_idxs_minus = [3, 7]    
    get_content_minus = jax.vmap(jax.vmap(
        lambda p, t :
        get_content(t,
                    matter_idxs_minus,
                    p),
        in_axes = (0, None),
        out_axes = 0),
                                 in_axes = (None, 0),
                                 out_axes = 1)

    polaritons = jnp.arange(8)
    cp = get_content_plus(polaritons, output["trafo"])
    cm = get_content_minus(polaritons, output["trafo"])
    delta_raw = cp - cm
    delta = delta_raw.reshape(8, fractions.size, damping.size)

    X, Y = jnp.meshgrid(damping, fractions)
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 10,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in [0, 1, 2, 3]:
            Z = energies[..., i]
            colors = delta[i]

            surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma((colors - colors.min()) / (colors.max() - colors.min())),
                                   linewidth=0.2,
                                   antialiased=True,
                                   alpha = 0.5,
                                   shade=False)

        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(delta)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=-0.05, aspect=20, label=r'$\Delta$')

        # ax.minorticks_on()
        ax.set_facecolor('white')

        # Remove background grid for a cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.view_init(elev=25, azim=80)

        ax.set_xlabel(r'$d$')

        ax.yaxis.set_rotate_label(False)        
        ax.set_ylabel(r'$\sqrt{N_- / N_+}$', rotation = 0, labelpad = 35)    

        ax.zaxis.set_rotate_label(False)  
        ax.set_zlabel(r'$\omega / \omega_b$', rotation = 0, labelpad = 35)
        ax.tick_params(axis='z', which='major', pad=15)

        plt.tight_layout()
        plt.savefig("energy_fraction_damping.pdf")

    
def plot_occupation_coupling():
    """plots asymptotic occupation for racemic mixture"""    
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    g = 1e-2
    gamma = 0.5    
    fraction = 1
    damping = 1
    
    # c_- / c_+
    coupling_gamma = 1
    coupling_ratios = jnp.linspace(0, 1, 100)
    one = jnp.ones_like(coupling_ratios)
    zero = 0 * one
    coupling = coupling_gamma * jnp.stack([one, coupling_ratios, zero, zero])

    kernel = get_kernel(omega_plus,
                        omega_minus,
                        omega_b,
                        g,
                        gamma = gamma,
                        fraction_minus = fraction,
                        anti_res = True,
                        damping = damping)    
    output = get_bogoliubov(kernel)

    # validate(**output)
    get_occ_vmapped = jax.vmap(lambda c : get_asymptotic_occupation(output, coupling = c), in_axes = 1, out_axes = 0)    
    occ = get_occ_vmapped(coupling)
    print(jnp.abs(occ.imag).max(), jnp.sum(occ < 0))

    # normalized energy
    occ = occ.real / occ.real.sum(axis = 1)[:, None]
    
    occ_plus = occ[:, 2]
    occ_minus = occ[:, 3]
    
    fig, axs = plt.subplots(1, 1)

    ax = axs
    # ax.plot(coupling_ratios, occ)
    ax.plot(coupling_ratios, occ_plus, label = r'$\langle n_+ \rangle$')
    ax.plot(coupling_ratios, occ_minus, '--', label = r'$\langle n_- \rangle$')
    ax.plot(coupling_ratios, occ_plus - occ_minus, '-.')
    ax.set_xlabel(r'$c_- / c_+$')
    ax.set_ylabel(r'$\Delta E / \Delta E_t$')    
    plt.legend()
    # plt.show()
    plt.savefig(f"occupation_coupling_{gamma}.pdf")


def plot_occupation_fraction_coupling():
    # Define ranges for g and gamma
    g_values = [1e-3, 1e-2, 1e-1]  # Example values for g
    gamma_values = [1e-2, 1e-1]  # Example values for gamma

    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    
    fractions = jnp.linspace(0, 1, 100)
    
    # c_- / c_+
    coupling_gamma = 1
    coupling_ratios = jnp.linspace(0, 1, 40)
    one = jnp.ones_like(coupling_ratios)
    zero = 0 * one
    coupling = coupling_gamma * jnp.stack([one, coupling_ratios, zero, zero]).T

    # Set up figure
    fig, axes = plt.subplots(len(g_values), len(gamma_values), figsize=(15, 10), sharex=True, sharey=True)

    # Iterate over g and gamma values to generate panels
    for i, g in enumerate(g_values):
        for j, gamma in enumerate(gamma_values):
            # Compute kernels for the given g and gamma
            get_kernel_vmapped =  jax.vmap(
                lambda x:
                get_kernel(omega_plus,
                           omega_minus,
                           omega_b,
                           g,
                           gamma=gamma,
                           anti_res = True,
                           fraction_minus=x,
                        damping=1),
                in_axes = 0,
                out_axes = 0)
            
            kernels = get_kernel_vmapped(fractions)
            output = get_bogoliubov_vmapped(kernels)
            f_tmp = jax.vmap( get_asymptotic_occupation, in_axes=({'energies': 0, 'kernel': 0, "trafo" : 0, "inverse" : 0, "energies_raw" : 0}, None), out_axes = 0)
            get_asymptotic_occupation_vmapped = jax.vmap(f_tmp, (None, 0), 1)

            # fractions x coupling
            occ =  get_asymptotic_occupation_vmapped(output, coupling)

            # sanity check print
            print(jnp.abs(occ.imag).max(), jnp.sum(occ < 0))
            
            # normalized transfer difference
            occ = occ.real / occ.real.sum(axis = -1)[..., None]
            delta_occ = occ[..., 2] - occ[..., 3]
            
            # Plot heatmap in the appropriate subplot
            ax = axes[i, j]
            
            # rows and columns of image
            im = ax.imshow(delta_occ.T, 
                           aspect='auto', 
                           cmap="coolwarm", 
                           origin='lower',
                           extent=[fractions.min(), fractions.max(), coupling.min(), coupling.max()])

            # Set labels and titles
            if i == len(g_values) - 1:
                ax.set_xlabel(r'$\frac{N_-}{N_+}$')
            if j == 0:
                ax.set_ylabel(r'$c_- / c_+$')                
            ax.set_title(f"g={g}, gamma={gamma}")

            # Add colorbar to each subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            # cbar.set_label(r"C_+ - C_-", fontsize=20, labelpad=-85)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
def plot_occupation_gamma_coupling():
    g_values = [1e-3, 1e-2, 1e-1] 
    f_values = [0.1, 1]  

    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    
    gammas = jnp.linspace(0, 2, 100)
    
    # c_- / c_+
    coupling_gamma = 1
    coupling_ratios = jnp.linspace(0, 1, 40)
    one = jnp.ones_like(coupling_ratios)
    zero = 0 * one
    coupling = coupling_gamma * jnp.stack([one, coupling_ratios, zero, zero]).T

    # Set up figure
    fig, axes = plt.subplots(len(g_values), len(f_values), figsize=(15, 10), sharex=True, sharey=True)

    # Iterate over g and gamma values to generate panels
    for i, g in enumerate(g_values):
        for j, f in enumerate(f_values):
            # Compute kernels for the given g and gamma
            get_kernel_vmapped =  jax.vmap(
                lambda x:
                get_kernel(omega_plus,
                           omega_minus,
                           omega_b,
                           g,
                           gamma=x,
                           anti_res = True,
                           fraction_minus=0,
                           damping=0),
                in_axes = 0,
                out_axes = 0)
            
            kernels = get_kernel_vmapped(gammas)
            output = get_bogoliubov_vmapped(kernels)
            f_tmp = jax.vmap( get_asymptotic_occupation, in_axes=({'energies': 0, 'kernel': 0, "trafo" : 0, "inverse" : 0, "energies_raw" : 0}, None), out_axes = 0)
            get_asymptotic_occupation_vmapped = jax.vmap(f_tmp, (None, 0), 1)

            # fractions x coupling
            occ =  get_asymptotic_occupation_vmapped(output, coupling)

            # sanity check print
            print(jnp.abs(occ.imag).max(), jnp.sum(occ < 0))
            
            # normalized transfer difference
            occ = occ.real / occ.real.sum(axis = -1)[..., None]
            delta_occ = occ[..., 2] - occ[..., 3]
            delta_occ = occ[..., 2]
            
            # Plot heatmap in the appropriate subplot
            ax = axes[i, j]            
            ax.set_title(fr"g={g}, N_+ / N_- ={f}")            
            ax.plot(gammas, occ[:, 0, :])
            # print(occ[:, 0, :].sum(axis = -1))
            # ax.plot(gammas, occ[:, 0, 2])
        
            # # rows and columns of image
            # im = ax.imshow(delta_occ.T, 
            #                aspect='auto', 
            #                cmap="coolwarm", 
            #                origin='lower',
            #                extent=[gammas.min(), gammas.max(), coupling.min(), coupling.max()])

            # # Set labels and titles
            # if i == len(g_values) - 1:
            #     ax.set_xlabel(r'$\gamma$')
            # if j == 0:
            #     ax.set_ylabel(r'$c_- / c_+$')                
            # ax.set_title(fr"g={g}, N_+ / N_- ={f}")

            # # Add colorbar to each subplot
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cbar = plt.colorbar(im, cax=cax)
            # # cbar.set_label(r"C_+ - C_-", fontsize=20, labelpad=-85)
            
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # matter content
    
    # plot_fraction_energies() # DONE
    plot_fraction_g_energy() # DONE
    # plot_damping_energies() # DONE
    # plot_energies_damping_fraction() # DONE

    # s matrix
    
    # plot_occupation_coupling()  # TODO pub-hübsch
    # plot_occupation_fraction_coupling()  # TODO pub-hübsch
    # plot_occupation_gamma_coupling()  # TODO pub-hübsch

    # TODO: appendix gs energy differencex
    # plot_gamma_energies() # TODO pub-hübsch

