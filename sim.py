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
    omega_plus = 1.2
    omega_minus = 1.2
    omega_b = 1
    g = 1e-2
    gammas = jnp.logspace(-2, 0.2, 100)
    # gammas = jnp.linspace(0.01, 1, 100)

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

    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):    
        fig, ax = plt.subplots(1, 1)

        mi, mx = content_plus.min(), content_plus.max()

        idx = 0
        line = add_segment(ax, gammas, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
        # line.set_label('Lower Polariton')
        # line.set_linestyle('--')

        idx = 3
        line = add_segment(ax, gammas, energies[:, idx], content_plus[idx], mi = mi, mx = mx)        
        # line.set_label('Upper Polariton')

        ax.set_xlabel(r'$\gamma_+ / \omega_b$')
        ax.set_ylabel(r'$\omega / \omega_b$')

        fig.colorbar(line, ax=ax, label=r"$C_+$")

        # plt.legend()
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig("energy_gamma.pdf")

    
def plot_fraction_energies():
    """plots (energy, fraction negative) annotated with polaritonic + matter fraction
    for mildly chiral molecule in perfect cavity for strong coupling, looping over different g"""

    omega_plus = 1
    omega_minus = 0
    omega_b = 1
    g_values = [1e-2]  # Looping over these g values
    gamma = 0.1
    fractions = jnp.linspace(0, 1, 20)

    # Define custom settings for plots
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig, axs = plt.subplots(1, 2, figsize = (10,5))

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

            ax = axs[0]
            idxs = [1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            styles = [':', '-', '--']
            for idx in idxs:
                ann = delta[idx]
                line = add_segment(ax, fractions, energies[:, idx], ann, mi=mi, mx=mx)
                line.set_linestyle(styles[idx-1])


            ax.set_xlabel(r'$\gamma_- / \gamma_+$')
            if i == 0:
                ax.set_ylabel(r'$\omega / \omega_b$')

            divider = make_axes_locatable(axs[0])  # replace axs[0] with the subplot axis you want
            cax = divider.append_axes("top", size="5%", pad=0.3)
            cbar = fig.colorbar(line, cax=cax, orientation='horizontal', label=r"$\Delta$")
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')

            ax.annotate(
                "(a)", xy=(-0.3, 1.3), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )

            ax = axs[1]
            idxs = [1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            styles = [':', '-', '--']
            colors = plt.get_cmap("tab10").colors  # nice set of distinct colors
            for idx in idxs:
                ann = delta[idx]
                ax.plot(fractions, jnp.abs(ann), ls = styles[idx-1], color = colors[idx+1])

            ax.set_xlabel(r'$\gamma_- / \gamma_+$')
            ax.set_ylabel(r'$\vert \Delta \vert$')
            ax.annotate(
                "(b)", xy=(-0.3, 1.12), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )
            ax.set_yscale('log')

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
        "lines.linewidth" : 2,
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
        ax.set_ylabel(r'$\gamma_- / \gamma_+$', rotation = 0, labelpad = 35)    

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
    dampings = jnp.linspace(0.01, 1, 100)
    fraction = 1

    # Define custom settings for plots
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig, axs = plt.subplots(1, 2, figsize = (10,5))

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

            ax = axs[0]
            idxs = [0, 1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            for idx in idxs:
                ann = delta[idx]
                line = add_segment(ax, dampings, energies[:, idx], ann, mi=mi, mx=mx)
                line.set_label(idxs[idx] + 1)


            ax.set_xlabel(r'$d$')
            if i == 0:
                ax.set_ylabel(r'$\omega / \omega_b$')
                
            divider = make_axes_locatable(axs[0])  # replace axs[0] with the subplot axis you want
            cax = divider.append_axes("top", size="5%", pad=0.3)
            cbar = fig.colorbar(line, cax=cax, orientation='horizontal', label=r"$\Delta$")
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')

            ax.annotate(
                "(a)", xy=(-0.3, 1.3), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )

            
            ax = axs[1]
            idxs = [0, 1, 2, 3]
            delta = content_plus - content_minus
            mi, mx = delta.min(), delta.max()
            colors = plt.get_cmap("tab10").colors  # nice set of distinct colors
            for idx in idxs:
                ann = delta[idx]
                ax.plot(dampings, jnp.abs(ann), color = colors[idx], label = f"{idx + 1}")
            ax.legend()

            ax.set_xlabel(r'$d$')
            ax.set_ylabel(r'$\vert \Delta \vert$')
            ax.annotate(
                "(b)", xy=(-0.3, 1.12), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )
            ax.set_yscale('log')


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
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        idxs = [0, 3]        
        
        for i in idxs:
            Z = energies[..., i]
            colors = delta[i]
            print(colors.max(), jnp.abs(colors).min())

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
        ax.set_ylabel(r'$\gamma_- / \gamma_+$', rotation = 0, labelpad = 35)    

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
        "lines.linewidth" : 2,
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

        
def plot_gamma_transfer():
    """plots energy transfer occupation for efficiency single enantiomer in perfectly chiral cavity"""    
    omega_plus = 1.2
    omega_minus = 1.2
    omega_b = 1
    g = 1e-2
    gammas = jnp.logspace(-2, 0.2, 100)
    fraction = 0
    damping = 0

    
    get_kernel_vmapped = lambda g : jax.vmap(
        lambda gamma : get_kernel(omega_plus,
                                  omega_minus,
                                  omega_b,
                                  g,
                                  fraction_minus = fraction,
                                  gamma=gamma,
                                  anti_res = True,                            
                                  damping=0),
        in_axes = 0, out_axes = 0)

    kernels = get_kernel_vmapped(g)(gammas)
    output = get_bogoliubov_vmapped(kernels)
    
    get_asymptotic_occupation_vmapped = jax.vmap( get_asymptotic_occupation, in_axes=({'energies': 0, 'kernel': 0, "trafo" : 0, "inverse" : 0, "energies_raw" : 0}, None), out_axes = 0)
    occ = get_asymptotic_occupation_vmapped(output, jnp.array([1, 0, 0, 0]))
    print(jnp.abs(occ.imag).max(), jnp.sum(occ < 0))
    
    # to energy => multiply by frequency
    occ *= jnp.array([omega_plus, omega_minus, omega_b, omega_b])
    
    # normalized energy
    occ = occ.real / occ.real.sum(axis = 1)[:, None]
    
    occ_plus = occ[:, 2]
    occ_minus = occ[:, 3]

    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):    
        fig, ax = plt.subplots(1, 1)
            
        # ax.plot(coupling_ratios, occ)
        ax.plot(gammas, occ_plus)
        ax.set_xlabel(r'$\gamma_+ / \omega_b$')
        ax.set_ylabel(r'$\eta_+$')
            
        plt.tight_layout()
        plt.xscale('log')
        plt.savefig(f"transfer_gamma.pdf")
    

def plot_fraction_coupling_transfer():
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    damping = 1
    gamma = 0.5
    g = 1e-2

    fraction = jnp.linspace(0., 1, 201)

    coupling_gamma = 1
    coupling_ratios = jnp.linspace(0, 1, 200)
    one = jnp.ones_like(coupling_ratios)
    zero = 0 * one
    coupling = coupling_gamma * jnp.stack([one, coupling_ratios, zero, zero]).T

    get_kernel_vmapped = jax.vmap(
        lambda f: get_kernel(omega_plus, omega_minus, omega_b, g=g, gamma=gamma,
                             anti_res=True, fraction_minus=f, damping=damping),
        in_axes=0, out_axes=0
    )

    kernels = get_kernel_vmapped(fraction)
    output = get_bogoliubov_vmapped(kernels)

    f_tmp = jax.vmap(get_asymptotic_occupation,
                     in_axes=({'energies': 0, 'kernel': 0, "trafo": 0,
                               "inverse": 0, "energies_raw": 0}, None),
                     out_axes=0)

    get_asymptotic_occupation_vmapped = jax.vmap(f_tmp, (None, 0), 1)

    occ = get_asymptotic_occupation_vmapped(output, coupling)
    occ *= jnp.array([omega_plus, omega_minus, omega_b, omega_b])

    # Normalizing data for plotting
    occ_norm = occ.real / occ.real.sum(axis=-1)[..., None]
    colorplot_data = occ_norm[..., 2] - occ_norm[..., 3]

    # --- Plotting ---
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(custom_params):
        fig, (ax_heatmap, ax_lines) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.2]})

        # Left plot: heatmap
        im = ax_heatmap.imshow(colorplot_data.T,
                               aspect='auto',
                               cmap="magma",
                               origin='lower',
                               interpolation='sinc',
                               extent=[fraction.min(), fraction.max(), coupling_ratios.min(), coupling_ratios.max()])

        divider = make_axes_locatable(ax_heatmap)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(r"$\eta_+ - \eta_-$")

        ax_heatmap.set_xlabel(r"$\gamma_- / \gamma_+$")
        ax_heatmap.set_ylabel(r"$c_- / c_+$")

        # Right plot: line plots
        line_indices = [10, 50, 100, 150, 201]  # example columns from coupling_ratios
        colorplot_data = colorplot_data.T
        colorplot_data /= jnp.max(colorplot_data, axis = 0)
        for idx in line_indices:
            ax_lines.plot(coupling_ratios, colorplot_data[:, idx], label=rf"$\gamma_- / \gamma_+ ={fraction[idx]:.2f}$")

        ax_lines.set_xlabel(r"$c_- / c_+$")
        ax_lines.set_ylabel(r"$\eta_+ - \eta_-$ (normalized)")
        ax_lines.legend()
        ax_lines.grid(True)

        plt.tight_layout()
        plt.savefig("fraction_coupling_transfer.pdf")

def plot_detuning_transfer():
    """plots energy transfer occupation for efficiency single enantiomer in perfectly chiral cavity"""    
    omega_plus = 1
    omega_minus = 1
    omega_b = 1
    g = 1e-2
    gamma = 0.5
    detuning = jnp.linspace(0, 10, 100)
    fraction = 1
    damping = 1

    coupling_ratios = [0, 0.1, 0.5]


    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 2,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):    
        fig, ax = plt.subplots(1, 1)

        for cr in coupling_ratios:
            get_kernel_vmapped = lambda g : jax.vmap(
                lambda d : get_kernel(omega_plus,
                                      omega_minus + d,
                                      omega_b,
                                      g,
                                      gamma=gamma,
                                      fraction_minus = 1,
                                      anti_res = True,                            
                                      damping=1),
                in_axes = 0, out_axes = 0)

            kernels = get_kernel_vmapped(g)(detuning)
            output = get_bogoliubov_vmapped(kernels)

            get_asymptotic_occupation_vmapped = jax.vmap( get_asymptotic_occupation, in_axes=({'energies': 0, 'kernel': 0, "trafo" : 0, "inverse" : 0, "energies_raw" : 0}, None), out_axes = 0)
            occ = get_asymptotic_occupation_vmapped(output, jnp.array([1, cr, 0, 0]))
            print(jnp.abs(occ.imag).max(), jnp.sum(occ < 0))

            # to energy => multiply by frequency
            occ *= jnp.array([omega_plus, omega_minus, omega_b, omega_b])

            # normalized energy
            occ = occ.real / occ.real.sum(axis = 1)[:, None]

            occ_plus = occ[:, 2]
            occ_minus = occ[:, 3]
            occ = occ_plus - occ_minus

            # ax.plot(coupling_ratios, occ)
            ax.plot(detuning, occ, label = rf'$c_- / c_+ =$ {cr}')
            ax.set_xlabel(r'$\Delta \omega / \omega_b$')
            ax.set_ylabel(r'$\eta_+ - \eta_-$')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"transfer_detuning.pdf")


def plot_gamma_ground_state():
    # J. Phys. Chem. Lett. 2023, 14, 3777−3784
    # shows difference between polaritonic ground and excited state energies for single-mode chiral cavity with singly chiral molecules inside when number of molecules is increased

    omega_c = 1.
    omega = 1.
    g = 1e-2

    # ~ sqrt(number) of molecules
    gammas = jnp.linspace(0.1, 10, 100)
    # linear scaling of gs discrimination for small number of molecules jnp.logspace(0, 1, 100)

    def get_energy_deltas(s):
        # corresponds to different enantiomers placed separately into perfectly chiral cavity at resonance

        # "reproduce" by zeroing out one mode, setting negative coupling to this mode for other chirality    
        k_plus = get_kernel(omega_c, 0, omega, g = g, gamma = s, diamagnetic = True, anti_res = True, damping = 0)
        energies_plus = get_bogoliubov(k_plus)["energies"][:4]

        k_minus = get_kernel(omega_c, 0, omega, g = -g, gamma = s, diamagnetic = True, anti_res = True, damping = 0)
        energies_minus = get_bogoliubov(k_minus)["energies"][:4]

        # energies come in pairs, one of them ios decoupled => 0
        return energies_plus, energies_minus

    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth" : 4,
        "pdf.fonttype": 42
    }

    with mpl.rc_context(rc=custom_params):    
        fig, ax = plt.subplots(1, 1)
    
        e_p, e_m = [], []
        for s in gammas:
            ep, em = get_energy_deltas(s)
            e_p.append(ep)
            e_m.append(em)

        e_p = jnp.array(e_p).real
        e_m = jnp.array(e_m).real

        # ax.plot(gammas**2, (e_p - e_m)[:, [0, -1]] )
        fac = 0.02
        delta_vac = e_p[:, 0] + e_p[:, -1] - (e_m[:, 0] + e_m[:, -1])
        ax.plot(gammas**2, delta_vac.real / 2, label = 'analytic')
        sqrt_term = gammas
        ax.plot(gammas**2, fac * sqrt_term, '--', label = r'$0.02 \cdot \gamma \omega_b$')
        
        ax.set_xlabel(r'$(\gamma / \omega_b)^2$')
        ax.set_ylabel(r'$\delta E / \omega_b$')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("sqrt_scaling.pdf")


        
if __name__ == '__main__':
    # matter content    
    # plot_gamma_energies() # DONE
    # plot_fraction_energies() # DONE
    # plot_fraction_g_energy() # DONE
    # plot_damping_energies() # DONE
    # plot_energies_damping_fraction() # DONE

    # s matrix    
    # plot_gamma_transfer()  # DONE
    plot_fraction_coupling_transfer() # TODO
    # plot_detuning_transfer() # DONE

    # appendix 
    # plot_gamma_ground_state() # DONE
