import numpy as np
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt

# TODO: bogoliubov is kaputt
# TODO: racemic (=1:1) case with both enantiomers present
# TODO: wtf am I seeing in relative deposited power?
# TODO: clarify what boundary conditions the semiclassical approximation corresponds to, i.e. we have sth like H_int = \int dw W_ij(w) a_i c_j(w) + h.c. and take c classical, i.e. our interaction term should be sth like F.T. <W_ij(w) c_j(w)> => W_ij(t) * <c_j(t)>, which makes only sense if we assume non-dispersive couplings, but we can cope with this analogous to [PhysRevA.74.033811.pdf]
# TODO: how to relate A coupling to dipole coupling? [done in carsten paper] => some realistic values by combining with [ACS]
# DONE : why exactly omega_a != omega_b is there some singularity? Probably not, this is likely just a resonance effect (cavity likes matter => both hybridize already at low frequencies)
# DONE is there bug in bogoliubov when we only vary first arg? No, this is just some ordering issue I will not bother to solve. plotting all modes alleviates this problem.
# DONE: n^{1/2} scaling of coupling also from PRA

# chosen vals from paper: w_b = w_a, freq-independent rates g_a = 0.04 w_a, g_b = 0.08 w_a
# plot abs(w) for varying values of g_+ / g_-, c_+ / c_-
def test_pra():
    Omegas = np.linspace(0.01, 10, 100)
    dummy = []
    omega_c = 1/1.7
    omega = 1
    
    for Omega in Omegas:
        
        M = np.zeros((4,4))

        M[0,0] = omega
        M[0,1] = Omega
        M[0,2] = 0
        M[0,3] = Omega

        M[1,0] = Omega
        M[1,1] = omega_c + 2*Omega**2
        M[1,2] = Omega
        M[1,3] = 2*Omega**2

        M[2,0] = 0
        M[2,1] = -Omega
        M[2,2] = -omega
        M[2,3] = -Omega

        M[3,0] = -Omega
        M[3,1] = -2*Omega**2
        M[3,2] = -Omega
        M[3,3] = -omega_c - 2*Omega**2

        x = bogoliubov(M)        
        content  = (np.abs(x[0, 0])**2 + np.abs(x[2, 0])**2) / np.linalg.norm(x[:, 0])**2
        dummy.append(content)
        
    plt.plot(Omegas, dummy)
    plt.xscale('log')            
    plt.legend()        
    plt.savefig("pra_matter_content.pdf")
    plt.close()

def generate_kernel_old(omega_a, omega_b, g_plus, g_minus):
    """
    Constructs the kernel. The vector ordering is $\begin{pmatrix} a a^{\dagger} \end{pmatrix}$, where $a = \begin{pmatrix} a_+ a_- b \end{pmatrix}$.

    The coupling parameters are light/matter eigenfrequencies and "paramagnetic" coupling constants. The corresponding diamagnetic couplings are computed following PhysRevA.74.033811.

    Parameters:
        omega_a: Degenerate eigenfrequency omega_a of cavity modes
        omega_b: Eigenfrequency of isolated bosonic matter excitation
        g_plus: Coupling between positively helical light and matter
        g_minus: Coupling between negatively helical light and matter

    Returns:
        A 6x6 numpy matrix.
    """
    Omega = g_plus * g_minus / omega_b
    Omega_plus = g_plus**2 / omega_b
    Omega_minus = g_minus**2 / omega_b

    # Matrix components
    M = np.zeros((6, 6), dtype=complex)

    # First row
    M[0, 0] = omega_a + 2 * Omega_plus
    M[0, 1] = 2 * Omega
    M[0, 2] = g_plus
    M[0, 3] = 2 * Omega_plus
    M[0, 4] = 2 * Omega
    M[0, 5] = g_plus

    # Second row
    M[1, 0] = 2 * Omega
    M[1, 1] = omega_a + 2 * Omega_minus
    M[1, 2] = g_minus
    M[1, 3] = 2 * Omega
    M[1, 4] = 2 * Omega_minus
    M[1, 5] = g_minus

    # Third row
    M[2, 0] = g_plus
    M[2, 1] = g_minus
    M[2, 2] = omega_b
    M[2, 3] = g_plus
    M[2, 4] = g_minus
    M[2, 5] = 0

    # Fourth row
    M[3, 0] = -2 * Omega_plus
    M[3, 1] = -2 * Omega
    M[3, 2] = -g_plus
    M[3, 3] = -omega_a - 2 * Omega_plus
    M[3, 4] = -2 * Omega
    M[3, 5] = -g_plus

    # Fifth row
    M[4, 0] = -2 * Omega
    M[4, 1] = -2 * Omega_minus
    M[4, 2] = -g_minus
    M[4, 3] = -2 * Omega
    M[4, 4] = -omega_a - 2 * Omega_minus
    M[4, 5] = -g_minus

    # Sixth row
    M[5, 0] = -g_plus
    M[5, 1] = -g_minus
    M[5, 2] = 0
    M[5, 3] = -g_plus
    M[5, 4] = -g_minus
    M[5, 5] = -omega_b

    return M

        
def generate_kernel(omega_a, omega_b, g_plus, g_minus, n_plus = 1, n_minus = 0, diamagnetic = True):
    """
    Constructs the kernel. The vector ordering is $\begin{pmatrix} a a^{\dagger} \end{pmatrix}$, where $a = \begin{pmatrix} a_+ a_- b \end{pmatrix}$.

    The coupling parameters are light/matter eigenfrequencies and "paramagnetic" coupling constants. The corresponding diamagnetic couplings are computed following PhysRevA.74.033811.

    Parameters:
        omega_a: Degenerate eigenfrequency omega_a of cavity modes
        omega_b: Eigenfrequency of isolated bosonic matter excitation
        g_plus: Coupling between positively helical light and matter
        g_minus: Coupling between negatively helical light and matter
        n_plus : scaling factor of + enantiomers
        n_minus : scaling factor of - enantiomers

    Returns:
        An 8x8 numpy matrix.
    """
    Omega = g_plus * g_minus / omega_b * (diamagnetic == True)
    Omega_plus = g_plus**2 / omega_b * (diamagnetic == True)
    Omega_minus = g_minus**2 / omega_b * (diamagnetic == True)

    # Matrix components
    M = np.zeros((8, 8), dtype=complex)

    # rows  = (a+, a-, b+, b-)
    
    ## creators    
    
    # First row (a_+)
    M[0, 0] = omega_a + 2 * Omega_plus
    M[0, 1] = 2 * Omega
    M[0, 2] = g_plus * n_plus
    M[0, 3] = g_minus * n_minus
    M[0, 4] = 2 * Omega_plus
    M[0, 5] = 2 * Omega
    M[0, 6] = g_plus * n_plus
    M[0, 7] = g_minus * n_minus

    # Second row (a_-)
    M[1, 0] = 2 * Omega
    M[1, 1] = omega_a + 2 * Omega_minus
    M[1, 2] = g_minus * n_plus
    M[1, 3] = g_plus * n_minus
    M[1, 4] = 2 * Omega
    M[1, 5] = 2 * Omega_minus
    M[1, 6] = g_minus * n_plus
    M[1, 7] = g_plus * n_minus

    # Third row (b_+)
    M[2, 0] = g_plus * n_plus
    M[2, 1] = g_minus * n_plus
    M[2, 2] = omega_b
    M[2, 3] = 0 # doesnt talk to opposite handedness (?)    
    M[2, 4] = g_plus * n_plus
    M[2, 5] = g_minus * n_plus
    M[2, 6] = 0
    M[2, 7] = 0

    # Fourth row (b_-)
    M[3, 0] = g_minus * n_minus
    M[3, 1] = g_plus * n_minus
    M[3, 2] = 0
    M[3, 3] = omega_b
    M[3, 4] = g_minus * n_minus
    M[3, 5] = g_plus * n_minus
    M[3, 6] = 0
    M[3, 7] = 0    

    ## annihilators

    # Fifth row (a^{\dagger}_+)
    M[4, 0] = -2 * Omega_plus
    M[4, 1] = -2 * Omega
    M[4, 2] = -g_plus * n_plus
    M[4, 3] = -g_minus * n_minus
    M[4, 4] = -omega_a - 2 * Omega_plus
    M[4, 5] = -2 * Omega
    M[4, 6] = -g_plus * n_plus
    M[4, 7] = -g_minus * n_minus

    # Sixth row (a^{\dagger}_-)
    M[5, 0] = -2 * Omega
    M[5, 1] = -2 * Omega_minus
    M[5, 2] = -g_minus * n_plus
    M[5, 3] = -g_plus * n_minus
    M[5, 4] = -2 * Omega
    M[5, 5] = -omega_a - 2 * Omega_minus
    M[5, 6] = -g_minus * n_plus
    M[5, 7] = -g_plus * n_minus

    # Seventh row (b^{\dagger}_+)
    M[6, 0] = -g_plus * n_plus
    M[6, 1] = -g_minus * n_plus
    M[6, 2] = 0
    M[6, 3] = 0
    M[6, 4] = -g_plus * n_plus
    M[6, 5] = -g_minus * n_plus
    M[6, 6] = -omega_b
    M[6, 7] = 0
    
    # Eigth row (b^{\dagger}_-)
    M[7, 0] = -g_minus * n_minus
    M[7, 1] = -g_plus * n_minus
    M[7, 2] = 0
    M[7, 3] = 0
    M[7, 4] = -g_minus * n_minus
    M[7, 5] = -g_plus * n_minus
    M[7, 6] = 0
    M[7, 7] = -omega_b

    return M

# bosonic bogoliubov
# G = diag(1, -1)
# diagonalize G H
# from evs T = [x_1, ... J x_1, ...], where J (u v) = (conj(v) conj(u))
# => T^{\dagger} (GH) T diagonalizes, so T : matter, light => polaritons
# so we need T^{-1} : polaritons => matter, light
# construct by taking the positive eigenvectors
def renormalize_inv(inv: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Renormalizes the inverse matrix `inv` so that:
        inv_new @ G @ inv_new.T == I
    
    Parameters:
    inv (np.ndarray): Initial inverse matrix (n x n)
    G (np.ndarray): Given matrix (n x n)
    
    Returns:
    np.ndarray: Renormalized inverse matrix
    """
    # Compute the deviation matrix
    res = inv @ G @ inv.T
    
    # Eigen decomposition of res
    eigvals, eigvecs = np.linalg.eig(res)  # res = U @ Λ @ U.T
    
    # Compute inverse square root of eigenvalues
    inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))
    
    # Compute res^(-1/2) = U @ Λ^(-1/2) @ U.T
    res_inv_sqrt = eigvecs @ inv_sqrt_eigvals @ eigvecs.T
    
    # Compute renormalized inverse matrix
    inv_new = inv @ res_inv_sqrt
    
    return inv_new

def bogoliubov(M, ret_vals = False):
    """bogoliubov transformation matrix $T$ for a kernel M, i.e. the matrix that diagonalizes $H = Ma a^{\\dagger}$ via $a' = T a$ obeying $TgT^{\\dagger} = g$"""
    n = M.shape[0]//2
    
    G = np.diag([1 if i < n else -1 for i in range(2*n)])

    energies, vecs = np.linalg.eig(M)    

    # positive (ev dist symmetric around zero)
    positive = np.argsort(energies)[n:]
    
    # x is N x N / 2
    x = vecs[:, positive]
    
    # renorming is crucial (cf bf discussion)
    prod = x.T @ G @ x
    x = x / jnp.sqrt(jnp.diagonal(prod))

    # Jx is N x N / 2
    Jx = np.concatenate([x[n:], x[:n]])
    
    # T = [x_1, ..., J x_1, ...]
    T = np.concatenate([x, Jx], axis = 1)

    # inverse [ [U^+, -V^+],
    #           [-conj(V^+), conj(U^+)] ]
    # as long as real => + => T
    imag = vecs.imag
    if np.sum(imag > 1e-12): 
        print( "problem with imag", np.sum(imag > 1e-12))
    inv_left = np.concatenate([x[:n].T, -x[n:].T])
    inv_right = np.concatenate([-x[n:].T, x[:n].T])
    inv = np.concatenate( [inv_left, inv_right], axis = 1)

    # pseudo-unitarity
    res = inv @ G @ inv.T    
    diff = jnp.linalg.norm(res  - G)
    print(diff)
    if diff > 0.9:
        print("Not pseudo-unitary")
        import pdb; pdb.set_trace()
    
    if ret_vals == True:
        return inv, energies
    return inv

def plot_matter_content(debug = False):
    # light, matter resonances, PARAMETERS for omega_a, omega_b from [PhysRevLett.112.016401]
    omega_b = 1.
    omega_a = 1. / 1.1
    # chiral coupling
    g_plus = 0.5 * omega_a
    g_minus = g_plus
    
    kernel_func = lambda gp, gm : generate_kernel(omega_a, omega_b, gp, gm)
    kernel_old_func = lambda gp, gm : generate_kernel_old(omega_a, omega_b, gp, gm)

    gsmall = jnp.diag( jnp.array([1, 1, 1, -1, -1, -1]) )
    glarge = jnp.diag( jnp.array([1, 1, 1, 1, -1, -1, -1, -1 ]) )

    polariton_energy_state_idx = 0
    
    gs = np.linspace(0.01, 10, 10)    
    for a in [0.0, 0.1, 0.8]:
        res = []

        res_db = []
        for g in gs:        
            kernel = kernel_func(g, g * a)

            x = bogoliubov(kernel)

            i = polariton_energy_state_idx
            # content = (np.abs(x[3, i])**2 + np.abs(x[7, i])**2) / np.linalg.norm(x[:, i])**2
            content = (np.abs(x[2, i])**2 + np.abs(x[5, i])**2) / np.linalg.norm(x[:, i])**2
            res.append(content)

            # check for different enantiomer
            # some ordering issue mismatches polariton modes between chiralities, so the first - mode might be the second + mode and so obn
            if debug:
                kernel = kernel_func(g * a, g)
                x = bogoliubov(kernel)
                i = 1
                content = (np.abs(x[2, i])**2 + np.abs(x[5, i])**2) / np.linalg.norm(x[:, i])**2
                res_db.append(content)
            
        plt.plot(gs, res, label = r'$\frac{g_-}{g_+}$ = ' + f'{a}')
        if debug:
            plt.plot(gs, res_db, label = r'$\frac{g_-}{g_+}$ = ' + f'{a}')
        plt.xscale('log')

    # Add vertical lines and labels at specified x-axis positions
    coupling_points = [0.1, 0.5, 1]
    coupling_labels = ['Strong', 'Ultra', 'Deep']

    for point, label in zip(coupling_points, coupling_labels):
        plt.axvline(x=point, color='black', linestyle='--', alpha=0.7)  # Vertical line
        plt.text(point, 1.05, label, rotation=10, fontsize=8, ha='right', va='bottom')  # Label above        

    plt.fill_between(gs, 0.2, 0.8, color='gray', alpha=0.5, label='Strong Hybridization')
    plt.xlabel(r"$\dfrac{g_+}{\omega_b}$")
    plt.ylabel("Matter Content in Lowest Polariton Mode")
    plt.legend()        
    plt.savefig("matter_content.pdf")
    plt.close()

def plot_energies():
    omega_b = 1.
    omega_a = 1. / 1.7
    scales = np.linspace(0, 1, 40)

    arr = []
    for scale in scales:
        # chiral coupling
        g_plus = scale * omega_b    
        _, x_p = bogoliubov(generate_kernel(omega_a, omega_b, g_plus, 0, diamagnetic=True), ret_vals=True)
        _, x_m = bogoliubov(generate_kernel(omega_a, omega_b, 0, 0, diamagnetic=True), ret_vals=True)
        res = np.sort(x_p) - np.sort(x_m)
        arr.append(res[0])
        
    # Create the plot
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(scales, arr, label=r'Energy Difference $\Delta$', color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)

    # Add labels with larger font size
    plt.xlabel(r'Coupling $\sim \sqrt{N}$', fontsize=14)
    plt.ylabel(r'Energy Gap $\dfrac{\Delta}{\omega_b}$', fontsize=14)

    # Add title
    plt.title('Energy Gap vs Coupling Strength', fontsize=16)

    # Add gridlines
    plt.grid(alpha=0.3, linestyle='--')

    # Add a legend
    plt.legend(fontsize=12)

    # Improve axis tick parameters
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot
    plt.savefig('gap.pdf')

def compute_energy_transfer(w_p, w_m, t):
    """computes spectrally integrated energy transfer from bogoliubov matrix    

    Parameters:
       w_p, w_m : constants coupling cavity to classical light field
       t : bogoliubov transformation matrix
      
    Returns:
       float
    """

    # t[i, j] => i-th polariton mode receives contribution from j-th matter mode
    # matter modes here reside at indices 2, 5
    matter_content = np.abs(t[:, 2])**2
    matter_content = matter_content[:3] + matter_content[3:]

    # coupling efficiency of external illumination to modes indexed 0,1 => plus, minus
    w = np.array([w_p, w_m, 0])    
    u, v = t[:3, :3], t[:3, 3:]
    # import pdb; pdb.set_trace()

    # return (np.abs(t[2, 0])**2 + np.abs(t[5, 0])**2) / np.linalg.norm(t[:, 0])**2
    matter_norm  = matter_content / np.linalg.norm(t[:, :3], axis = 0)**2
    # import pdb; pdb.set_trace()
    return matter_content @ np.abs(w @ (u + v))**2
    # return np.sum(np.abs(w @ (u + v))**2)

def transfer_difference(omega_a, omega_b, w, wm, g, gm):
    # energy transferred to excess "plus"
    tp = compute_energy_transfer(w,
                                 wm,
                                 bogoliubov(generate_kernel(omega_a, omega_b, g, 0, diamagnetic = True)))

    # energy transferred to excess "minus"
    tm = compute_energy_transfer(w,
                                 wm,
                                 bogoliubov(generate_kernel(omega_a, omega_b, 0, gm, diamagnetic = True)))

    return (tp - tm)/(tp + tm)

def plot_energy_transfer():
    # light, matter resonances
    omega_b = 1.
    omega_a = 1. / 1.7
    # chiral coupling
    g_plus = 0.01 * omega_b

    # Example ranges for gs and ws, assuming they are arrays of values
    gs = np.linspace(0.01, 1.0, 50)  # range for g values
    ws = np.linspace(0.01, 1.0, 50)  # range for w values

    w_plus = 1.  # Example constant value for w_plus

    # Calculate the energy transfer matrix
    res = [[transfer_difference(omega_a, omega_b, w_plus, w * w_plus, g_plus, g * g_plus) for w in ws] for g in gs]

    res = np.array(res)

    # plt.plot(gs, res[:, 0])
    # plt.show()
    # 1/0
    
    # Plot the real part of the matrix res
    # res /= res.max()

    plt.figure(figsize=(8, 6))
    interpolation = 'sinc'
    # interpolation = None #'sinc'
    plt.imshow(np.real(res), extent=[ws.min(), ws.max(), gs.min(), gs.max()],
               origin='lower', aspect='auto', cmap='viridis', interpolation=interpolation)
    plt.colorbar(label=r'Energy selectivity $\dfrac{E_+ - E_-}{E_+ + E_-}$')
    plt.xlabel(r'cavity selectivity $w_- / w_+$')
    plt.ylabel(r'chirality $g_- / g_+$')
    plt.title('Energy selectivity')
    plt.savefig('transfer.pdf')
    plt.close()        

if __name__ == '__main__':
    # test_pra()
    plot_matter_content()
    # plot_energy_transfer()
    # plot_energies()
