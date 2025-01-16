import numpy as np
import scipy
import matplotlib.pyplot as plt

# TODO: wtf am I seeing in relative deposited power?
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
    plt.savefig("matter_content.pdf")
    plt.close()
        
def generate_kernel(omega_a, omega_b, g_plus, g_minus):
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

def bogoliubov(M):
    """bogoliubov transformation matrix $T$ for a kernel M, i.e. the matrix that diagonalizes $H = Ma a^{\\dagger}$ via $a' = T a$ obeying $TgT^{\\dagger} = g$"""
    n = M.shape[0]//2
    
    vals, vecs = np.linalg.eig(M)    

    positive = np.argwhere(vals >= 0).squeeze()
    vecs = vecs[:, positive]
    ordered = np.argsort(vals[positive])
    vecs = vecs[:, ordered]   
    lower_part = np.concatenate([vecs[n:], vecs[:n]])
    
    T = np.zeros((2*n,2*n))

    T[:, n:] = lower_part
    T[:, :n] = vecs

    G = np.diag([1 if i < n else -1 for i in range(2*n)])
    res = T @ G @ T.T    
    T[:, :n] /= np.sqrt(np.diag(res)[:n])
    T[:, n:] /= np.sqrt(np.diag(res)[:n])

    x = np.linalg.inv(T)    
    res = x @ G @ x.T    
    x[:, :n] /= np.sqrt(np.diag(res)[:n])
    x[:, n:] /= np.sqrt(np.diag(res)[:n])
    res = x @ G @ x.T    
    # print(np.round(res, 2))
    
    return x

def plot_matter_content(debug = False):
    # light, matter resonances, PARAMETERS for omega_a, omega_b from [PhysRevLett.112.016401]
    omega_b = 1.
    omega_a = 1. / 1.1
    # chiral coupling
    g_plus = 0.5 * omega_a
    g_minus = g_plus
    
    kernel_func = lambda gp, gm : generate_kernel(omega_a, omega_b, gp, gm)

    gs = np.linspace(0.01, 10, 1000)    
    for a in [0.0, 0.1, 0.8]:
        res = []

        res_db = []
        for g in gs:        
            kernel = kernel_func(g, g * a)
            x = bogoliubov(kernel)            
            content = (np.abs(x[2, 0])**2 + np.abs(x[5, 0])**2) / np.linalg.norm(x[:, 0])**2
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
                                 bogoliubov(generate_kernel(omega_a, omega_b, g, gm)))

    # energy transferred to excess "minus"
    tm = compute_energy_transfer(w,
                                 wm,
                                 bogoliubov(generate_kernel(omega_a, omega_b, gm, g)))

    return tp - tm

def plot_energy_transfer():
    # light, matter resonances
    omega_b = 1.
    omega_a = 1. / 1.7
    # chiral coupling
    g_plus = 0.5 * omega_b

    # Example ranges for gs and ws, assuming they are arrays of values
    gs = np.linspace(0.01, 1.0, 50)  # range for g values
    ws = np.linspace(0.01, 1.0, 50)  # range for w values

    w_plus = 1.0  # Example constant value for w_plus

    # Calculate the energy transfer matrix
    res = [[transfer_difference(omega_a, omega_b, w_plus, w * w_plus, g, g * g_plus) for w in ws] for g in gs]

    res = np.array(res) 

    # plt.plot(gs, res[:, 0])
    # plt.show()
    # 1/0
    
    # Plot the real part of the matrix res
    res /= res.max()

    plt.figure(figsize=(8, 6))
    plt.imshow(np.real(res), extent=[ws.min(), ws.max(), gs.min(), gs.max()],
               origin='lower', aspect='auto', cmap='viridis', interpolation='sinc')
    plt.colorbar(label='Energy Transfer')
    plt.xlabel('w values')
    plt.ylabel('g values')
    plt.title('2D Plot of Energy Transfer (Real Part)')
    plt.show()
    plt.close()        

if __name__ == '__main__':
    # plot_matter_content()
    plot_energy_transfer()
