import numpy as np
import scipy
import matplotlib.pyplot as plt

# chosen vals from paper: w_b = w_a, freq-independent rates g_a = 0.04 w_a, g_b = 0.08 w_a
# plot abs(w) for varying values of g_+ / g_-, c_+ / c_-
# plot light-matter content
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
    Constructs the kernel

    Parameters:
        omega_a: Parameter omega_a.
        omega_b: Parameter omega_b.
        Omega: Parameter Omega.
        Omega_plus: Parameter Omega_+.
        Omega_minus: Parameter Omega_-.
        g_plus: Parameter g_+.
        g_minus: Parameter g_-.

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
    
def generate_m_matrix(omega, omega_a, omega_b, g_plus, g_minus, Gamma_pp, Gamma_pm, Gamma_mm, Gamma_b):
    """
    Constructs the M-matrix

    Parameters:
        omega: Frequency argument.
        omega_a: Parameter omega_a.
        omega_b: Parameter omega_b.
        g_plus: Parameter g_+.
        g_minus: Parameter g_-.
        Gamma_pp: Function Gamma_{++}(omega).
        Gamma_pm: Function Gamma_{+-}(omega).
        Gamma_mm: Function Gamma_{--}(omega).
        Gamma_b: Function Gamma_{b}(omega).

    Returns:
        A 6x6 numpy matrix.
    """    
    kernel = generate_kernel(omega_a, omega_b, g_plus, g_minus)

    # First row
    kernel[0, 0] += Gamma_pp(omega)
    kernel[0, 1] += Gamma_pm(omega)

    # Second row
    kernel[1, 0] += Gamma_pm(omega)
    kernel[1, 1] += Gamma_mm(omega)

    # Third row
    kernel[2, 2] += Gamma_b(omega)

    # Fourth row
    kernel[3, 3] += Gamma_pp(-omega)
    kernel[3, 4] +=  Gamma_pm(-omega)

    # Fifth row
    kernel[4, 3] += np.conj(Gamma_pm(-omega))
    kernel[4, 4] += np.conj(Gamma_mm(-omega))

    # Sixth row
    kernel[5, 5] += np.conj(Gamma_b(-omega))

    return -omega * np.eye(*kernel.shape) + kernel

def bogoliubov(M):
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
    print(np.round(res, 2))
    
    return x

def plot_matter_content(kernel_func):    
    gs = np.linspace(0.01, 10, 1000)    
    for a in [0.0, 0.1, 0.8]:
        res = []
        for g in gs:        
            kernel = kernel_func(g, g*a)
            x = bogoliubov(kernel)            
            content = (np.abs(x[2, 0])**2 + np.abs(x[5, 0])**2) / np.linalg.norm(x[:, 0])**2
            res.append(content)
            
        plt.plot(gs, res, label = r'$\frac{g_-}{g_+}$ = ' + f'{a}')
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

if __name__ == '__main__':    
    ## system parameters##
    # light, matter resonances
    omega_b = 1.
    omega_a = 1.# / 1.7
    # chiral coupling
    g_plus = 0.5 * omega_a
    g_minus = g_plus
    
    kernel_func = lambda gp, gm : generate_kernel(omega_a, omega_b, gp, gm)
    # plot_matter_content(kernel_func)
    
    ## bath parameters ##
    # chiral reflectivities (non-dispersive)
    c_ph_p, c_ph_m = 0.5, 0.5
    # electronic bath coupling
    c_el = 0.5
    # photonic bath DOS (non-dispersive)
    rho_ph = 0.01
    # electronic bath DOS (non-dispersive)
    rho_el = 0.01

    for c_scale in [1.0, 0.1, 0.01]:
        c_ph_m = c_scale * c_ph_p
        
        # chiral losses (imaginary part is parametric)
        imag = 0# 1e-4j * omega_a
        wrap = lambda c : lambda w : c  if w >= 0 else 0
        Gamma_pp = wrap(c_ph_p**2 * rho_ph + imag)
        Gamma_mp = wrap(c_ph_p*c_ph_m * rho_ph + imag)
        Gamma_pm = wrap(c_ph_p*c_ph_m * rho_ph + imag)
        Gamma_mm = wrap(c_ph_m**2 * rho_ph + imag)
        # electronic losses (non-dispersive)
        Gamma_b = wrap(c_el ** 2 * rho_el + imag)

        ws = np.linspace(omega_b - 1*omega_b, omega_b + 1*omega_b, 150)
        res = []
        for omega in ws:
            m_matrix = generate_m_matrix(omega, omega_a, omega_b, g_plus, g_minus, Gamma_pp, Gamma_pm, Gamma_mm, Gamma_b)
            i_matrix =  np.array([
                [c_ph_p * rho_ph, 0], 
                [c_ph_m * rho_ph, 0],
                [0, c_el * rho_el]
            ], dtype = complex)    
            prefac_matrix = np.array( [ [c_ph_p, c_ph_m, 0], [0, 0, c_el] ] )
            u = np.eye(2) + prefac_matrix @ (np.linalg.inv(m_matrix)[:3, :3]) @ i_matrix
            scattering_matrix = u#np.linalg.qr(u, mode = 'complete').Q
            print(scattering_matrix @ scattering_matrix.conj().T)
            absorption = np.abs(scattering_matrix[0,1])
            res.append(absorption)
        res = np.array(res)
        plt.plot(ws, res, label = r'$\frac{c_-}{c_+}$ = ' + f'{c_scale}')
        
    plt.xlabel(r"$\dfrac{\omega}{\omega_b}$")
    plt.ylabel("% Absorption")    
    plt.legend()
    plt.show()
    # plt.savefig("abs.pdf")
    plt.close()
