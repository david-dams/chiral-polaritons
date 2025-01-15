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


def compute_energy_transfer(w_p, w_m, t):
    """computes spectrally integrated energy transfer from bogoliubov matrix    

    Parameters:
       w_p, w_m : constants coupling cavity to classical light field
       t : bogoliubov transformation matrix
      
    Returns:
       float
    """

    omega_a, omega_b = 1., 1.    
    kernel = generate_kernel(omega_a, omega_b, g_plus, g_minus)
    x = bogoliubov(kernel)            

if __name__ == '__main__':    
    compute_energy_transfer()
