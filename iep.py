# Irreversible Entropy Production in open quantum systems
# M. Garbellini

from qutip import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools
from more_itertools import distinct_permutations as idp



def kronecker_identity_permutations(op, n, const):
    operators = list([op] + list(itertools.repeat(0, n-1)))
    permutations = [list(l) for l in list(set(itertools.permutations(operators)))]
    oplist = [identity(2), sigmax(), sigmay(), sigmaz(), sigmap(), sigmam()]

    for i in range(len(permutations)):
        for j in range(len(permutations[i])):
            permutations[i][j] = oplist[permutations[i][j]]
            if permutations[i][j] != 0 : permutations[i][j] *= const[i]
    kronk_list = []
    for perm in permutations:
        kronk_list.append(tensor(perm))
    kronk = sum(kronk_list)
    return kronk
def energy_gaps(a): return a[0]-a[1]

def non_degenerate_energy_gaps(H):
    eigen = H.eigenenergies()
    gaps = list(map(energy_gaps,[list(l) for l in list(set(itertools.permutations(eigen, 2)))]))
    num_unique_gaps = len(set(gaps))
    if len(gaps) != num_unique_gaps:
        print("Hamiltonian has degenerate energy gaps")

    print("...Hamiltonian has non-degenerate energy gaps...")
    print("...finding equilibrium state...")
    Ek = H.eigenstates()
    ck2 = np.zeros(2**(envN+1), dtype=np.complex_)
    for j in range(2**(envN+1)):
        ck2[j] = rho.matrix_element(Ek[1][j].dag(), Ek[1][j])
    omega = []
    for j in range(2**(envN+1)):
        omega.append(ck2[j]*(Ek[1][j]*Ek[1][j].dag()))
    omega = sum(omega)
    rhoeq = omega.ptrace(0)
    beta = np.log((1-rhoeq.diag()[0])/rhoeq.diag()[0])/2
    bound = 0.5*np.sqrt(4/np.sum(ck2**2))

    return(omega,rhoeq,beta,bound)

envN = 10
omega_env = np.random.uniform(1.1, 1.2, envN)
coupling = np.random.uniform(0.1, 0.5, envN)
beta = 0.5
rho00 = 1.
rho01 = 0.

# Hamiltonians
HE = kronecker_identity_permutations(3, envN, omega_env)
HS = sigmaz()
HI = tensor(sigmap(),kronecker_identity_permutations(5, envN, coupling)) + tensor(sigmam(),kronecker_identity_permutations(4, envN, coupling))
H = tensor(identity(2), HE) + tensor(HS,tensor(list(itertools.repeat(identity(2), envN)))) + HI

#initial states
rhoE = (-beta*HE).expm()
rhoE /= rhoE.tr()
rhoS = Qobj([[rho00, rho01], [np.conjugate(rho01),1-rho00]])
rho = tensor(rhoE, rhoS)

#non-degenerate energy gaps
omega,rhoeq,beta,bound = non_degenerate_energy_gaps(H)
print(omega.shape)
print(rhoeq)
print(beta, np.real(bound))
