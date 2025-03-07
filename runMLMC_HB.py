# main.py
import sys
import pyerrors as pe
import emcee
import argparse
import numpy as np
import torch
import sys
sys.path.append('/home/ankur/HAN/MLMC_HB')
from mlmc_hb import*
from obs import Obs
# Main function to run the MCMC simulation and multi-level updates
def main(args):
    N = args.N  # Lattice size
    T = 1 / args.beta_c  # Temperature
    level = args.level
    beta_c = args.beta_c
    beta_I = args.beta_I
    beta_f = args.beta_f
    sample_size=args.sample_size
    print("Coarse Lattice, L:", N)
    print("Fine Lattice, L:", 2**(level+1))
    print("beta_c:", beta_c)
    print("beta_I", beta_I)
    print("beta_f", beta_f)
    print("Sample size:", sample_size)
    # Initialize lattice
    lattice = initial_state(N)
    # Perform MCMC sampling using the heat bath method
    lat = []
    for i in range(10*sample_size):  # Number of iterations
        lattice = heatbath_update_fast(lattice, T)
        if i % 10 == 0:  # Save every 10th sample
            lat.append(np.copy(lattice))

    conf = np.array(lat)
    conf = conf.reshape(-1, N * N)  # Reshape configurations
    Coarse_Conf = conf
    # Sampling and acceptance loop
    Mag=[]
    W=[]
    J=1
    Z = 2 *np.exp(4*beta_c *J) + 8 + 6* np.exp(-4 *beta_c* J)
    for i in range(0, sample_size):
        conf_c = Coarse_Conf[i].reshape(N, N)
        lnpc= coarse_density(conf_c, beta_c)-np.log(Z)
        
        proposal = torch.tensor(conf_c.reshape(1, 1, N, N))
        lnq = lnpc
        for l in range(level):
            proposal, p_curr_prop = fine_proposal(proposal, 2 * proposal.shape[-1], beta_I, beta_f)
            lnq += p_curr_prop
        E=-fine_density(proposal.reshape(proposal.shape[-1], proposal.shape[-1]), beta_f) 
        w=beta_f*E+lnq
        W.append(grab(w))
        #mag=torch.mean(fine_lattice.flatten())
        #Mag.append(np.abs(mag))
        #En.append(E/proposal.shape[-1]**2)
        progress = i * 100 / (sample_size)
        sys.stdout.write(f"\rProgress: {progress:.2f}%")
        sys.stdout.flush()
    W_array=np.array(W)
    print("W shape:", W_array.shape)
    F,F_e, ess, ess_e=gamma_analysis(-W_array.flatten(),proposal.shape[-1])
    print(f" Gamma method--> Ensemble quantities:  ESS: {ess} ± {ess_e}, F: {F*.44*(proposal.shape[-1])**2} ± {F_e}")

    
    F_8,F_c1_8,F_c2_8=bootstrap_rev(-W_array.flatten(),proposal.shape[-1],ess=False)
    Ess_8,ess_c1_8,ess_c2_8=bootstrap_rev(-W_array.flatten(),proposal.shape[-1],ess=True)
    print(f" Bootsrapp-->Ensemble quantities:  ESS: {Ess_8} ± {ess_c1_8}, F: {F_8*.44*(proposal.shape[-1])**2} ± {F_c1_8}")
      #Z = 2 * np.exp(4 * beta * J) + 8 + 6 * np.exp(-4 * beta * J)
###python main.py --N 4 --level 4 --beta_c 0.39 --beta_I 0.44 --beta_f 0.44 sample_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-level MCMC with Heat Bath and MLHB')
    parser.add_argument('--N', type=int, default=2, help='Lattice size (N x N)')
    parser.add_argument('--level', type=int, default=4, help='Number of levels in MLHB')
    parser.add_argument('--beta_c', type=float, default=0.39, help='Beta for coarse level')
    parser.add_argument('--beta_I', type=float, default=0.44, help='Intermediate beta')
    parser.add_argument('--beta_f', type=float, default=0.44, help='Beta for fine level')
    parser.add_argument('--sample_size', type=int, default=1000, help='Ensemble size')

    args = parser.parse_args()

    # Call the main function
    main(args)
