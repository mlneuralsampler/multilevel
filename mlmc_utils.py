import torch
import torch.nn as nn
import numpy as np
from obs import Obs
from scipy.special import logsumexp
def grab(var):
    if torch.is_tensor(var):
        return var.detach().cpu().numpy()
    else:
        return var
def ising_energy_numpy(x,beta):
    energy=x*(np.roll(x,1,-1)+np.roll(x,1,-2))
    return -energy.sum((-2,-1))

def calculate_rev(work,L,ess):
    N = work.shape[0]
    log_ess = -np.log(N) + 2 * logsumexp(-work) - logsumexp(-2 * work)
    Ess = np.exp(log_ess)
    logZ = logsumexp(-work)-np.log(N)
    if ess:
        return Ess
    else:
        return -logZ/L**2/.44

def bootstrap_rev(work, L,ess, num_resamples=1000):
    N = work.shape[0]
    betaF_values = np.zeros(num_resamples)
    
    for i in range(num_resamples):
        resample = np.random.choice(work, size=N, replace=True)
        betaF_values[i] = calculate_rev(resample,L,ess)
    mean_betaF = np.mean(betaF_values)
    lower_bound = np.percentile(betaF_values, 16)
    upper_bound = np.percentile(betaF_values, 84)
    
    return mean_betaF, mean_betaF-lower_bound,upper_bound-mean_betaF


def gamma_analysis(work,L):
    #Loss
    obs1=Obs([work], ['ensemble1'])
    loss=np.mean(obs1)
    loss.gamma_method(S=0)
    del loss, obs1
    #betaF
    w0=np.mean(work)
    obs1= Obs([np.exp(-work+w0)], ['ensemble1'])
    betaF=-np.log(np.mean(obs1))+w0
    betaF.gamma_method(S=0)
    print("BetaF:")
    print(betaF.value/.44/L**2)
    print(betaF.dvalue/.44/L**2)
    #ESS
    obs2= Obs([np.exp(2*(-work+w0))], ['ensemble1'])
    ESS=np.mean(obs1)**2/np.mean(obs2)
    ESS.gamma_method(S=0)
    print("ESS:")
    print(ESS.value)
    print(ESS.dvalue) 
    return betaF.value/.44/L**2,betaF.dvalue/.44/L**2,ESS.value,ESS.dvalue

# Helper function to create a checkerboard mask
def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker

# Define the embedding class
class EmbeddingC_F(torch.nn.Module):
    def __init__(self, Lf, channels, device):
        super().__init__()
        self.Lf = Lf          # Final size after embedding
        self.device = device  # Device (e.g., cpu or cuda)
        self.channels = channels  # Number of channels

    def forward(self, sample_c):
        sample = torch.zeros((sample_c.shape[0], self.channels, self.Lf, self.Lf), device=sample_c.device)
        sample[:, :, ::2, ::2] = sample_c
        return sample, 0

    def reverse(self, sample_f):
        sample = sample_f[:, :, ::2, ::2]
        return sample, 0

class MLHB_level(torch.nn.Module):
    def __init__(self, Lf, beta, local_energy, device, level):
        super(MLHB_level, self).__init__()
        self.Lf = Lf
        self.beta = beta
        self.local_energy = local_energy
        self.device = device
        self.mask = make_checker_mask((self.Lf, self.Lf), 0)
        if level == 0:
            self.forward = self.forward_course
        elif level == 1:
            self.forward = self.forward_fine

    def forward_course(self, x):
        x = x.reshape(1, x.shape[-1], -1)
        local_en = (
            torch.roll(x, shifts=(1, 1), dims=(-2, -1)) +
            torch.roll(x, shifts=(1, -1), dims=(-2, -1)) +
            torch.roll(x, shifts=(-1, 1), dims=(-2, -1)) +
            torch.roll(x, shifts=(-1, -1), dims=(-2, -1))
        )
        x_hat = 1 / (1 + torch.exp(-2 * self.beta * local_en))
        with torch.no_grad():
            acc = x_hat > torch.rand(x.shape).to(x.device)
            x_new = torch.where(acc, 1.0, -1.0)
            x[:, 1::2, 1::2] = x_new[:, 1::2, 1::2]
        log_prob = torch.log(1 / (1 + torch.exp(-2 * self.beta * local_en * x)))
        log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(dim=1)
        return x.reshape(1, 1, x.shape[-1], -1), log_prob

    def forward_fine(self, x):
        x = x.reshape(1, x.shape[-1], -1)
        x_hat = 1 / (1 + torch.exp(-2 * self.beta * self.local_energy(x)))
        with torch.no_grad():
            acc = x_hat > torch.rand(x.shape).to(x.device)
            x_new = torch.where(acc, 1.0, -1.0)
        x = x_new * self.mask.to(x.device) + x
        log_prob = torch.log(1 / (1 + torch.exp(-2 * self.beta * self.local_energy(x) * x_new)))
        log_prob *= self.mask.to(x.device)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return x.reshape(1, 1, x.shape[-1], -1), log_prob

# Local energy function for the Ising model in 2D (nearest neighbor interactions)
def local_ising_energy(x):
    return (torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2))

def initial_state(N):
    return np.random.choice([-1, 1], size=(N, N))

def fine_density(lattice, K):
    ln_q = 0
    N = lattice.shape[-1]
    lattice = lattice.reshape(N, N)
    for i in range(N):
        for j in range(N):
            NN_sum = lattice[i, (j - 1) % N] + lattice[i, (j + 1) % N] + lattice[(i - 1) % N, j] + lattice[(i + 1) % N, j]
            dH = -K * NN_sum * lattice[i, j]
            ln_q += dH
    return -ln_q / 2.0

def coarse_density(lattice, K_1):
    ln_q = 0
    N = lattice.shape[-1]
    lattice = lattice.reshape(N, N)
    for i in range(N):
        for j in range(N):
            NN_sum = lattice[i, (j - 1) % N] + lattice[i, (j + 1) % N] + lattice[(i - 1) % N, j] + lattice[(i + 1) % N, j]
            dH = -K_1 * NN_sum * lattice[i, j]
            ln_q += dH
    return -ln_q / 2.0

def fine_proposal(small_sample, Lf, beta1, beta2):
    embedding_model = EmbeddingC_F(Lf, channels=1, device='cpu')
    hb_model_coarse = MLHB_level(Lf=Lf, beta=beta1, local_energy=local_ising_energy, device='cpu', level=0)
    hb_model_fine = MLHB_level(Lf=Lf, beta=beta2, local_energy=local_ising_energy, device='cpu', level=1)

    embedded_tensor, _ = embedding_model.forward(small_sample)
    coarse_output, coarse_log_prob = hb_model_coarse.forward_course(embedded_tensor)
    fine_output, fine_log_prob = hb_model_fine.forward_fine(coarse_output)

    return fine_output, fine_log_prob + coarse_log_prob
def heatbath_update_fast(lattice, T):
    N = lattice.shape[0]  # Assuming a square lattice
    for parity in [0, 1]:  # Checkerboard update for two parities (black and white)
        local_field = (np.roll(lattice, 1, axis=0) + np.roll(lattice, -1, axis=0) +
                       np.roll(lattice, 1, axis=1) + np.roll(lattice, -1, axis=1))
        for i in range(N):
            for j in range(N):
                if (i + j) % 2 == parity:  # Black or white lattice points
                    prob_up = 1 / (1 + np.exp(-2 * local_field[i, j] / T))
                    lattice[i, j] = 1 if np.random.rand() < prob_up else -1
    return lattice
