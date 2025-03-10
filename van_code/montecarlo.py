import torch
from van_code.utils import grab
import time


def serial_sample_generator(model, batch_size, N_samples):
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            x,logq=model(batch_size)
            logq=logq.squeeze()
            logp = -model.energy(x,model.beta)

        yield x[batch_i], logq[batch_i], logp[batch_i]


def make_mcmc_ensemble(model, batch_size, N_samples,device):
    history = {
        'x' : [],
        'logq' : [],
        'logp' : [],
        'accepted' : [],
        'time': [],
    }

    # build Markov chain
    sample_gen = serial_sample_generator(model, batch_size, N_samples)
    for param in model.parameters():
        param.requires_grad = False
    t0=time.time()
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            # always accept first proposal, Markov chain must start somewhere
            accepted = True
            p_accept=1
            draw=0
        else:
            # Metropolis acceptance condition
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1).to(device) # ~ [0,1]
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq
        # Update Markov chain
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(grab(new_x))
        history['accepted'].append(accepted)
    t1=time.time()
    history['time'].append(t1-t0)
    return history
