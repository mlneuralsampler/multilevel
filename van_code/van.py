import torch
import torch.nn as nn
import numpy as np
from numpy import log
from van_code.utils import compute_metrics, grab, print_metrics
import time


class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2 + 1:] = 0
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

    def extra_repr(self):
        return (super(MaskedConv2d, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


class PixelCNN(nn.Module):
    def __init__(self, **kwargs):
        super(PixelCNN, self).__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.half_kernel_size = kwargs['half_kernel_size']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.final_conv = kwargs['final_conv']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5

        layers = []
        layers.append(
            MaskedConv2d(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=True))
        for count in range(self.net_depth - 2):
            if self.res_block:
                layers.append(
                    self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(
                    self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.net_width, self.net_width if self.final_conv else 1))
        if self.final_conv:
            layers.append(nn.PReLU(self.net_width, init=0.5))
            layers.append(nn.Conv2d(self.net_width, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers).to(self.device)

        self.energy_layer=MaskedConv2d(
                1,
                1,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=True)

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.bias))
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):
        x_hat = self.net(x)

        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                          1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat

    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            device=self.device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(sample)
                sample[:, :, i, j] = torch.bernoulli(
                    x_hat[:, :, i, j]) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = torch.randint(
                2, [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob

    def mcmc(self,batch_shape):
        samples,_=self.sample(batch_shape)
        log_prob=self.log_prob(samples)
        return samples, log_prob


def train_van(net, energy, beta, nepochs, batch_size, optimizer, print_freq, history_file):
    history={'rein_loss': [], 'varF': [], 'betaF': [], 'ESS': []}
    t0=time.time()
    for epoch in range(nepochs):
        optimizer.zero_grad()
        with torch.no_grad():
            sample, x_hat=net.sample(batch_size)

        log_prob = net.log_prob(sample)
        with torch.no_grad():
            w=energy(sample.squeeze(),beta)+log_prob
            ess, betaF=compute_metrics(w)

        reinforce_loss=torch.mean((w-w.mean())*log_prob)
        reinforce_loss.backward()
        optimizer.step()

        history['rein_loss'].append(grab(reinforce_loss))
        history['varF'].append(grab(w.mean()))
        history['betaF'].append(grab(betaF))
        history['ESS'].append(grab(ess))
        if (epoch+1)%print_freq==0:
            print_metrics(history_file,history, epoch+1, print_freq, t0)

    history['time']=time.time()-t0
    return history


def sample_van(net, energy, beta, batch_size, nbatch):
    W, m, En, t0 = [], [], [], time.time()
    with torch.no_grad():
        for i in range(nbatch):
            sample, _ =net.sample(batch_size)
            log_p=net.log_prob(sample)
            E=energy(sample.squeeze(),beta)
            w=E+log_p
            W.append(grab(w))
            En.append(grab(E))
            m.append(grab(sample.mean((-2,-1))))
    t1=time.time()
    W=np.asarray(W).reshape(-1)
    En=np.asarray(En).reshape(-1)
    m=np.asarray(m).reshape(-1)
    return W,En,m,t1-t0


def sample_forward(net,energy,beta,config,nbatch,device):
    W=[]
    t0=time.time()
    with torch.no_grad():
        n,T,R=config.shape
        for i in range(nbatch):
            samples=torch.tensor(config.reshape((nbatch,n//nbatch,T,R))[i]).float().squeeze().to(device)
            log_p=net.log_prob(samples.unsqueeze(1))
            E=energy(samples.squeeze(),beta)
            w=E+log_p
            W.append(grab(w))
    t1=time.time()
    return np.asarray(W).reshape(-1),t1-t0


def serial_sample_generator(model, batch_size, N_samples):
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            x,_=model.sample(batch_size)
            logq= model.log_prob(x)
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
