
import torch
import torch.nn as nn
import numpy as np
from numpy import log
from van_code.utils import compute_metrics, grab, print_metrics, make_checker_mask, save
import time
import os
import sys
import gc
sys.stdout.flush()


def make_conv_net(
        hidden_sizes,
        kernel_size,
        in_channels=1,
        out_channels=1,
        padding=None,
        stride=None,
        use_tanh=False,
        use_bias=True,
        sig_last=True
):
    '''
    Convolutionaal Neural Netowrk
    hiddens_sizes=[N_filters for hidden layer 1, .... ,N_filters for hidden layer n]
    num hidden layers = len(hidden_sizes)
    '''
    sizes = [in_channels] + hidden_sizes + [out_channels]
    if padding:
        padding_size = padding
    else:
        padding_size = np.asarray(kernel_size) // 2
    if not stride:
        stride = 1
    net = []
    for i in range(len(sizes) - 1):
        if sizes[i+1]==out_channels:
            conv = torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size[i], padding=padding_size[i], stride=stride, padding_mode='circular', bias=use_bias)
        else:
            conv = torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size[i], padding=padding_size[i], stride=1, padding_mode='circular', bias=use_bias)
        #torch.nn.init.xavier_normal_(conv.weight, gain=torch.nn.init.calculate_gain('tanh'))
        #torch.nn.init.normal_(conv.weight, std=0.01)
        net.append(conv)
        if i != len(sizes) - 2:
            if (use_tanh):
                net.append(torch.nn.Tanh())
            else:
                net.append(torch.nn.LeakyReLU(0.1))

    #net[-1].weight.data.zero_()
    #if use_bias:
        #torch.nn.init.constant_(net[-1].bias.data, 0.0)
    if sig_last:
        net.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*net)


###PixelCNN
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
        self.net = nn.Sequential(*layers)#.to(self.device)

        #torch.nn.init.ones_(self.energy_layer.weight.data)
        #self.energy_layer.bias.data.zero_()

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

    def forward(self,sample):
        with torch.no_grad():
            sample, _ = self.sample(sample)
        log_prob= self.log_prob(sample)
        return sample, log_prob

    def forward_net(self, x):
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

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # x_hat = p(x_{i, j} == +1 | x_{0, 0}, ..., x_{i, j - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    def sample(self, sample):
        #sample = torch.zeros(
         #   [batch_size, 1, self.L, self.L],)
            #device=next(self.net.parameters()).device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward_net(sample)
                sample[:, :, i, j] = torch.bernoulli(
                    x_hat[:, :, i, j]) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = torch.randint(
                2, [sample.shape[0], 1, 1, 1],
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
        x_hat = self.forward_net(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward_net(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob

### Conditional VAN for Multilevel
class VAN_CNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAN_CNN,self).__init__()
        self.hidden_size=kwargs['hidden_size']
        self.kernel_size=kwargs['kernel_size']
        self.epsilon = kwargs['epsilon']
        self.device=kwargs['device']
        self.mixture_components=6
        self.net=make_conv_net(hidden_sizes=self.hidden_size, kernel_size=self.kernel_size)#.to(self.device)
        ### level=0 -> intermediate level
        ### level=1 -> fine level
        if kwargs['level'] == 0:
            self._cond=self._cond_inter
        elif kwargs['level'] == 1:
            self._cond = self._cond_fine

    def _cond_inter(self, i, j):
        return (i%2!=0 and j%2!=0)

    def _cond_fine(self,i,j):
        return (i%2==0 and j%2!=0) or (i%2!=0 and j%2==0)

    def forward(self,x):
        logq=torch.zeros(x.shape[0]).to(x.device)
        for i in range(x.shape[-2]):
            for j in range(x.shape[-1]):
                if self._cond(i,j):
                    x_hat=self.net(x)
                    with torch.no_grad():
                        x[:,:,i,j]=torch.bernoulli(x_hat[:,:,i,j])*2-1
                    logq+=(torch.log(x_hat[:,:,i,j]+self.epsilon)*((x[:,:,i,j]+1)/2)+torch.log(1-x_hat[:,:,i,j]+self.epsilon)*(1-((x[:,:,i,j]+1)/2))).squeeze()
                    #print('vacnn forward',i,j,logq.mean(),x[0,:,i,j],x_hat[0,:,i,j])
        return x,logq

    def reverse(self,x):
        logq=torch.zeros(x.shape[0]).to(x.device)
        for i in reversed(range(x.shape[-2])):
            for j in reversed(range(x.shape[-1])):
                if self._cond(i,j):
                    xij=torch.clone(x[:,:,i,j])
                    with torch.no_grad():
                        x[:,:,i,j]=0
                    x_hat=self.net(x)
                    logq+=(torch.log(x_hat[:,:,i,j]+self.epsilon)*((xij+1)/2)+torch.log(1-x_hat[:,:,i,j]+self.epsilon)*(1-((xij+1)/2))).squeeze()
        return x,logq


class HB_level(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HB_level,self).__init__()
        self.Lf=kwargs['Lf']
        self.beta=kwargs['beta']
        self.local_energy=kwargs['local_energy']
        self.device=kwargs['device']
        self.mask=make_checker_mask((self.Lf,self.Lf),0)#.to(self.device)
        if kwargs['level']==0:
            self.forward=self.forward_course
        elif kwargs['level']==1:
            self.forward=self.forward_fine
            self.reverse=self.reverse_fine

    def forward_course(self,x):
        x=x.squeeze()
        #print(x[0])
        local_e=self.local_energy(x)
        x_hat= 1/(1+torch.exp(-2*self.beta*self.local_energy(x)))
        with torch.no_grad():
            acc = x_hat > torch.rand(x.shape).to(x.device)
            x_new=torch.where(acc,1.0,-1.0)
            #print(x_new[0,1::2,1::2])
            x[:,1::2,1::2]=x_new[:,1::2,1::2]
        #print(x[0])
        log_prob=torch.log(1/(1+torch.exp(-2*self.beta*self.local_energy(x)*x)))[:,1::2,1::2]
        #print(log_prob[0])
        log_prob=log_prob.reshape(log_prob.shape[0],-1).sum(dim=1)
        return x.unsqueeze(1),log_prob

    def forward_fine(self,x):
        x=x.squeeze()
        local_e=self.local_energy(x)
        x_hat= 1/(1+torch.exp(-2*self.beta*self.local_energy(x)))
        with torch.no_grad():
            acc = x_hat > torch.rand(x.shape).to(x.device)
            x_new=torch.where(acc,1.0,-1.0)

        x=x_new*self.mask.to(x.device)+x
        log_prob=torch.log(1/(1+torch.exp(-2*self.beta*self.local_energy(x)*x_new)))
        log_prob*=self.mask.to(x.device)
        log_prob=log_prob.view(log_prob.shape[0],-1).sum(dim=1)
        return x.unsqueeze(1),log_prob

    def reverse_fine(self,x):
        x=x.squeeze()
        log_prob=torch.log(1/(1+torch.exp(-2*self.beta*self.local_energy(x*(1-self.mask.to(x.device)))*x)))
        log_prob*=self.mask.to(x.device)
        log_prob=log_prob.view(log_prob.shape[0],-1).sum(dim=1)
        return (x*(1-self.mask.to(x.device))).unsqueeze(1),log_prob

class VANUpsampling(torch.nn.Module):
    '''from Course to Fine with VAN: two level step q_i and q_f
       Heatbath can be used for q_f '''
    def __init__(self,net_i,net_f):
        super().__init__()
        self.net_i=net_i
        self.net_f=net_f


    def forward(self,sample_c):
        sample_i,logq_i=self.net_i(sample_c)
        sample_f,logq_f=self.net_f(sample_i)
        return sample_f,logq_i+logq_f

    def reverse(self,sample_f):
        sample_i,logq_f = self.net_f.reverse(sample_f)
        sample_c,logq_i =self.net_i.reverse(sample_i)
        return sample_c, logq_i+logq_f

class EmbeddingC_F(torch.nn.Module):
    def __init__(self,Lf,channels,device):
        super().__init__()
        self.Lf=Lf
        self.device=device
        self.channels=channels

    def forward(self,sample_c):
        with torch.no_grad():
            sample=torch.zeros((sample_c.shape[0],self.channels,self.Lf,self.Lf),device=sample_c.device)
            sample[:,:,::2,::2]=sample_c
        return sample, 0

    def reverse(self, sample_f):
        with torch.no_grad():
            #sample=torch.zeros((sample_f.shape[0],self.channels,self.Lf//2,self.Lf//2),device=sample_f.device)
            sample = sample_f[:, :, ::2, ::2]
        return sample, 0


class MultilevelBlock(torch.nn.Module):
    def __init__(self,embedding, van_upsampling):
        super().__init__()
        self.embedding=embedding
        self.van_upsampling=van_upsampling

    def forward(self, sample):
        sample, _ = self.embedding(sample)
        sample, dlogq = self.van_upsampling(sample)
        #print('inside block, gpu:',sample.device)
        return sample, dlogq

    def reverse(self, sample):
        sample, dlogq = self.van_upsampling.reverse(sample)
        sample, _ = self.embedding.reverse(sample)
        return sample, dlogq


class Multilevel(torch.nn.Module):
    def __init__(self, Lc, net_hyp, nlevels, hb_last, local_energy, beta, device):
        super().__init__()
        self.Lc = Lc
        self.Lf = Lc
        self.nlevels = nlevels
        self.in_channels = 1
        self._init_blocks(net_hyp, hb_last, local_energy, beta, device)
        self.current_level = nlevels-1

    def _init_blocks(self, net_hyp, hb_last, local_energy, beta, device):
        layers = []
        j = 0
        print("k_size",net_hyp['kernel_size'])
       # net_hyp['kernel_size'] = [net_hyp['kernel_size'][i:i + 2] for i in range(0, len(net_hyp['kernel_size']), 2)]
       # net_hyp['hidden_size']=net_hyp['hidden_size'].reshape(-1,2)
        for i in range(self.nlevels-1):
            self.Lf *= 2
            embedding = EmbeddingC_F(self.Lf, int(self.in_channels), device)
            net_i = VAN_CNN(
                hidden_size=net_hyp['hidden_size'][j],
                kernel_size=net_hyp['kernel_size'][j],
                epsilon=net_hyp['epsilon'],
                channels=int(self.in_channels),
                level=0,
                device=device
            )
            print('nnot gd',j,net_hyp['kernel_size'][j])
            j += 1
            net_f = VAN_CNN(
                hidden_size=net_hyp['hidden_size'][j],
                kernel_size=net_hyp['kernel_size'][j],
                epsilon=net_hyp['epsilon'],
                channels=int(self.in_channels),
                level=1,
                device=device
            )
            print(j,net_hyp['kernel_size'][j])
            j += 1
            van_cnn = VANUpsampling(net_i, net_f)
            layers.append(MultilevelBlock(embedding, van_cnn))

        self.Lf *= 2
        print('Lf=',self.Lf)
        embedding = EmbeddingC_F(self.Lf, int(self.in_channels), device)
        net_i = VAN_CNN(
            hidden_size=net_hyp['hidden_size'][j],
            kernel_size=net_hyp['kernel_size'][j],
            channels=int(self.in_channels),
            epsilon=net_hyp['epsilon'],
            level=0,
            device=device
        )
        print(j, net_hyp['kernel_size'][j])

        if hb_last:
            net_f = HB_level(
                Lf=self.Lf,
                beta=beta,
                local_energy=local_energy,
                level=1,
                device=device
            )
            van_cnn = VANUpsampling(net_i, net_f)
            layers.append(MultilevelBlock(embedding, van_cnn))
            j += 1
        else:
            net_f = VAN_CNN(
                hidden_size=net_hyp['hidden_size'][j],
                kernel_size=net_hyp['kernel_size'][j],
                channels=self.in_channels,
                epsilon=net_hyp['epsilon'],
                level=1,
                device=device
            )
            van_cnn = VANUpsampling(net_i, net_f)
            layers.append(MultilevelBlock(embedding, van_cnn))

        self.layers = torch.nn.ModuleList(layers)

    def freeze_all_layers(self):
        """Freeze all layers initially."""
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, until):
        """Unfreeze all layers up to and including the specified layer index."""
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                param.requires_grad = True
            if i == until:
                break

    def forward(self, sample, log_prob):
        for i in range(self.current_level + 1):  # Use only the unfreezed layers
            sample, dlog = self.layers[i](sample)
            log_prob = log_prob + dlog
        return sample, log_prob

    def reverse(self, sample, log_prob):
        for i in reversed(range(self.current_level+1)):
            sample, dlog = self.layers[i].reverse(sample)
            log_prob = log_prob + dlog
        return sample, log_prob
    def load_pretrained(self, level):
        if (level+1) == self.nlevels:
            prev_layer = self.layers[level-1].van_upsampling.net_i
            self.layers[level].van_upsampling.net_i.load_state_dict(prev_layer.state_dict())
        else:
            prev_layer = self.layers[level-1]
            self.layers[level].load_state_dict(prev_layer.state_dict())
    def transfer_weights_adaptive(self,source_layer, target_layer,k1,k2):
        # Get kernel sizes
        K_init = k1
        K_final =k2
        # Get the source and target weights
        source_weights = source_layer.weight.data
        target_weights = target_layer.weight.data
        # Initialize target weights to zero
         #target_weights.zero_()

        if K_init == K_final:
             print("Good")
        # If the kernel sizes are the same, copy directly
             target_weights = source_weights.clone()
        elif K_init < K_final:
             print("Not not 1")
         # Case: K_init is smaller, place the weights in the center of the larger kernel
             center_offset = (K_final - K_init) // 2
             target_weights[:, :, center_offset:center_offset + K_init, center_offset:center_offset + K_init] = source_weights
        else:
             print("Not good 2")
             # Case: K_init is larger, extract the central part to fit into the smaller kernel
             center_offset = (K_init - K_final) // 2
             target_weights = source_weights[:, :, center_offset:center_offset + K_final, center_offset:center_offset + K_final].clone()
         # Assign the adapted weights to the target layer
        target_layer.weight.data = target_weights

         # Transfer the bias directly if it exists
        if source_layer.bias is not None and target_layer.bias is not None:
             target_layer.bias.data = source_layer.bias.data.clone()
             print("Is it true :bias")
    def load_diff_kernel(self,level):
        if self.nlevels ==2:
             net1_i = self.layers[level-1].van_upsampling.net_i.net
             net2_i=self.layers[level].van_upsampling.net_i.net
             e=self.layers[level-1].van_upsampling.net_i.kernel_size
             f=self.layers[level].van_upsampling.net_i.kernel_size
             conv_layers_1 = [layer for layer in net1_i if isinstance(layer, nn.Conv2d)]
             conv_layers_2 = [layer for layer in net2_i if isinstance(layer, nn.Conv2d)]


             for layer1, layer2,e1,f1 in zip(conv_layers_1, conv_layers_2,e,f):
                  print(e1,f1)
                  if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                       self.transfer_weights_adaptive(layer1, layer2,e1,f1)
        else:
             net1_i = self.layers[level-1].van_upsampling.net_i.net
             net2_i=self.layers[level].van_upsampling.net_i.net
             e=self.layers[level-1].van_upsampling.net_i.kernel_size
             f=self.layers[level].van_upsampling.net_i.kernel_size
             conv_layers_1 = [layer for layer in net1_i if isinstance(layer, nn.Conv2d)]
             conv_layers_2 = [layer for layer in net2_i if isinstance(layer, nn.Conv2d)]


             for layer1, layer2,e1,f1 in zip(conv_layers_1, conv_layers_2,e,f):
                  print(e1,f1)
                  if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                       self.transfer_weights_adaptive(layer1, layer2,e1,f1)

             net_prev_f= self.layers[level-2].van_upsampling.net_f
             net1_f=net_prev_f.net
             k1_f=net_prev_f.kernel_size
             net_curr_f=self.layers[level-1].van_upsampling.net_f
             net2_f=net_curr_f.net
             k2_f=net_curr_f.kernel_size
             conv_layers_1 = [layer for layer in net1_f if isinstance(layer, nn.Conv2d)]
             conv_layers_2 = [layer for layer in net2_f if isinstance(layer, nn.Conv2d)]

             for layer1, layer2,c,d in zip(conv_layers_1, conv_layers_2,k1_f,k2_f):
                  if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                      self.transfer_weights_adaptive(layer1, layer2,c,d)
    def verify_transfer(self, level):
        # Verify weights for net_i
        net1_i = self.layers[level-1].van_upsampling.net_i.net
        net2_i = self.layers[level].van_upsampling.net_i.net
        for i, (layer1, layer2) in enumerate(zip(net1_i, net2_i)):
            if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                 print(f"net_i Layer {i}: Weights Equal? {torch.equal(layer1.weight.data, layer2.weight.data)}")

    # Verify weights for net_f
        if (level+1)<self.nlevels:
             net1_f = self.layers[level-1].van_upsampling.net_f.net
             net2_f = self.layers[level].van_upsampling.net_f.net
             for i, (layer1, layer2) in enumerate(zip(net1_f, net2_f)):
                  if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                       print(f"net_f Layer {i}: Weights Equal? {torch.equal(layer1.weight.data, layer2.weight.data)}")

class NeuralVANMultilevel(torch.nn.Module):
    '''
    Base class for multilevel. It trains the full upsampling model in one block, i.e., it performs the backward pass
    through the entire model and trains all at once.
    '''
    def __init__(self, Lc, van_hyp, net_hyp, nlevels, hb_last, energy, local_energy, beta, device):
        super().__init__()
        self.Lc = Lc
        self.Lf = Lc
        self.beta = beta
        self.energy = energy
        self.device = device
        self.nlevels = nlevels #Number of intermediate/fine layers
        device_counts = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        device_ids = [i for i in range(len(device_counts))]
        if len(device_ids) > 0:
            print(device_ids)

        self.in_channels=1

        van=PixelCNN(
            in_channels=self.in_channels,
            L=Lc,
            net_depth=van_hyp['net_depth'],
            net_width=van_hyp['net_width'],
            half_kernel_size=van_hyp['half_kernel_size'],
            bias=van_hyp['bias'],
            z2=van_hyp['z2'],
            res_block=van_hyp['res_block'],
            x_hat_clip=False,
            final_conv=van_hyp['final_conv'],
            epsilon=1.e-7,
            device=device
        )
        self.van = torch.nn.DataParallel(van, device_ids=device_ids).to(device)

        layers = Multilevel(Lc, net_hyp, nlevels, hb_last, local_energy, beta, device)
        self.layers = torch.nn.DataParallel(layers, device_ids=device_ids).to(device)
        self.Lf = self.layers.module.Lf

    def forward(self, bs):
        sample = torch.zeros([bs, 1, self.Lc, self.Lc]).to(self.device)
        sample, log_prob = self.van(sample)
        sample, log_prob = self.layers(sample, log_prob)
        return sample, log_prob.squeeze()

    def reverse(self, sample):
        # does not support multi gpus
        log_prob = torch.zeros([sample.shape[0]]).to(self.device)
        sample, log_prob = self.layers.module.reverse(sample, log_prob)
        dlogq = self.van.module.log_prob(sample)
        return log_prob.squeeze()+dlogq.squeeze()

    def train(
            self,
            nepochs,
            batch_size,
            optimizer,
            scheduler,
            print_freq,
            history_path,
            weights_path,
            on_file=True
    ):     
        scaler = torch.cuda.amp.GradScaler() # this function will be deprecated. When this happens use function below.

        history = {
            'loss': [],
            'varF': [],
            'var_varF': [],
            'betaF': [],
            'ESS': []
        }
        t0 = time.time()
        for i in range(nepochs):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                samples, log_prob = self(batch_size)
                with torch.no_grad():
                    w = self.energy(samples.squeeze(), self.beta)+log_prob
                    ess, betaF = compute_metrics(w)
                loss = torch.mean((w-w.mean()) * log_prob)
            del samples
            scaler.scale(loss).backward()  # Added for mixed precision training
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(w.mean())
            history['loss'].append(grab(loss))
            history['varF'].append(grab(w.mean()))
            history['var_varF'].append(grab(w.var()))
            history['betaF'].append(grab(betaF))
            history['ESS'].append(grab(ess))
            if (i+1) % print_freq == 0:
                if on_file:
                    print_metrics(history_path, history, i+1, print_freq, t0)
                    print(f'step: {i+1},'
                          f' loss: {grab(loss)},'
                          f' w_mean: {grab(w.mean())},'
                          f' w_var: {grab(w.var())},'
                          f' ess: {grab(ess)},'
                          f' free_en: {grab(betaF)},'
                          f' log_prob: {grab(log_prob.mean())}')
                save(self, optimizer, weights_path)
            # print(time.time()-t0,'print')
        history['time'] = time.time()-t0
        return history

    def sample_n(self, nbatch, batch_size):
        W = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(nbatch):
                samples, log_prob = self(batch_size)
                w = self.energy(samples.squeeze(), self.beta)+log_prob
                W.append(grab(w))
        return np.asarray(W).reshape(-1), time.time()-t0

    def w_MCMC(self, sample):
        energy = self.energy(sample.squeeze(), self.beta)
        log_prob = self.reverse(sample)
        w = energy+log_prob
        return w

    def sample_n_OBS(self, nbatch, batch_size):
        W = []
        log_pr = []
        BetaE = []
        m_abs = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(nbatch):
                samples, log_prob = self(batch_size)
                betaE = self.energy(samples.squeeze(), self.beta)
                w = betaE+log_prob
                W.append(grab(w))
                log_pr.append(grab(log_prob))
                BetaE.append(grab(betaE))
                m_abs.append(grab(samples.mean((-1, -2))))
        return np.asarray(W).reshape(-1), np.asarray(BetaE).reshape(-1), np.asarray(m_abs).reshape(-1), time.time()-t0


    def sample_from_MCMC(self, config, nbatch):
        WF=[]
        n, T, R=config.shape
        for i in range(nbatch):
            samples = torch.tensor(config.reshape((nbatch,n//nbatch,T,R))[i]).float() ##Remeber to comment!!!
            wf = self.w_MCMC(samples.unsqueeze(1).to(self.device))
            WF.append(grab(wf))
        return np.asarray(WF).reshape(-1)


class NeuralVANMultilevel_block_wise(NeuralVANMultilevel):
    '''
    Differently from the `NeuralVANMultilevel`, this class sequentially trains the model, i.e., it first trains the
    first block, then the second and so on.
    '''
    def __init__(self, Lc, van_hyp, net_hyp, nlevels, hb_last, energy, local_energy, beta, device):
        super().__init__(Lc, van_hyp, net_hyp, nlevels, hb_last, energy, local_energy, beta, device)
       # self.layers.module.current_level = nlevels-1
       # self.layers.module.freeze_all_layers()

    def eval(self,):
        self.layers.module.current_level = self.nlevels-1
    def verify_transfer(self, level):
         # Verify weights for net_i
         net1_i = self.layers[level-1].van_upsampling.net_i.net
         net2_i = self.layers[level].van_upsampling.net_i.net
         for i, (layer1, layer2) in enumerate(zip(net1_i, net2_i)):
              if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                   print(f"net_i Layer {i}: Weights Equal? {torch.equal(layer1.weight.data, layer2.weight.data)}")
         # Verify weights for net_f
         if level+1<self.nlevel:
             net1_f = self.layers[level-1].van_upsampling.net_f.net
             net2_f = self.layers[level].van_upsampling.net_f.net
             for i, (layer1, layer2) in enumerate(zip(net1_f, net2_f)):
                  if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                       print(f"net_f Layer {i}: Weights Equal? {torch.equal(layer1.weight.data, layer2.weight.data)}")
    def transfer_weights_adaptive(self,source_layer, target_layer,k1,k2):
        # Get kernel sizes
        K_init = k1
        K_final =k2
        # Get the source and target weights
        source_weights = source_layer.weight.data
        target_weights = target_layer.weight.data
        # Initialize target weights to zero
         #target_weights.zero_()

        if K_init == K_final:
             print("Good")
        # If the kernel sizes are the same, copy directly
             target_weights = source_weights.clone()
        elif K_init < K_final:
             print("Not not 1")
         # Case: K_init is smaller, place the weights in the center of the larger kernel
             center_offset = (K_final - K_init) // 2
             target_weights[:, :, center_offset:center_offset + K_init, center_offset:center_offset + K_init] = source_weights
        else:
             print("Not good 2")
             # Case: K_init is larger, extract the central part to fit into the smaller kernel
             center_offset = (K_init - K_final) // 2
             target_weights = source_weights[:, :, center_offset:center_offset + K_final, center_offset:center_offset + K_final].clone()
         # Assign the adapted weights to the target layer
        target_layer.weight.data = target_weights

         # Transfer the bias directly if it exists
        if source_layer.bias is not None and target_layer.bias is not None:
             target_layer.bias.data = source_layer.bias.data.clone()
             print("Is it true :bias")
    def train(
            self,
            nepochs,
            batch_size,
            lr,
            print_freq,
            history_path,
            flex_kernel=True,
            on_file=True,
    ):
        history = {'loss': [], 'varF': [], 'var_varF': [], 'betaF': [], 'ESS': []}
        t0 = time.time()
        scaler = torch.cuda.amp.GradScaler() # this function will be deprecated. When this happens use function below.
        # scaler = torch.amp.GradScaler(self.device)  # Added for mixed precision training
        optimizer = torch.optim.Adam(self.van.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.92,
            patience=5000,
            min_lr=1e-07
        )
        if self.nlevels==0:
            print("Training VAN layers...")
            for i in range(nepochs):
                optimizer.zero_grad()
                with torch.no_grad():
                    samples = torch.zeros([batch_size, 1, self.Lc, self.Lc]).to(self.device) #comment .to(device) in mps implementation
                samples, log_prob = self.van(samples)
                with torch.no_grad():
                    w = self.energy(samples.squeeze(), self.beta) + log_prob
                    ess, betaF = compute_metrics(w)
                loss = torch.mean((w - w.mean()) * log_prob)
                loss.backward()
                optimizer.step()
                scheduler.step(w.mean())
                history['loss'].append(grab(loss))
                history['varF'].append(grab(w.mean()))
                history['var_varF'].append(grab(w.var()))
                history['betaF'].append(grab(betaF))
                history['ESS'].append(grab(ess))
                if (i + 1) % print_freq == 0:
                    print(f'step: {i + 1},'
                          f' loss: {grab(loss)},'
                      f' w_mean: {grab(w.mean())},'
                      f' w_var: {grab(w.var())},'
                      f' ess: {grab(ess)},'
                      f' free_en: {grab(betaF)},'
                      f' log_prob: {grab(log_prob.mean())}')
                    if on_file:
                         print_metrics(history_path, history, i + 1, print_freq, t0)
                with open(history_path, 'a') as f:
                     f.write("Total time taken :"+str(time.time() - t0)+"\n")
            print("Total time taken :", time.time() - t0)
            history['time'] = time.time() - t0

#            torch.save(self.state_dict(), "weights_path")
               #train the rest of the model and save
        if self.nlevels>0:
            if self.nlevels>1:
                level=self.nlevels-1
                self.layers.module.load_diff_kernel(level)  # Transfer weights from previous layers
                self.layers.module.verify_transfer(level)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.parameters())), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.92,
                patience=5000,
                min_lr=1e-07
            )
            print("Training Started at level_",self.nlevels)

            for i in range(nepochs):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(): #this function will be deprecated. If this happens use function below.
                #with torch.amp.autocast(str(self.device)):  # Added for mixed precision training
                    samples, log_prob = self(batch_size)
                    with torch.no_grad():
                        w = self.energy(samples.squeeze(), self.beta) + log_prob
                        ess, betaF = compute_metrics(w)
                    loss = torch.mean((w - w.mean()) * log_prob)
                del samples
                scaler.scale(loss).backward()  # Added for mixed precision training
                scaler.step(optimizer)
                scaler.update()  # Added for mixed precision training# Added for mixed precision training
                scheduler.step(w.mean())
                history['loss'].append(grab(loss))
                history['varF'].append(grab(w.mean()))
                history['var_varF'].append(grab(w.var()))
                history['betaF'].append(grab(betaF))
                history['ESS'].append(grab(ess))
                if (i + 1) % print_freq == 0:
                    if on_file:
                         print_metrics(history_path, history, i + 1, print_freq, t0)
                    print(f'step: {i + 1},'
                          f' loss: {grab(loss)},'
                          f' w_mean: {grab(w.mean())},'
                          f' w_var: {grab(w.var())},'
                          f' ess: {grab(ess)},'
                          f' free_en: {grab(betaF)},'
                          f' log_prob: {grab(log_prob.mean())}')
            with open(history_path, 'a') as f:
                f.write("Total time taken :"+str(time.time() - t0)+"\n")
            print("Total time taken :", time.time() - t0)
            history['time'] = time.time() - t0
        return history
           #   load all block except the last two where you transfer the weight
#        ep = 0
#        tvan = t0
#        for level in range(self.nlevels):
#            print("Time taken for this level now", time.time() - tvan,)
#            with open(history_path, 'a') as f:
#                f.write("\n"+"Time taken for this level now"+str(time.time() - tvan)+"\n")
#                f.write(f"Unfreezing and training up to level {level+1}"+"\n")
#            tvan = time.time()
#            print(f"Unfreezing and training up to level {level+1}")
#            self.layers.module.unfreeze_layers(level)
#            self.layers.module.current_level = level
#            if flex_kernel:
#                if level > 0:
#                    self.layers.module.load_diff_kernel(level)  # Transfer weights from previous layers
#                    self.layers.module.verify_transfer(level)
#            else:
#                 if level>0:
#                     self.layers.module.load_pretrained(level)
#            ep += 1
#            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.parameters())), lr=lr[ep])
#            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                optimizer,
#                'min',
#                factor=0.92,
#                patience=5000,
#                min_lr=1e-07
#            )
#            for i in range(nepochs[ep]):
#                optimizer.zero_grad()
#
#                with torch.cuda.amp.autocast(): #this function will be deprecated. If this happens use function below.
#                #with torch.amp.autocast(str(self.device)):  # Added for mixed precision training
#                    samples, log_prob = self(batch_size[ep])
#                    with torch.no_grad():
#                        w = self.energy(samples.squeeze(), self.beta) + log_prob
#                        ess, betaF = compute_metrics(w)
#                    loss = torch.mean((w - w.mean()) * log_prob)
#                del samples
#                scaler.scale(loss).backward()  # Added for mixed precision training
#                scaler.step(optimizer)
#                scaler.update()  # Added for mixed precision training# Added for mixed precision training
#                scheduler.step(w.mean())
#               history['loss'].append(grab(loss))
#                history['varF'].append(grab(w.mean()))
#                history['var_varF'].append(grab(w.var()))
#                history['betaF'].append(grab(betaF))
#                history['ESS'].append(grab(ess))
#                if (i + 1) % print_freq == 0:
#                    if on_file:
#                        print_metrics(history_path, history, i + 1, print_freq, t0)
#
#                    print(f'step: {i + 1},'
#                          f' loss: {grab(loss)},'
#                          f' w_mean: {grab(w.mean())},'
#                          f' w_var: {grab(w.var())},'
#                          f' ess: {grab(ess)},'
#                          f' free_en: {grab(betaF)},'
#                          f' log_prob: {grab(log_prob.mean())}')
#
       #             if level == self.nlevels-1 and (i+1) > 2999:
       #                 save(self, optimizer, w_path+'_'+str(i+1)+'.chckpnt')
       #             if level == self.nlevels-1 and (i+1) % 1000 == 0:
       #                 save(self, optimizer, w_path+'_'+str(i+1)+'.chckpnt')
       # with open(history_path, 'a') as f:
       #     f.write("Total time taken :"+str(time.time() - t0)+"\n")
       # print("Total time taken :", time.time() - t0)
       # history['time'] = time.time() - t0
       # return history

    def vanilla_training(self,nepochs,batch_size,optimizer,scheduler,print_freq,history_path,weights_path,on_file):
        return super().train(nepochs,batch_size,optimizer,scheduler,print_freq,history_path,weights_path)
