import argparse
import random
import pickle
from van_code.van import PixelCNN, train_van, sample_van, sample_forward,make_mcmc_ensemble
from van_code.utils import *
from van_code.ising import ising_energy

parser = argparse.ArgumentParser()
#### Theory
parser.add_argument("--Nt", type=int, default=8)
parser.add_argument("--Ns", type=int, default=8)
parser.add_argument("--beta", type=float, default=0.44)


##### PixelCNN
parser.add_argument("--net_depth", type=int, default=6)
parser.add_argument("--net_width", type=int, default=32)
parser.add_argument("--half_kernel_size", type=int, default=12)
parser.add_argument("--bias", action="store_true")
parser.add_argument("--not_z2", action="store_false")
parser.add_argument("--not_residual", action="store_false")
parser.add_argument("--x_hat_clip", action="store_true")
parser.add_argument("--eps", type=float, default=1.e-7)


### training
parser.add_argument("--train", action="store_true")
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--bs", type=int, default=100)
parser.add_argument("--lr",type=float, default=0.0001 )
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--loss",type=str, default="reinforce")

#### Sampling
parser.add_argument("--measures", action="store_true")
parser.add_argument("--measures_md", action="store_true")
parser.add_argument("--measures_imh", action="store_true")
parser.add_argument("--bs_eval", type=int, default=1000)
parser.add_argument("--nmeas", type=int, default=1000)
parser.add_argument("--data_cluster", type=int, default=1000000)
## Paths
parser.add_argument("--main_path",type=str, default="/leonardo_work/INF24_sft_1/ecellini" )
parser.add_argument("--base_path",type=str, default="/multilevelRG/data/van" )


parser.add_argument("--hyp",type=int, default=0 )
parser.add_argument("--seed", type=int, default=137)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'


def init(args):
    ##Theory
    Nt=args.Nt
    Ns=args.Ns
    lattice_shape=(Nt,Ns)
    beta=args.beta
    path=args.main_path+args.base_path+'/Nt'+str(Nt)+'_Ns'+str(Ns)+'_beta'+str(beta)
    info = 'Nt '+str(Nt)+' Ns '+str(Ns)+'\n'
    info+= 'beta '+str(beta)+'\n'
    ##net
    net_depth = args.net_depth
    net_width= args.net_width
    half_kernel_size = args.half_kernel_size
    bias=args.bias
    z2=args.not_z2
    residual=args.not_residual
    x_hat_clip=args.x_hat_clip
    eps=args.eps

    net=PixelCNN(L=Nt,net_depth=net_depth, net_width = net_width, half_kernel_size = half_kernel_size,bias=bias,z2=z2,res_block=residual,x_hat_clip=x_hat_clip,final_conv=True,epsilon=eps,device=device).to(device)

    path+='_depth'+str(net_depth)+'_width'+str(net_width)+'_half_ker'+str(half_kernel_size)

    hyp=args.hyp
    if hyp != 0:
        path+='_hyp'+str(hyp)

    return  net, path, info


def training(args):
    model, path, info = init(args)
    epochs=args.epochs
    bs=args.bs
    lr=args.lr
    info+='batch size '+str(bs)+'\n'
    info+=' lr '+str(lr)+'\n'


    weights_path=path+'.chckpnt'
    history_file_training=path+'_history.log'
    history_dict=path+'_historyDict.pkl'
    print_freq=args.print_freq
    with open(history_file_training, 'a') as f:
        f.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    history=train_van(model,ising_energy,args.beta,epochs,bs,optimizer,print_freq,history_file_training)
    with open(history_dict, 'wb') as f:
        pickle.dump(history, f)
    save(model,optimizer,weights_path)


def measures(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='batch size '+str(bs)+'n meas '+str(nmeas)+'\n'

    weights_path=path+'.chckpnt'
    measures_path=path+'_measures'+'.log'
    with open(measures_path, 'a') as f:
        f.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)

    w,E,m,time =sample_van(model,ising_energy,args.beta, bs,nmeas)

    with open(measures_path, 'a') as f:
        f.write(str(time)+'\n')

    gamma_analysis(w,measures_path)
    _=gamma_analysis_OBS(E,w,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(m),w,measures_path) #Absolute |m|


def measures_modedrop(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='batch size '+str(bs)+'n meas '+str(nmeas)+'\n'

    weights_path=path+'.chckpnt'
    measures_path=path+'_measures_modedrop'+'.log'
    with open(measures_path, 'a') as f:
        f.write(info)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)

    w,E,m,time =sample_van(model,ising_energy,args.beta, bs,nmeas)
    ising_path=args.main_path+'/multilevelRG/data/config'+'/Ising_data_nx{}_beta0.4400000000_data{}.dat'.format(model.L,args.data_cluster)
    data=np.genfromtxt(ising_path).reshape(-1,model.L,model.L)
    wf,timef=sample_forward(model,ising_energy,args.beta,data,bs,device)
    print(wf.shape)
    with open(measures_path, 'a') as f:
        f.write(str(time)+'\n')
        f.write(str(timef)+'\n')

    gamma_analysis_modedrop(w,wf,measures_path)
    _=gamma_analysis_OBS(E,w,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(m),w,measures_path) #Absolute |m|


def measures_NMCMC(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='batch size '+str(bs)+'n meas '+str(nmeas)+'\n'

    weights_path=path+'.chckpnt'
    measures_path=path+'_measures_nmcmc'+'.log'
    with open(measures_path, 'a') as f:
        f.write(info)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)
    model.energy=ising_energy
    model.beta=args.beta
    with torch.no_grad():
        ensemble=make_mcmc_ensemble(model, bs,args.data_cluster,model.device)
    cluster_analysis(np.asarray(ensemble['x']).reshape((-1,model.L,model.L) ),ising_energy,args.beta,measures_path)


def main(args):
    if args.train:
        training(args)

    if args.measures:
        measures(args)

    if args.measures_md:
        measures_modedrop(args)

    if args.measures_imh:
        measures_NMCMC(args)


if __name__ == "__main__":
    main(args)
