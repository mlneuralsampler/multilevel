import torch
import numpy as np
import pickle
from van_code.nn import NeuralVANMultilevel,NeuralVANMultilevel_block_wise
from van_code.utils import *
from van_code.ising import local_ising_energy, ising_energy, analytical_solution
from van_code.montecarlo import *


def init(args):
    device=get_device()
    info='Device '+str(device)
    if device=='cuda':
        info+=' '+torch.cuda.get_device_name(torch.cuda.current_device())+' '+'x'+str(torch.cuda.device_count())
    info+='\n'

    ##Theory
    Lc=args.Lc
    beta=args.beta
    ##net
    net_depth = args.net_depth
    net_width= args.net_width
    half_kernel_size = args.half_kernel_size
    bias=args.bias
    z2=args.not_z2
    residual=args.not_residual
    x_hat_clip=args.x_hat_clip
    eps=args.eps

    nlevels=args.n_blocks
    #if not args.not_hb_last:
    hb_last=args.not_hb_last
    hidden_size_nn=[]
    kernel_size_nn=[]
    for i in range(nlevels):
        for j in range(2):
            hidden_size_nn.append(args.hidden_sizes)
            kernel_size_nn.append(args.kernel_size)

    van_hyp={'net_depth':net_depth,'net_width': net_width,'half_kernel_size':half_kernel_size,'bias':bias,'z2':z2,'res_block':residual,'x_hat_clip':x_hat_clip,'final_conv':True,'epsilon':eps,'device':device}
    net_hyp={'hidden_size':hidden_size_nn,'kernel_size':kernel_size_nn,'epsilon':eps,'level':0,'device':device}

    model =NeuralVANMultilevel_block_wise(Lc, van_hyp, net_hyp, nlevels, hb_last,ising_energy, local_ising_energy, beta, device)

    Lf=model.Lf

    main_path=args.main_path+'data/'
    path='Lf'+str(Lf)+'_beta'+str(beta)+'_nblocks'+str(nlevels)


    info+= 'Lf '+str(Lf)
    info+= ' beta '+str(beta)+'\n'
    info+= 'nblocks '+str(nlevels)

    path+='_PCNNdepth'+str(net_depth)+'_width'+str(net_width)+'_half_ker'+str(half_kernel_size)+'_CCNNhs' + '_'.join(map(str, args.hidden_sizes))+'_ks' + '_'.join(map(str, args.kernel_size))

    hyp=args.hyp
    if hyp != 0:
        path+='_hyp'+str(hyp)

    return  model, path, info

def training(args):
    model, path, info = init(args)
    nepochs=args.epochs
    bs=args.bs
    lr=[args.lr]*(args.n_blocks+1)
    info+='\nbatch size '+str(bs)+'\n'
    info+='lr '+str(lr)+'\n'

    main_path=args.main_path+'data/'
    lc_path=main_path+'training/'+path
    weights_path=main_path+'model/'+path+'.chckpnt'
    history_file_training=lc_path+'_history.log'
    history_dict=lc_path+'_historyDict.pkl'
    print_freq=args.print_freq
    with open(history_file_training, 'a') as ff:
       ff.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.92, patience=args.patience, min_lr=1e-07, verbose=True)

    if not args.vanilla_train:
        history=model.train(nepochs,bs,lr,print_freq,history_file_training,main_path+'model/'+path,True,True)
    else:
        history=model.vanilla_training(args.vanilla_epochs,args.vanilla_bs,optimizer,scheduler,print_freq,history_file_training,weights_path,True)

    with open(history_dict, 'wb') as f:
        pickle.dump(history, f)
    save(model,optimizer,weights_path)


def measures(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='\nbatch size '+str(bs)+' nmeas '+str(nmeas)+'\n'

    main_path=args.main_path+'data/'
    weights_path=main_path+'model/'+path+'.chckpnt'
    measures_path=main_path+'results/'+path+'_measures'+'.log'
    with open(measures_path, 'a') as f:
        f.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)
    model.eval()
    w,E,m,t =model.sample_n_OBS(nmeas,bs)

    with open(measures_path, 'a') as f:
        f.write(str(t)+'\n')

    gamma_analysis(w,measures_path)
    _=gamma_analysis_OBS(E,w,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(m),w,measures_path) #Absolute |m|


def measures_modedrop(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='\nbatch size '+str(bs)+' nmeas '+str(nmeas)+'\n'

    main_path=args.main_path+'data/'
    weights_path=main_path+'model/'+path+'.chckpnt'
    measures_path=main_path+'results/'+path+'_measures_modedrop.log'
    clstr_path=main_path+'results/'+path+'_measuresCluster.log'
    path_ising=main_path+'config/Ising_data_nx{}_beta0.4400000000_data{}.dat'.format(model.Lf,args.data_cluster) #Fixed beta

    #Cluster
    data=np.genfromtxt(path_ising).reshape(-1,model.Lf,model.Lf)
    cluster_analysis(data,ising_energy,args.beta,clstr_path)

    with open(measures_path, 'a') as f:
        f.write(info)
    #load model
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)
    model.eval()
    #reverse
    w,E,m,t =model.sample_n_OBS(nmeas,bs)
    wf=model.sample_from_MCMC(data,nmeas)

    with open(measures_path, 'a') as f:
        f.write(str(t)+'\n')

    gamma_analysis_modedrop(w,wf,measures_path)
    _=gamma_analysis_OBS(E,w,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(m),w,measures_path) #Absolute |m|


def measures_IMH(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas

    info+='\nbatch size '+str(bs)+' nmeas '+str(nmeas)+'\n'

    main_path=args.main_path+'data/'
    weights_path=main_path+'model/'+path+'.chckpnt'
    measures_path=main_path+'results/'+path+'_measuresIMH'+'.log'
    with open(measures_path, 'a') as f:
        f.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)
    model.eval()
    with torch.no_grad():
        ensemble=make_mcmc_ensemble(model, bs,args.data_cluster,model.device)
    with open(measures_path, 'a') as f:
        f.write(str(ensemble['time'])+'\n')
    cluster_analysis(np.asarray(ensemble['x']).reshape((-1,model.Lf,model.Lf) ),model.energy,args.beta,measures_path)
