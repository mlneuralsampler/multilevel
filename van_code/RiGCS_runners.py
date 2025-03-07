import torch
import numpy as np
import pickle
from van_code.nn_RiGCS import NeuralVANMultilevel,NeuralVANMultilevel_block_wise
from van_code.utils import *
from van_code.ising import phi4_energy, local_ising_energy, ising_energy, analytical_solution
from van_code.montecarlo import *
import sys
import glob
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
    for i in range(6):
        h=4*i
        for j in range(2):
            hidden_size_nn.append(args.hidden_sizes)
            kernel_size_nn.append(args.kernel_size[h:h+2])
            h=h+2
    kernel_size_nn.append(args.kernel_size[-2:])

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
  #  model_prev= torch. load_model(save_weight_full)
    model, path, info, = init(args)
    nepochs=args.train_epoch
    bs=args.train_bs
    lr=args.lr
    info+='\nbatch size '+str(bs)+'\n'
    info+='lr '+str(lr)+'\n'

    save_weight=args.main_path+'RiGCS_training/models/'+'level_'+str(args.n_blocks)+'epochs_'+str(args.train_epoch)+'_'.join(map(str, args.kernel_size))+'_RiGCS_model.pth'
    load_weight=args.main_path+'RiGCS_training/models/'+'level_'+str(args.n_blocks-1)+'epochs_'+str(args.pretrain_epoch)+'_'.join(map(str, args.kernel_size))+'_RiGCS_model.pth'
    history_file_training=args.main_path+'RiGCS_training/history/'+path+'_history.log'
    history_dict=args.main_path+'RiGCS_training/history/'+path+'_historyDict.pkl'
    print_freq=args.print_freq
    with open(history_file_training, 'a') as ff:
       ff.write(info)

    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.92, patience=args.patience, min_lr=1e-07, verbose=True)
    if not args.vanilla_train:
        if args.n_blocks>0:
             saved_state_dict = torch.load(load_weight)
             model.load_state_dict(saved_state_dict, strict=False)  # Load matching layers only
        history=model.train(nepochs,bs,lr,print_freq,history_file_training,True,True)
    else:
        history=model.vanilla_training(args.vanilla_epochs,args.vanilla_bs,optimizer,scheduler,print_freq,history_file_training,weights_path,True)

    with open(history_dict, 'wb') as f:
        pickle.dump(history, f)
    torch.save(model.state_dict(), save_weight)

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
    with torch.no_grad():
         w,E,m,t =model.sample_n_OBS(nmeas,bs)

    with open(measures_path, 'a') as f:
        f.write(str(t)+'\n')

    gamma_analysis(w,measures_path)
   # _=gamma_analysis_OBS(E,w,measures_path) #Internal Energy
   # _=gamma_analysis_OBS(np.abs(m),w,measures_path) #Absolute |m|

# Load the data from the file and prepare for batching
def load_dat_in_batches(file_path, batch_sz,L):
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            # Convert space-separated values to a numpy array
            data = np.fromstring(line.strip(), sep=' ')
           # batch.append(data)
            if len(data) == L**2:
                batch.append(data)
            #else:
             #   print(f"Skipping malformed line in {file_path}: {line.strip()}")

            if len(batch) == batch_sz:
                # Convert to PyTorch tensor and move to GPU
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to('cuda')
                yield batch_tensor  # Yield a batch for processing
                batch = []  # Reset for next batch

        # Process remaining data if any
        if batch:
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to('cuda')
            yield batch_tensor
def process_multiple_dat_files(model,dat_files, batch_sz, L,bs,beta,nmeas):
    W=[]
    Wf=[]
    E=[]
    M=[]
    M_clus=[]
    E_clus=[]
    for file_path in dat_files:
        print(f"Processing file: {file_path}")
        for data in load_dat_in_batches(file_path, batch_sz,L):
            data=torch.tensor(data).reshape(-1,L,L)
            #print(data.shape)
            with torch.no_grad():
                e_clus=ising_energy(data,beta)
                m_clus=data.mean((-2,-1))
           # print(type(bs),type(model.L)
                w,e,m,t =model.sample_n_OBS(nmeas,bs)
            #print("Time:",t)
                wf=model.sample_from_MCMC(data,nmeas)
            Wf.append(grab(wf))
            W.append(grab(w))
            M.append(grab(m))
            E.append(grab(e))
            M_clus.append(grab(m_clus))
            E_clus.append(grab(e_clus))
    print("W shape",len(W))
   # print("After 2nd",t)
    return np.array(Wf).flatten(),np.array(W).flatten(),np.array(M).flatten(),np.array(E).flatten(),np.array(M_clus).flatten(), np.array(E_clus).flatten(),t
 # Perform inference on the batch
def measures_modedrop_old(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas
    print("Lf:", model.Lf)
    info+='\nbatch size '+str(bs)+' nmeas '+str(nmeas)+'\n'
    measures_path=main_path+'results/'+path+'_measures_modedrop.log'
    clstr_path=main_path+'results/'+path+'_measuresCluster.log'
    print("Model Weight path:" ,weights_path)
   # path_ising=main_path+'config/Ising_data_nx{}_beta0.4400000000_data{}.dat'.format(model.Lf,args.data_cluster) #Fixed beta
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    load(model,optimizer,weights_path)
    model.eval()
    data_path = main_path+'config/{}'.format(model.Lf)  # Update this to your directory

     # Use glob to find all .dat files in the directory
    data_files = glob.glob(f'{data_path}/*.dat')
    Wf,W, M,E,M_clus, E_clus, t=process_multiple_dat_files(model,data_files, args.batch_sz,model.Lf,bs,args.beta,nmeas)
    #Cluster
    #data=np.genfromtxt(path_ising).reshape(-1,model.Lf,model.Lf)
    #cluster_analysis(data,ising_energy,args.beta,clstr_path)
    print(np.array(Wf).shape, type(W), type(M), np.array(E).shape, np.array(M_clus).shape, np.array(E_clus).shape)
    sys.stdout.flush()

    with open(measures_path, 'a') as f:
        f.write(info)

    cluster_obs(E_clus,clstr_path)
    cluster_obs(np.abs(M_clus),clstr_path)

    with open(measures_path, 'a') as f:
        f.write(str(t)+'\n')

    gamma_analysis_modedrop(W,Wf,measures_path)
    _=gamma_analysis_OBS(E,W,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(M),W,measures_path) #Absolute |m|

def measures_modedrop(args):
    model, path, info = init(args)
    bs=args.bs_eval
    nmeas=args.nmeas
    print("Lf:", model.Lf)
    info+='\nbatch size '+str(bs)+' nmeas '+str(nmeas)+'\n'
    main_path=args.main_path+'data/'
    load_weight=args.main_path+'RiGCS_training/models/'+'level_'+str(args.n_blocks)+'epochs_'+str(args.train_epoch)+'_'.join(map(str, args.kernel_size))+'_RiGCS_model.pth'
    measures_path=args.main_path+'RiGCS_inference/'+path+'_measures_modedrop.log'
    clstr_path=args.main_path+'RiGCS_inference/'+path+'_measuresCluster.log'
    print("Model Weight path:" ,load_weight)

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    saved_state_dict = torch.load(load_weight)
    model.load_state_dict(saved_state_dict, strict=True)
   # load(model,optimizer,weights_path)
    model.eval()
    data_path = main_path+'config/{}'.format(model.Lf)
    data_files = glob.glob(f'{data_path}/*.dat')
    Wf,W, M,E,M_clus, E_clus,t=process_multiple_dat_files(model,data_files,args.mc_bs,model.Lf,bs,args.beta,nmeas)
    print(np.array(Wf).shape, type(W), type(M), np.array(E).shape, np.array(M_clus).shape, np.array(E_clus).shape)
    sys.stdout.flush()

    with open(measures_path, 'a') as f:
        f.write(info)

    cluster_obs(E_clus,clstr_path)
    cluster_obs(np.abs(M_clus),clstr_path)

    with open(measures_path, 'a') as f:
        f.write(str(t)+'\n')

    gamma_analysis_modedrop(W,Wf,measures_path)
    _=gamma_analysis_OBS(E,W,measures_path) #Internal Energy
    _=gamma_analysis_OBS(np.abs(M),W,measures_path) #Absolute |m|



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
