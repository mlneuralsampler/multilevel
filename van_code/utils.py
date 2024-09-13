import time
import torch
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from van_code.obs import Obs

def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker#.to(torch_device)


def grab(var):
    if torch.is_tensor(var):
        return var.detach().cpu().numpy()
    else:
        return var

def save(model,optimizer,path):
    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},path)

def load(model,optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def write(history,root):
    ess_file = root+ '_ESS.dat'
    loss_var_file = root + '_lossvar.dat'
    loss_file = root + '_loss.dat'

    with open(ess_file, 'w') as f:
        for item in history['ESS']:
            f.write("%f\n" % item)

    with open(loss_var_file, 'w') as f:
        for item in history['var_varF']:
            f.write("%f\n" % item)

    with open(loss_file, 'w') as f:
        for item in history['loss']:
            f.write("%f\n" % item)

def compute_metrics_np(w):
    N = w.shape[-1]
    logZ = logsumexp(-w,-1)
    log_ess = 2.0 * logZ - logsumexp(-2 * w,-1)
    ess_per_cfg = np.exp(log_ess) / N
    logZ = logZ - np.log(N)
    return ess_per_cfg, -logZ

def compute_metrics_Zp(w):
    N = w.shape[-1]
    logZ = logsumexp(w,-1)
    log_ess = 2.0 * logZ - logsumexp(2 * w,-1)
    ess_per_cfg = np.exp(log_ess) / N
    logZ = logZ - np.log(N)
    return ess_per_cfg, logZ

def compute_metrics(w):
    N = w.shape[-1]
    logZ = torch.logsumexp(-w,-1)
    log_ess = 2.0 * logZ - torch.logsumexp(-2 * w,-1)
    ess_per_cfg = torch.exp(log_ess) / N
    logZ = logZ - torch.log(torch.tensor(N))
    return ess_per_cfg, -logZ

def print_metrics(history_file,history, epoch, avg_last_N_epochs, t0):
    with open(history_file, 'a') as f:
        f.write(f'\n ==  Epoch {epoch} metrics ==\n')
        for key, val in history.items():
            avgd = np.mean(val[-avg_last_N_epochs:])
            f.write(f'\t{key} {avgd:g}\n')
        f.write(str(time.time()-t0))


def gamma_analysis(work,meas_file):
    #Loss
    obs1=Obs([work], ['ensemble1'])
    loss=np.mean(obs1)
    loss.gamma_method(S=0)
    with open(meas_file, 'a') as ff:
        ff.write(str(loss.value)+' '+str(loss.dvalue)+' '+str(loss.ddvalue)+'\n')
    del loss, obs1
    #betaF
    w0=np.mean(work)
    obs1= Obs([np.exp(-work+w0)], ['ensemble1'])
    betaF=-np.log(np.mean(obs1))+w0
    betaF.gamma_method(S=0)
    with open(meas_file, 'a') as ff:
        ff.write(str(betaF.value)+' '+str(betaF.dvalue)+' '+str(betaF.ddvalue)+'\n')
    del betaF
    #ESS
    obs2= Obs([np.exp(2*(-work+w0))], ['ensemble1'])
    ESS=np.mean(obs1)**2/np.mean(obs2)
    ESS.gamma_method(S=0)
    with open(meas_file, 'a') as ff:
        ff.write(str(ESS.value)+' '+str(ESS.dvalue)+' '+str(ESS.ddvalue)+'\n')
    del ESS, obs2



def gamma_analysis_OBS(observable, work, meas_file=None):
    #Obs : observable computed on generated samples s (1d arrays )
    # work: log\tilde{w}=\betaJ(s)+logq(s) (1d array with the same shape as Obs)
    #meas_file file to save results
    #betaF
    w0=np.mean(work)
    obs1= Obs([np.exp(-work+w0)], ['ensemble1'])
    #General Observable
    obs2= Obs([np.exp(-work+w0)*observable], ['ensemble1'])
    O=obs2/obs1
    O.gamma_method()
    if meas_file:
        with open(meas_file, 'a') as ff:
            ff.write(str(O.value)+' '+str(O.dvalue)+' '+str(O.ddvalue)+'\n')
    return O


def gamma_analysis_modedrop(work, work3, meas_file):
    #Work: log\tilde(w) of generated samples
    #Work 3: log(W) of MCMC samples
    #Loss
    obs1=Obs([work], ['ensemble1'])
    loss=np.mean(obs1)
    loss.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(loss.value)+' '+str(loss.dvalue)+' '+str(loss.ddvalue)+'\n')
    #betaF
    w0=np.mean(work)
    obs1= Obs([np.exp(-work+w0)], ['ensemble1'])
    betaF=-np.log(np.mean(obs1))+w0
    betaF.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(betaF.value)+' '+str(betaF.dvalue)+' '+str(betaF.ddvalue)+'\n')
    #ESS
    obs12= Obs([np.exp(2*(-work+w0))], ['ensemble1'])
    ESS1=np.mean(obs1)**2/np.mean(obs12)
    ESS1.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(ESS1.value)+' '+str(ESS1.dvalue)+' '+str(ESS1.ddvalue)+'\n')
    ####

    #####
    #Work3
    obs3=Obs([work3], ['ensemble2'])
    loss3=np.mean(obs3)
    loss3.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(loss3.value)+' '+str(loss3.dvalue)+' '+str(loss3.ddvalue)+'\n')
    #betaF
    w03=work3.max()
    obs3= Obs([np.exp(work3-w03)], ['ensemble2'])
    betaF3=+np.log(np.mean(obs3))+w03
    betaF3.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(betaF3.value)+' '+str(betaF3.dvalue)+' '+str(betaF3.ddvalue)+'\n')
    #ESS
    obs23= Obs([np.exp(2*(work3-w03))], ['ensemble2'])
    ESS3=np.mean(obs3)**2/np.mean(obs23)
    ESS3.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(ESS3.value)+' '+str(ESS3.dvalue)+' '+str(ESS3.ddvalue)+'\n')


    ###MODED
    modeD=np.exp(betaF3-betaF)
    modeD.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(modeD.value)+' '+str(modeD.dvalue)+' '+str(modeD.ddvalue)+'\n')




def cluster_obs(obs,meas_file):
    m=Obs([obs], ['ensemble1'])
    m=np.mean(m)
    m.gamma_method()
    with open(meas_file, 'a') as ff:
        ff.write(str(m.value)+' '+str(m.dvalue)+' '+str(m.ddvalue)+'\n')
        ff.write(str(np.mean(list(m.e_tauint.values())))+' '+str(np.mean(list(m.e_dtauint.values())))+'\n')

def cluster_analysis(config,energy,beta,meas_file):
    samples=torch.tensor(config)
    E=energy(samples,beta)
    cluster_obs(grab(E),meas_file)
    m=samples.mean((-2,-1))
    cluster_obs(grab(m.abs()),meas_file)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        default_dtype_torch=torch.float32
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = torch.device("cpu")
    return device




def get_data_modedrop(path):
    dict= {'l':[],'|m|':[]}
    df= pd.read_csv(path,on_bad_lines='skip')
    dfnp=np.asarray(df)

    rF=str(dfnp[-8]).strip().split()
    rESS=str(dfnp[-7]).strip().split()

    fF=str(dfnp[-5]).strip().split()
    fESS=str(dfnp[-4]).strip().split()

    md=str(dfnp[-3]).strip().split()

    U=str(dfnp[-2]).strip().split()
    m=str(dfnp[-1]).strip().split()


    rF=get_tree(rF)
    rESS=get_tree(rESS)

    fF=get_tree(fF)
    fESS=get_tree(fESS)

    md=get_tree(md)

    U=get_tree(U)
    m=get_tree(m)
    return [rF,rESS,fF,fESS,md,U,m]


def get_data_MCMC(path):
    dict= {'U':[],'|m|':[]}
    df= pd.read_csv(path,header=None)
    dfnp=np.asarray(df.to_records(index=False))
    U=str(dfnp[-4]).strip().split()
    tU=str(dfnp[-3]).strip().split()
    m=str(dfnp[-2]).strip().split()
    tm=str(dfnp[-1]).strip().split()

    U=get_two(U)
    tU=get_two(tU)
    m=get_two(m)
    tm=get_two(tm)
    return U,tU,m,tm


def get_tree(U):
    return [float(U[0].replace('[','').replace("'",'').replace('(','')),float(U[1].replace("'",'')),float(U[2].replace(']','').replace("'",'').replace(',)',''))]
def get_two(tU):
    return [float(tU[0].replace('[','').replace("'",'').replace('(','')),float(tU[1].replace(']','').replace("'",'').replace(',)',''))]
