from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd
import json
from scipy import sparse
import sys,math,time
from numpy import linalg as LA

@dataclass
class CustomParameters:
    random_state: int    = 123
    alpha: float         = 0.01
    beta: float          = 0.01
    lamda : float        = 0.1
    s: int               = 20
    maxIt: int           = 10
    h: float             = 0.5
    w : float            = 1



class AlgorithmArgs(argparse.Namespace):
    # from TimeEval 1.2.10  
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)

def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    return df.iloc[:, 1:-1].values
# temporal dependency model S(i,j)
def S(lamda,w,obs_i,obs_j):
    diff   = np.abs(obs_i-obs_j)
    period = np.power(np.cos(w * diff),2)
    decay  = np.power(np.e, (- lamda * diff))
    return period * decay

# construct Z by eign decomp K
def getZ(Wi,h,lamda,w):
    n_obs = Wi.shape[0]
    K=np.fromfunction(lambda i, j: S(lamda,w,i,j), (n_obs, n_obs))
    D,V = LA.eig(K)
    Z=V[:,0:h]
    return Z
# detector for a single window Wi
def detector(Wi,Z,Ri,Pi,D_r,D_p,alpha,beta,maxIt):

    # get shape
    n=Wi.shape[0]
    p=Wi.shape[1]
    h=Wi.shape[1]

    # Initialisation
    D_p = np.eye(p)
    D_r = np.eye(n)
    Iw =np.eye(p)
    Ir = np.eye(n)
    t=0
    loss=np.zeros(maxIt)

    # start timing
    start_time=time.time()
    while t < maxIt:
        # update Pi and Ri
        Pi = LA.pinv((Wi.T@Wi)+alpha * D_p)  @ (Wi.T @(Ri.T+Z))
        Ri = (Wi@Pi - Z).T @ (LA.pinv( Ir+beta*D_r )).T
        # update D_p and D_r
        D_p =  sparse.eye(p)* 0.5 * np.apply_along_axis(np.linalg.norm,1,Pi)
        D_r = sparse.eye(n)* 0.5 * np.apply_along_axis(np.linalg.norm,1,Ri.T)
        # compute loss
        l_21_Pi=np.sum(np.apply_along_axis(np.linalg.norm, 1, Pi))
        l_21_Ri=np.sum(np.apply_along_axis(np.linalg.norm, 1, Ri))
        loss[t]= LA.norm(Wi@Pi-Ri.T-Z, 'fro') + alpha * l_21_Pi+ beta*l_21_Ri
        t=t+1

    tmp = time.time()-start_time
    return {'Pi':Pi,'Ri':Ri,'loss':loss,'time':tmp,'maxIt':maxIt,'alpha':alpha,'beta':beta,'D_r':D_r,'D_p':D_p}

def execute(args: AlgorithmArgs):
    # get hyper-parameters
    alpha        = args.customParameters.alpha
    beta         = args.customParameters.beta
    s            = args.customParameters.s
    h            = args.customParameters.h
    w            = args.customParameters.w
    lamda        = args.customParameters.lamda
    maxIt        = args.customParameters.maxIt

    # get data
    T = load_data(args)
    print(T.shape)
    # create windows with stride=1 from T
    W_win = T[np.arange(s)[None, :] + np.arange(T.shape[0]-s+1)[:, None]]
    Pi=np.random.rand(T.shape[1],math.ceil(h * W_win[0].shape[1]))
    D_p = np.eye(T.shape[1])
    D_r = np.eye(s)
    Ri = np.random.rand(math.ceil(h * W_win[0].shape[1]),s)
    # anomaly scores sequences
    Scores = np.zeros([T.shape[0]-s+1,s])
    index = 0
    Z = getZ(W_win[0],math.ceil(h * W_win[0].shape[1]),lamda,w)
    for Wi in W_win :
        if (index % 500):
            print("window : ",index, " from ",T.shape[0]-s+1," windows")
        results =  detector(Wi,Z,Ri,Pi,D_r,D_p,alpha=alpha,beta=beta,maxIt=10)
        Scores[index,:] = np.apply_along_axis(np.linalg.norm,1,results['Ri'].T)
        B = np.random.rand(math.ceil(h * W_win[0].shape[1]),s)
        W=np.random.rand(T.shape[1],math.ceil(h * W_win[0].shape[1]))
        D_p = np.eye(T.shape[1])
        D_r = np.eye(s)
        index =index+1

    Scores.tofile(args.dataOutput, sep="\n")

if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
