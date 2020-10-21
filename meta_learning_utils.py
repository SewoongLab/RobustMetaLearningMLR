import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse.linalg import svds
from tqdm import tqdm
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram

norm = np.linalg.norm

from matplotlib import rc
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'weight':'bold','size':16})
rc('text', usetex=True)
fontSpecLarge = {'fontsize': 20,}
fontSpecMedium = {'fontsize': 15,}

plt.rcParams['font.family'] = 'serif'

##############################
# Functions for prior settings
##############################

def get_W(k,d,init_type="iid_normal"):
    if init_type == "iid_normal":
        return np.random.normal(0,1./np.sqrt(d),(k,d))
    elif init_type == "random_orth":
        Z = np.random.normal(0,1,(d,d))
        u, ss, vt = svds(Z, k=k, which='LM', return_singular_vectors="u")
        return u.T

def get_p(k,init_type="uniform"):
    if init_type == "uniform":
        p = np.ones(k); p /= norm(p,1)
        return p

def get_s(k,init_type="uniform"):
    if init_type == "uniform":
        return np.ones(k)

def get_M(W,p):
    k, d = W.shape
    M = np.zeros((d,d))
    for i in range(k):
        M += p[i]*np.outer(W[i],W[i])
    return M

def calculate_Delta(W):
    k, d = W.shape
    Z = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i!=j:
                Z[i,j] = norm(W[i]-W[j])
            else:
                Z[i,j] = np.inf
    return np.min(Z)

##############################
# Functions to generate data
##############################

def get_examples_from_id(w_id,W,s,t):
    X = np.random.normal(0,1,(t,d))
    y = np.dot(X,W[w_id]) + np.random.normal(0,s[w_id],t)
    return X, y

def get_examples_random_task(W,s,p,t):
    k = W.shape[0]
    w_id = np.random.choice(k,size=1,replace=True,p=p)[0]
    X, y = get_examples_from_id(w_id,W,s,t)
    return w_id, X, y

def get_random_ids(p,n):
    return np.random.choice(k,size=n,replace=True,p=p)

def get_random_meta_data(W,s,p,n,t):
    return map(list, zip(*[get_examples_random_task(W,s,p,t) for i in range(n)]))

def beta_hat_from_X_y(X,y):
    return np.dot(X.T,y)/X.shape[0]

def beta_hat_from_id(w_id,W,s,t):
    X, y = get_examples_from_id(w_id,W,s,t)
    return np.dot(X.T,y)/t

def beta_hats_from_ids(w_ids,W,s,t):
    return [beta_hat_from_id(w_id,W,s,t) for w_id in w_ids]

##############################
# Functions for subspace estimation
##############################

def get_Mhat(w_ids,W,s,t):
    n = len(w_ids)
    k, d = W.shape
    beta_hat_1 = np.array(beta_hats_from_ids(w_ids,W,s,t))
    beta_hat_2 = np.array(beta_hats_from_ids(w_ids,W,s,t))
    Mhat = np.dot(beta_hat_1.T,beta_hat_2)/n
    Mhat = (Mhat + Mhat.T)/2
    return Mhat

def compute_subspace(Mhat,k):
    u, ss, vt = svds(Mhat, k=k, which='LM', return_singular_vectors="u")
    return u[:,np.flip(np.argsort(ss))]

def evaluate_subspace(W,U,rho):
    k, d = W.shape
    z = np.zeros(k)
    for l in range(k):
        z[l] = norm(np.dot(U,np.dot(U.T,W[l])) - W[l])/rho
    return np.max(z)

##############################
# Functions for clustering
##############################

# can be improved using block matrix multiplication
def get_H(w_ids_n1,U,W,s,t,L,projection=True):
    n1 = len(w_ids_n1)
    HH = np.zeros((L,n1,n1))
    for l in tqdm(range(L)):
        betas_1 = np.array(beta_hats_from_ids(w_ids_n1,W,s,t))
        betas_2 = np.array(beta_hats_from_ids(w_ids_n1,W,s,t))
        if projection:
            UTb_1 = np.dot(U.T,betas_1.T).T # every row is a U.T*beta_i
            UTb_2 = np.dot(U.T,betas_2.T).T # every row is a U.T*beta_i
        else:
            UTb_1 = betas_1
            UTb_2 = betas_2
        v = np.sum(UTb_1*UTb_2,axis=1)
        HH[l] -= np.dot(UTb_1,UTb_2.T)
        HH[l] += v
        HH[l] += HH[l].T
    if L>1:
        H = np.median(HH,axis=0)
    else:
        H = HH[0]
    np.fill_diagonal(H, 0, wrap=False)
    return H, UTb_1, UTb_2

def mode(a):
    if a.shape[0]==0:
        return None
    (values,counts) = np.unique(a,return_counts=True)
    return values[np.argmax(counts)]

def get_cluster_mean(X,C,k):
    X_mean = []; X_hist = []
    for i in range(k):
        X_mean.append(np.mean(X[C==i],axis=0))
        X_hist.append(np.sum(C==i))
    return np.array(X_mean), np.array(X_hist)

def get_cluster_id_from_centers(x,X_means,k):
    return np.argmin(norm(X_means-x,axis=1))

def clustering(t,w_ids,L,U,W,s,projection=True,plot=""):
    d, k = U.shape
    H, UTb_1, UTb_2 = get_H(w_ids,U,W,s,t,L,projection); np.fill_diagonal(H, 0, wrap=False)
    if len(plot):
        plt.hist(np.reshape(H,-1),100,range=(-1,5)); plt.savefig(plot); plt.close()
    Z = linkage(ssd.squareform(np.abs(H)), method="average")
    C_1 = fcluster(Z, k, criterion='maxclust')-1
    clustering_acc = clustering_accuracy(C_1,w_ids,k)
    return clustering_acc, C_1, UTb_1, UTb_2, H

##############################
# Functions to evaluate estimators
##############################

def clustering_accuracy(C,w_ids,k):
    class_acc = []; modes = []
    for l in range(k):
        assigned_ids = C[w_ids==l]
        mode_id = mode(assigned_ids); modes.append(mode_id)
        if mode_id==None:
            print("One cluster wasn't even assigned")
            return 0.
        mask = np.ones_like(assigned_ids); mask[assigned_ids!=mode_id] = 0.
        class_acc.append(np.mean(mask))
    if len(np.unique(modes)) != k:
        print("Detected number of clusters = %d/%d. " % (len(np.unique(modes)),k))
        return 0.
    else:
        clustering_acc = np.mean(class_acc)
        return clustering_acc

def get_maps(C,w_ids,k):
    cluster_map = np.zeros(k,dtype=int) # original to new
    original_map = np.zeros(k,dtype=int) # new to original
    # l is original id
    for l in range(k):
        assigned_ids = C[w_ids==l]
        mode_id = mode(assigned_ids)
        cluster_map[l] = mode_id
        original_map[mode_id] = l
    return cluster_map, original_map

def W_estimation_error(W,W_hat,cluster_map):
    z = []
    # iterate over original ids
    for i in range(len(cluster_map)):
        z.append(norm(W[i]-W_hat[cluster_map[i]]))
    return np.array(z)

def p_estimation_error(p,p_hat,cluster_map):
    z = np.abs(p-p_hat[cluster_map])
    return np.max(z)

def get_r2(W,W_hat,s,original_map):
    return s[original_map]**2 + norm(W[original_map]-W_hat,axis=1)**2

##############################
# Functions for classification
##############################

def get_l(t,k,r2_hat,W_hat,X,y):
    z = np.zeros(k)
    for i in range(k):        
        z[i] = norm(y - np.dot(X,W_hat[i]))**2
    l = -z/(2*r2_hat**2) + t*np.log(1./np.sqrt(r2_hat))
    return l

def classification_accuracy(C,w_ids,cluster_map,k):
    z = np.zeros(k)
    # iterate over original ids
    for l in range(k):
        correct = C[w_ids==l]==cluster_map[l]
        if np.sum(correct)<1:
            return 0
        z[l] = np.mean(correct)
    return z

##############################
# Functions for prediction
##############################

def get_l_pred(t,k,s2_hat,p_hat,W_hat,X,y):
    z = np.zeros(k)
    for i in range(k):
        z[i] = norm(y - np.dot(X,W_hat[i]))**2
    l = -z/(2*r2_hat**2) + t*np.log(1./np.sqrt(s2_hat)) + np.log(p_hat)
    return l

