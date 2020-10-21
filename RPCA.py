import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.sparse.linalg import svds

norm = np.linalg.norm

from matplotlib import rc
rc('text', usetex=True)
font = {'family' : 'serif', 'weight' : 'bold', 'size'   : 18}
rc('font', **font)
fontSpecLarge = {'fontsize': 18,}
fontSpecMedium = {'fontsize': 12,}

from matplotlib.ticker import FormatStrFormatter
##########################################

def generate_bad_example(n,d):
    X = np.random.normal(0,1,(n,d))
    X[:,0] *= np.sqrt(1.1)
    X[:,1] = np.sign(np.random.random(n)-0.5)*X[:,0]/np.sqrt(1.1)
    return X

def add_perturbation_to_bad_example(X,alpha):
    n,d = X.shape
    Z = X.copy()
    H = np.zeros(n,dtype=bool)
    for i in range(n):
        if np.random.random(1)<alpha:
            Z[i,0] = 0; Z[i,1] = np.sign(Z[i,1])*2/alpha**(1./4); H[i] = True
    return Z,H

# use alpha for gaussian, and 2*alpha for heavy tail
def first_filter(z,S0,alpha):
    lower_lim = np.percentile(z[S0], alpha * 100)
    upper_lim = np.percentile(z[S0], (1 - alpha) * 100)
    return (z > lower_lim) & (z < upper_lim)

# modified double_filter choosing the best subspace
def double_filter(X,S,k,alpha,nu):
    n,d = X[S].shape
    S0 = S.copy() # S0 = S_{t-1}
    _, _, VT = svds(X[S0], k=k, which='LM', return_singular_vectors="v"); U0 = VT.T
    z = norm(np.dot(U0.T,X.T),axis=0)**2
    SG = first_filter(z,S0,alpha); SG = SG*S0 # SG \subseteq S_{t-1}
    mu_S0 = z[S0].mean(); mu_SG = z[SG].mean()
    Z = np.random.random()
    W = Z*np.max((z-mu_SG)[S0*(~SG)])
    S1 = SG|(((z-mu_SG)<W)*(S0*(~SG)))
    return S1, mu_SG

# S is a bit mask of size len(X), on which double_filtering will run
def double_filter(X,S,k,alpha,nu):
    n,d = X[S].shape
    S0 = S.copy()
    _, _, VT = svds(X[S0], k=k, which='LM', return_singular_vectors="v"); U0 = VT.T
    z = norm(np.dot(U0.T,X.T),axis=0)**2
    SG = first_filter(z,S0,alpha); SG = SG*S0
    mu_S0 = z[S0].mean(); mu_SG = z[SG].mean()
    if mu_S0-mu_SG <= 2*alpha*np.log(1/alpha): # the constant (48) probably needs to be tuned correctly depending on the desired error
        return S, mu_SG
    else:
        Z = np.random.random()
        W = Z*np.max((z-mu_SG)[S0*(~SG)])
        S1 = SG|(((z-mu_SG)<W)*(S0*(~SG)))
        return S1, mu_SG

def robust_subspace_estimation(X,alpha,delta,k,nu):
    n,d = X.shape
    S = []
    S0 = np.ones(n,dtype=bool); S.append(S0)
    S_max = np.zeros(n,dtype=bool)
    # for l in range(int(np.log(2/delta)/np.log(6))+1):
    for l in range(1):
        t = 0; opt = 0
        while t<int(np.ceil(9*alpha*n)+1):
            if t>0 and (S[t]!=S[t-1]).sum()==0:
                break
            t += 1
            S_new, mu_SG = double_filter(X,S[t-1],k,alpha,nu)
            S.append(S_new.copy());
        if S_max.sum()<S[t].sum():
            S_max = S[t].copy()
    _, _, VT = svds(X[S_max], k=k, which='LM', return_singular_vectors="v"); U_hat = VT.T
    return U_hat, S_max

def V_bar(w,Y,t):
    n,d = Y.shape
    v = np.abs(np.dot(Y,w))
    threshold = np.percentile(v,t*100.0/n)
    return np.mean(v[v<threshold]**2)

def HRPCA(Y,k,T,t):
    Y_hat = Y.copy()
    s = 0; opt = 0; n,d = Y.shape
    S = np.ones(n,dtype=bool); removed_indexes = []; s_max = 0;
    while s<T:
        Sigma_hat = np.dot(Y[S].T,Y[S])/S.sum()
        _, _, VT = svds(Sigma_hat, k=k, which='LM', return_singular_vectors="v"); W = VT.T
        v_bar_sum = 0;
        for i in range(k):
            v_bar_sum += V_bar(W[:,i],Y,t)
        if v_bar_sum>opt:
            opt = v_bar_sum; W_bar = W.copy(); s_max = s;
        Q = np.dot(Y,W); p = np.sum(Q**2,axis=1); p[~S] = 0; p/=p.sum()
        removed_indexes.append(np.random.choice(n,1,p=p)[0])
        S[removed_indexes[-1]] = False;
        s += 1
    return W_bar, removed_indexes[:s_max]


d = 16
k = 1
alpha = 0.01
nu = 1 # not used
n = 10000
delta = 0.01
c = 0.1
a = 5
X = generate_bad_example(n,d)
Z,H = add_perturbation_to_bad_example(X,alpha); G = ~H
Sigma = np.eye(d); Sigma[0,0] += 0.1

U_hat, S_max = robust_subspace_estimation(Z,alpha,delta,k,nu)
print("fraction of corruption in data: %f" % H.mean())
print("Captured variance by RPCA: %f" % np.trace(np.dot(U_hat.T,np.dot(Sigma,U_hat))))
print("fraction of corruption after RPCA: %f"  % float((H*S_max).sum()/S_max.sum()))
print("fraction of good points removed after RPCA: %f" % (((~S_max) * G).sum() / G.sum()))
print("fraction of points removed by RPCA: %f" % float(1.-S_max.sum()/n))

t = int((1.-alpha)*n)
T = n/4
W_bar, removed_indexes=HRPCA(Z, k, T, t)
S_max_hrpca=np.ones(n, dtype=bool); S_max_hrpca[removed_indexes] = False
print("fraction of corruption in data: %f" % H.mean())
print("Captured variance by HRPCA: %f" % np.trace(np.dot(W_bar.T, np.dot(Sigma, W_bar))))
print("fraction of corruption after HRPCA: %f" % float((H*S_max_hrpca).sum() / S_max.sum()))
print("fraction of good points removed after HRPCA: %f" % (((~S_max_hrpca) * G).sum() / G.sum()))
print("fraction of points removed by HRPCA: %f" % float(1.-S_max_hrpca.sum()/n))

n = 10000; d = 10; k = 1
delta = 0.001; nu = 1

Alpha = np.linspace(0.005,0.025,9)
# Alpha = [0.01]
times_RPCA = 100
times_HRPCA = 10

Top_good_eigval_RPCA = np.zeros((len(Alpha),times_RPCA))
Var_captured_RPCA = np.zeros((len(Alpha),times_RPCA));
Frac_corrupt_left_RPCA = np.zeros((len(Alpha),times_RPCA));
Frac_good_removed_RPCA = np.zeros((len(Alpha),times_RPCA));
Frac_removed_RPCA = np.zeros((len(Alpha),times_RPCA));

Top_good_eigval_HRPCA = np.zeros((len(Alpha),times_HRPCA))
Var_captured_HRPCA = np.zeros((len(Alpha),times_HRPCA))
Frac_corrupt_left_HRPCA = np.zeros((len(Alpha),times_HRPCA))
Frac_good_removed_HRPCA = np.zeros((len(Alpha),times_HRPCA))
Frac_removed_HRPCA = np.zeros((len(Alpha),times_HRPCA))

Sigma = np.eye(d); Sigma[0,0] += 0.1;
for i_alpha,alpha in enumerate(Alpha):
    for time in range(times_RPCA):
        X = generate_bad_example(n,d); Sigma_hat = np.dot(X.T,X)/n; _, _, VT = svds(Sigma_hat, k=k, which='LM', return_singular_vectors="v"); V = VT.T;
        Z,H = add_perturbation_to_bad_example(X,alpha); G = ~H
        Top_good_eigval_RPCA[i_alpha,time] = np.trace(np.dot(V.T,np.dot(Sigma,V)))
        U_hat, S_max = robust_subspace_estimation(Z,alpha,delta,k,nu)
        Var_captured_RPCA[i_alpha,time] = np.trace(np.dot(U_hat.T,np.dot(Sigma,U_hat)))
        Frac_corrupt_left_RPCA[i_alpha,time] = float((H*S_max).sum()/S_max.sum())
        Frac_good_removed_RPCA[i_alpha,time] = (((~S_max)*G).sum()/G.sum())
        Frac_removed_RPCA[i_alpha,time] = (n-S_max.sum())/n
        print("RPCA",alpha,time,Top_good_eigval_RPCA[i_alpha,time]-Var_captured_RPCA[i_alpha,time])
    for time in range(times_HRPCA):
        X = generate_bad_example(n,d); Sigma_hat = np.dot(X.T,X)/n; _, _, VT = svds(Sigma_hat, k=k, which='LM', return_singular_vectors="v"); V = VT.T;
        Z,H = add_perturbation_to_bad_example(X,alpha); G = ~H
        Top_good_eigval_HRPCA[i_alpha,time] = np.trace(np.dot(V.T,np.dot(Sigma,V)))
        W_bar, removed_indexes = HRPCA(Z,k,n/4,(1-alpha)*n); S_max_hrpca = np.ones(n,dtype=bool); S_max_hrpca[removed_indexes] = False
        Var_captured_HRPCA[i_alpha,time] = np.trace(np.dot(W_bar.T,np.dot(Sigma,W_bar)))
        Frac_corrupt_left_HRPCA[i_alpha,time] = float((H*S_max_hrpca).sum()/S_max.sum())
        Frac_good_removed_HRPCA[i_alpha,time] = (((~S_max_hrpca)*G).sum()/G.sum())
        Frac_removed_HRPCA[i_alpha,time] = (n-S_max_hrpca.sum())/n
        print("HRPCA",alpha,time,Top_good_eigval_HRPCA[i_alpha,time]-Var_captured_HRPCA[i_alpha,time])

Top_good_eigval_RPCA = np.array(Top_good_eigval_RPCA)
Var_captured_RPCA = np.array(Var_captured_RPCA)
Frac_corrupt_left_RPCA = np.array(Frac_corrupt_left_RPCA)
Frac_good_removed_RPCA = np.array(Frac_good_removed_RPCA)
Frac_removed_RPCA = np.array(Frac_removed_RPCA)
Top_good_eigval_HRPCA = np.array(Top_good_eigval_HRPCA)
Var_captured_HRPCA = np.array(Var_captured_HRPCA)
Frac_corrupt_left_HRPCA = np.array(Frac_corrupt_left_HRPCA)
Frac_good_removed_HRPCA = np.array(Frac_good_removed_HRPCA)
Frac_removed_HRPCA = np.array(Frac_removed_HRPCA)

dd = {}
dd['Top_good_eigval_RPCA'] = Top_good_eigval_RPCA
dd['Var_captured_RPCA'] = Var_captured_RPCA
dd['Frac_corrupt_left_RPCA'] = Frac_corrupt_left_RPCA
dd['Frac_good_removed_RPCA'] = Frac_good_removed_RPCA
dd['Frac_removed_RPCA'] = Frac_removed_RPCA
dd['Top_good_eigval_HRPCA'] = Top_good_eigval_HRPCA
dd['Var_captured_HRPCA'] = Var_captured_HRPCA
dd['Frac_corrupt_left_HRPCA'] = Frac_corrupt_left_HRPCA
dd['Frac_good_removed_HRPCA'] = Frac_good_removed_HRPCA
dd['Frac_removed_HRPCA'] = Frac_removed_HRPCA

filename = 'alpha_%d_%d.npy' % (times_RPCA,times_HRPCA)
np.save(filename,dd)

times_RPCA = 100; times_HRPCA = 10
filename = 'alpha_%d_%d.npy' % (times_RPCA,times_HRPCA)
dd = np.load(filename,allow_pickle=True)[()]

Alpha = np.linspace(0.005,0.025,9); n = 10000

Top_good_eigval_RPCA = dd['Top_good_eigval_RPCA']
Var_captured_RPCA = dd['Var_captured_RPCA']
Frac_corrupt_left_RPCA = dd['Frac_corrupt_left_RPCA']
Frac_good_removed_RPCA = dd['Frac_good_removed_RPCA']
Frac_removed_RPCA = dd['Frac_removed_RPCA']
Top_good_eigval_HRPCA = dd['Top_good_eigval_HRPCA']
Var_captured_HRPCA = dd['Var_captured_HRPCA']
Frac_corrupt_left_HRPCA = dd['Frac_corrupt_left_HRPCA']
Frac_good_removed_HRPCA = dd['Frac_good_removed_HRPCA']
Frac_removed_HRPCA = dd['Frac_removed_HRPCA']

Top_good_eigval = np.hstack((Top_good_eigval_RPCA,Top_good_eigval_HRPCA))
Oracle_var = Top_good_eigval.mean(axis=1)
Oracle_var_std = Top_good_eigval.std(axis=1)/np.sqrt(times_RPCA+times_HRPCA)

RPCA_var = Var_captured_RPCA.mean(axis=1)
RPCA_var_std = Var_captured_RPCA.std(axis=1)/np.sqrt(times_RPCA)
HRPCA_var = Var_captured_HRPCA.mean(axis=1)
HRPCA_var_std = Var_captured_HRPCA.std(axis=1)/np.sqrt(times_HRPCA)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

y_oracle_tick = [Oracle_var.mean()]
plt.errorbar(Alpha, RPCA_var, yerr=RPCA_var_std, fmt="--o", ms="8", label=r"Algorithm $2$", c=colors[0])
plt.errorbar(Alpha, HRPCA_var, yerr=HRPCA_var_std, fmt="--s", ms="8", label=r"HRPCA", c=colors[1])
plt.axhline(y=Oracle_var.mean(), xmin=0, xmax=1, c="g", label=r"Oracle")
plt.grid(which='both',linestyle='dotted')
plt.legend(loc='center', bbox_to_anchor=(0.5,1.09),fancybox=False, shadow=False, ncol=3, **{'fontsize': 15,})
plt.ylim([1.0,1.1])
plt.yticks(list(plt.yticks()[0]) + y_oracle_tick)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
plt.tight_layout()
plt.savefig(filename[:-4]+"_.png",dpi=300); plt.close()


RPCA_frac_left = n*Frac_corrupt_left_RPCA.mean(axis=1)
HRPCA_frac_left = n*Frac_corrupt_left_HRPCA.mean(axis=1)
RPCA_frac_good_rem = n*Frac_good_removed_RPCA.mean(axis=1)
HRPCA_frac_good_rem = n*Frac_good_removed_HRPCA.mean(axis=1)
RPCA_frac_left_std = n*Frac_corrupt_left_RPCA.std(axis=1)/np.sqrt(times_RPCA)
HRPCA_frac_left_std = n*Frac_corrupt_left_HRPCA.std(axis=1)/np.sqrt(times_HRPCA)
RPCA_frac_good_rem_std = n*Frac_good_removed_RPCA.std(axis=1)/np.sqrt(times_RPCA)
HRPCA_frac_good_rem_std = n*Frac_good_removed_HRPCA.std(axis=1)/np.sqrt(times_HRPCA)

plt.errorbar(Alpha,RPCA_frac_left, yerr=RPCA_frac_left_std,fmt="--o",ms="8",label=r"Alg. $2$ remaining corrupted",fillstyle='none',c=colors[0])
plt.errorbar(Alpha,HRPCA_frac_left, yerr=HRPCA_frac_left_std,fmt="--s",ms="8",label=r"HRPCA remaining corrupted",fillstyle='none',c=colors[1])
plt.errorbar(Alpha,RPCA_frac_good_rem, yerr=RPCA_frac_good_rem_std,fmt=":o",ms="8",label=r"Alg. $2$ removed uncorrupted",c=colors[0])
plt.errorbar(Alpha,HRPCA_frac_good_rem, yerr=HRPCA_frac_good_rem_std,fmt=":s",ms="8",label=r"HRPCA removed uncorrupted",c=colors[1])
plt.yscale('log');
plt.grid(which='both',linestyle='dotted')
plt.legend(loc='center', bbox_to_anchor=(0.5,1.15),fancybox=False, shadow=False, ncol=2, **fontSpecMedium)
plt.tight_layout()
plt.savefig(filename[:-4]+"_frac.png",dpi=300); plt.close()

