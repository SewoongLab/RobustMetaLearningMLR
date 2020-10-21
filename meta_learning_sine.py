from meta_learning_utils import *


#####################################
# Function overloads to generate data
#####################################

def get_examples_from_id(w_id,Omega,Phi,s,t):
    X_raw = np.random.uniform(-1,1,t)
    y = np.sin(2*np.pi*Omega[w_id]*X_raw + Phi[w_id]) + np.random.normal(0,s[w_id],t)
    return X_raw, y

def get_examples_random_task(Omega,Phi,s,p,t):
    k = Omega.shape[0]
    w_id = np.random.choice(k,size=1,replace=True,p=p)[0]
    X_raw, y = get_examples_from_id(w_id,Omega,Phi,s,t)
    return w_id, X_raw, y

def get_random_ids(p,n):
    return np.random.choice(k,size=n,replace=True,p=p)

def get_random_meta_data(Omega,Phi,s,p,n,t):
    return map(list, zip(*[get_examples_random_task(Omega,Phi,s,p,t) for i in range(n)]))

def features(X_raw,d):
    X = np.polynomial.chebyshev.chebvander(X_raw,d-1)/((1-X_raw**2)**(1./4))[:,None]
    X[:,0] /= np.sqrt(2)
    return X

def features(X_raw,d):
    return np.polynomial.chebyshev.chebvander(X_raw,d-1)

def beta_hat_from_X_y(X,y):
    return np.dot(X.T,y)/X.shape[0]

def beta_hat_from_id(w_id,Omega,Phi,s,t,d):
    X_raw, y = get_examples_from_id(w_id,Omega,Phi,s,t)
    X = features(X_raw,d)
    return np.dot(X.T,y)/t

def beta_hats_from_ids(w_ids,Omega,Phi,s,t,d):
    return [beta_hat_from_id(w_id,Omega,Phi,s,t,d) for w_id in w_ids]

############################################
# Function overloads for subspace estimation
############################################

def get_Mhat(w_ids,Omega,Phi,s,t,d):
    n = len(w_ids); k = len(Omega)
    beta_hat_1 = np.array(beta_hats_from_ids(w_ids,Omega,Phi,s,t,d))
    beta_hat_2 = np.array(beta_hats_from_ids(w_ids,Omega,Phi,s,t,d))
    Mhat = np.dot(beta_hat_1.T,beta_hat_2)/n
    Mhat = (Mhat + Mhat.T)/2
    return Mhat

#####################################
# Function overloads for clustering
#####################################

def get_H(w_ids_n1,U,Omega,Phi,s,t,d,L,projection=True):
    n1 = len(w_ids_n1)
    HH = np.zeros((L,n1,n1))
    for l in tqdm(range(L)):
        betas_1 = np.array(beta_hats_from_ids(w_ids_n1,Omega,Phi,s,t,d))
        betas_2 = np.array(beta_hats_from_ids(w_ids_n1,Omega,Phi,s,t,d))
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

def clustering(t,w_ids,L,U,Omega,Phi,s,d,projection=True,plot=""):
    d, k = U.shape
    H, UTb_1, UTb_2 = get_H(w_ids,U,Omega,Phi,s,t,d,L,projection); np.fill_diagonal(H, 0, wrap=False)
    Z = linkage(ssd.squareform(np.abs(H)), method="average")
    C_1 = fcluster(Z, k, criterion='maxclust')-1
    clustering_acc = clustering_accuracy(C_1,w_ids,k)
    return clustering_acc, C_1, UTb_1, UTb_2, H

#####################################
# Function overloads for prediction
#####################################

def prediction_errors(w_ids,t,Omega,Phi,s,W_hats,s2_hat,p_hats):
    n = len(w_ids)
    k, d = W_hats.shape
    C = []; Gen_Err_MAP = []; Gen_Err_PM = []; Train_Err_MAP = []; Train_Err_PM = []; LS_Train_Err = []; LS_Test_Err = []
    for w_id in tqdm(w_ids):
        X_train_raw, y_train = get_examples_from_id(w_id,Omega,Phi,s,t); X_train = features(X_train_raw,d)
        X_test_raw, y_test = get_examples_from_id(w_id,Omega,Phi,s,t); X_test = features(X_test_raw,d)
        l = get_l_pred(t,k,s2_hat,p_hats,W_hats,X_train,y_train)
        L = np.exp(l); L /= np.sum(L)
        i_hat = np.argmax(l); C.append(i_hat)
        beta_hat_map = W_hats[i_hat]
        gen_err_map = norm(np.dot(X_test,beta_hat_map)-y_test)**2/t
        Gen_Err_MAP.append(gen_err_map)
        beta_hat_pm = np.zeros_like(beta_hat_map)
        for i in range(k):
            beta_hat_pm += W_hats[i]*L[i]
        gen_err_pm = norm(np.dot(X_test,beta_hat_pm)-y_test)**2/t
        Gen_Err_PM.append(gen_err_pm)
        train_err_map = norm(np.dot(X_train,beta_hat_map)-y_train)**2/t; train_err_pm = norm(np.dot(X_train,beta_hat_pm)-y_train)**2/t
        Train_Err_MAP.append(train_err_map); Train_Err_PM.append(train_err_pm)
        w_hat_ls = np.linalg.lstsq(X_train,y_train,rcond=-1)[0]
        train_err = norm(np.dot(X_train,w_hat_ls)-y_train)**2/t; test_err = norm(np.dot(X_test,w_hat_ls)-y_test)**2/t
        LS_Train_Err.append(train_err); LS_Test_Err.append(test_err)
    C = np.array(C); Gen_Err_MAP = np.array(Gen_Err_MAP); Gen_Err_PM = np.array(Gen_Err_PM)
    return C, Gen_Err_MAP, Gen_Err_PM, Train_Err_MAP, Train_Err_PM, LS_Train_Err, LS_Test_Err


############################
# Generate data priors
############################

k = 8
d = 16

Omega = np.linspace(1,2,k)
Phi = np.linspace(0,2*np.pi,k)[np.random.permutation(k)]
s = 0.5*np.ones(k)
p = np.ones(k)/k

############################
# Subspace estimation
############################

t1 = 2**3
n1 = 2**13

w_ids = get_random_ids(p,n1)

Mhat = get_Mhat(w_ids,Omega,Phi,s,t1,d)
U = compute_subspace(Mhat,k)

############################
# Clustering
############################

t2 = 16
n2 = 2**7
L = 1

w_ids = get_random_ids(p,n2);
clustering_acc, C_1, UTb_1, UTb_2, H = clustering(t2,w_ids,L,U,Omega,Phi,s,d,projection=True)
print("Clustering accuracy = {:.3f}, t2 = {:d}".format(clustering_acc,t2))

UTb_mean, cluster_hist = get_cluster_mean(np.concatenate([UTb_1,UTb_2],axis=0),np.concatenate([C_1,C_1]),k)
C = C_1
cluster_map, original_map = get_maps(C,w_ids,k) # cluster_map[original_id] = new_id, original_map[new_id] = original_id

###################################
# Initial parameter estimation
###################################

UTW_hat = UTb_mean
W_hat = np.dot(U,UTW_hat.T).T
p_hat = cluster_hist/norm(cluster_hist,1)

r2_hat = np.zeros(k)
X_super_init = []; y_super_init = []

for l in range(k):
    X_raw, y = get_examples_from_id(original_map[l],Omega,Phi,s,cluster_hist[l]*t2)
    X = features(X_raw,d)
    z = y - np.dot(X,W_hat[l])
    r2_hat[l] = norm(z)**2/(cluster_hist[l]*t2)
    X_super_init.append(X); y_super_init.append(y)

p_estimation_err = p_estimation_error(p,p_hat,cluster_map)/p.min()
print("r2 err = {:5f}, p_err = {:.5f}".format((r2_hat - s**2).mean(),p_estimation_err))

##################################################
# Classification and final parameter estimation
##################################################

t3 = 2**2
n3 = 2**13

w_ids = get_random_ids(p,n3)
C = []; XX = []; yy = []
for w_id in tqdm(w_ids):
    X_raw, y = get_examples_from_id(w_id,Omega,Phi,s,t3)
    X = features(X_raw,d)
    XX.append(X); yy.append(y)
    l = get_l(t3,k,r2_hat,W_hat,X,y)
    C.append(np.argmax(l))

C = np.array(C)
XX = np.stack(XX); yy = np.stack(yy)
classification_acc = np.mean(classification_accuracy(C,w_ids,cluster_map,k))
print("Classification accuracy = {:.4f}, n3 = {:d}, t3 = {:d}".format(classification_acc,n3,t3))

X_super = []; y_super = []
W_hats = []; cluster_hist = np.zeros(k,dtype=int); s2_hat = np.zeros(k)

for l in range(k):
    X_super.append(np.vstack(XX[C==l])); y_super.append(np.hstack(yy[C==l]))
    w_hat, s2, _, _ = np.linalg.lstsq(np.vstack([X_super[l],X_super_init[l]]),np.hstack([y_super[l],y_super_init[l]]),rcond=-1)
    W_hats.append(w_hat)
    cluster_hist[l] = np.sum(C==l)
    s2_hat[l] = s2/(cluster_hist[l]*t3-d)

W_hats = np.array(W_hats)
p_hats = cluster_hist/cluster_hist.sum()

p_estimation_err = p_estimation_error(p,p_hats,cluster_map)*np.sqrt(d/t3)
s2_estimation_err = np.max(np.abs(s**2-s2_hat)/s**2)*np.sqrt(d)

print("s2 err = {:5f}, p_err = {:.5f}".format(s2_estimation_err,p_estimation_err))

##############################################################################
# Prediction
##############################################################################

n = 1024
tt = np.array([2**2])

Train_MAP_err = np.zeros(tt.shape[0]); Train_Bayes_err = np.zeros(tt.shape[0]);
Test_MAP_err = np.zeros(tt.shape[0]); Test_Bayes_err = np.zeros(tt.shape[0]);
Train_LS_err = np.zeros(tt.shape[0]); Test_LS_err = np.zeros(tt.shape[0]);
Classification_Acc = np.zeros(tt.shape[0])
Train_MAP_std = np.zeros(tt.shape[0]); Train_Bayes_std = np.zeros(tt.shape[0]);
Test_MAP_std = np.zeros(tt.shape[0]); Test_Bayes_std = np.zeros(tt.shape[0]);
Train_LS_std = np.zeros(tt.shape[0]); Test_LS_std = np.zeros(tt.shape[0]);

for i,t in enumerate(tt):
    w_ids = get_random_ids(p,n)
    C, Gen_Err_MAP, Gen_Err_PM, Train_Err_MAP, Train_Err_PM, LS_Train_Err, LS_Test_Err = prediction_errors(w_ids,t,Omega,Phi,s,W_hats,s2_hat,p_hats)
    classification_acc = np.mean(classification_accuracy(C,w_ids,cluster_map,k))
    print("Classification accuracy = %f" % classification_acc)
    Train_MAP_err[i] = np.mean(Train_Err_MAP); Train_Bayes_err[i] = np.mean(Train_Err_PM)
    Test_MAP_err[i] = np.mean(Gen_Err_MAP); Test_Bayes_err[i] = np.mean(Gen_Err_PM)
    Train_LS_err[i] = np.mean(LS_Train_Err); Test_LS_err[i] = np.mean(LS_Test_Err)
    Train_MAP_std[i] = np.std(Train_Err_MAP); Train_Bayes_std[i] = np.std(Train_Err_PM)
    Test_MAP_std[i] = np.std(Gen_Err_MAP); Test_Bayes_std[i] = np.std(Gen_Err_PM)
    Train_LS_std[i] = np.std(LS_Train_Err); Test_LS_std[i] = np.std(LS_Test_Err)
    Classification_Acc[i] = classification_acc

print(Test_MAP_err, Test_Bayes_err)

#################################
# Plotting data
#################################

X_train_raws = []; X_trains = []; y_trains = []
for w_id in range(k):
    X_train_raw, y_train = get_examples_from_id(w_id,Omega,Phi,s,t); X_train = features(X_train_raw,d)
    X_train_raws.append(X_train_raw); X_trains.append(X_train); y_trains.append(y_train)

fig_n = 2; fig_m = 4
unit_n = 8; unit_m = 6
fig, axs = plt.subplots(fig_n, fig_m,figsize=(fig_m*unit_m,fig_n*unit_n))

for (i,j),ax in np.ndenumerate(axs):
    w_id = i*fig_m+j
    X_train_raw = X_train_raws[w_id]; X_train = X_trains[w_id]; y_train = y_trains[w_id]
    ax.scatter(X_train_raw,y_train,label=r'Train Samples')
    ax.set_xlabel(r"$x$",**fontSpecLarge); ax.set_ylabel(r"$y$",**fontSpecLarge)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
    ax.grid(which='both',linestyle='dotted');

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(pad=5, w_pad=0.5, h_pad=0.5)
fig.legend(handles,labels,loc='upper left', bbox_to_anchor=(0.2,0.06), ncol=len(labels),**fontSpecLarge)
fig.suptitle(r'Samples for $\tau =$ %d' % (t), fontsize=32)
fig.savefig("sin_wave_samples_"+str(t)+".png",dpi=150,bbox_inches='tight'); plt.close()

#################################
# Plotting data with functions
#################################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig_n = 2; fig_m = 4
unit_n = 8; unit_m = 6
fig, axs = plt.subplots(fig_n, fig_m,figsize=(fig_m*unit_m,fig_n*unit_n))

for (i,j),ax in np.ndenumerate(axs):
    w_id = i*fig_m+j
    X_train_raw = X_train_raws[w_id]; X_train = X_trains[w_id]; y_train = y_trains[w_id]
    l = get_l_pred(t,k,s2_hat,p_hats,W_hats,X_train,y_train)
    L = np.exp(l); L /= np.sum(L)
    i_hat = np.argmax(l);
    beta_hat_map = W_hats[i_hat]
    beta_hat_pm = np.dot(L,W_hats)
    xx = np.linspace(-1,1,201)
    y_true_fn = np.sin(2*np.pi*Omega[w_id]*xx + Phi[w_id])
    y_pred_map_fn = np.dot(features(xx,d),beta_hat_map)
    y_pred_bayes_fn = np.dot(features(xx,d),beta_hat_pm)
    # ax.plot(xx,y_true_fn,label=r'True function')
    # ax.plot(xx,y_pred_map_fn,label=r'MAP Predicted function',linestyle="-.")
    # ax.plot(xx,y_pred_bayes_fn,label=r'Bayes Predicted function',linestyle="-.",color=colors[1])
    ax.scatter(X_train_raw,y_train,label=r'Train Samples',s=100)
    ax.set_xlabel(r"$x$",**fontSpecLarge); ax.set_ylabel(r"$y$",**fontSpecLarge)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
    ax.grid(which='both',linestyle='dotted');

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(pad=5, w_pad=0.5, h_pad=0.5)
fig.legend(handles,labels,loc='upper left', bbox_to_anchor=(0.2,0.06), ncol=len(labels),**fontSpecLarge)
# fig.suptitle(r'Bayes test error = %.3f, $t_H =$ %d, $\tau =$ %d' % (Test_Bayes_err[0],t2,t), fontsize=32)
fig.suptitle(r'Samples for $\tau =$ %d' % (t), fontsize=32)
# fig.savefig("sin_wave_"+str(t2)+"_"+str(t)+".png",dpi=150,bbox_inches='tight'); plt.close()
# fig.savefig("sin_wave_predfn_w_samples.png",dpi=150,bbox_inches='tight'); plt.close()
fig.savefig("sin_wave_samples_"+str(t)+".png",dpi=150,bbox_inches='tight'); plt.close()

##############################

# tau = 2
# X_raw, y = get_examples_from_id(w_id,Omega,Phi,s,tau)
# plt.scatter(X_raw,y)
# plt.savefig("t_"+str(tau)+".png"); plt.close()

###############################

fig_n = 2; fig_m = 4
unit_n = 8; unit_m = 6
fig, axs = plt.subplots(fig_n, fig_m,figsize=(fig_m*unit_m,fig_n*unit_n))

for (i,j),ax in np.ndenumerate(axs):
    w_id = i*fig_m+j
    beta_hat_map = W_hats[w_id]
    xx = np.linspace(-1,1,201)
    y_true_fn = np.sin(2*np.pi*Omega[w_id]*xx + Phi[w_id])
    y_pred_map_fn = np.dot(features(xx,d),beta_hat_map)
    # ax.plot(xx,y_true_fn,label=r'True function')
    # ax.plot(xx,y_pred_map_fn,label=r'Predicted function')
    ax.scatter(X_train_raws[w_id],y_trains[w_id],label=r'Train Samples',s=100)
    ax.set_xlabel(r"$x$",**fontSpecLarge); ax.set_ylabel(r"$y$",**fontSpecLarge)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
    ax.grid(which='both',linestyle='dotted');

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(pad=5, w_pad=0.5, h_pad=0.5)
fig.legend(handles,labels,loc='upper left', bbox_to_anchor=(0.3,0.06), ncol=len(labels),**fontSpecLarge)
fig.suptitle(r'Samples and Centers for $\tau =$ %d' % (t), fontsize=32)
fig.savefig("sin_wave_centers_w_samples.png",dpi=150,bbox_inches='tight'); plt.close()


####

X_train_raws_16 = X_train_raws; X_trains_16 = X_trains; y_trains_16 = y_trains