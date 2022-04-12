#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from scipy.linalg import expm, sinm, cosm

import numpy as np
from sklearn.preprocessing import scale, normalize
import scipy as sp
from itertools import product
import pickle
import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import jit, jacfwd, jacrev
from jax import value_and_grad
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from jax import random
from jax.example_libraries import optimizers

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from google.colab import files
import plotly.express as px


import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from sklearn import preprocessing
import plotly.graph_objects as go

import numpy as np
from numpy.random import normal as normal
print('numpy: '+np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
print('matplotlib: '+matplotlib.__version__)


# #  Data processing
# 

# In[4]:


with open('mnist_lowpixel_1000.pickle', 'rb') as f:
    XX,XX_test,color,color_test = pickle.load(f)

XX = XX/16

i_0 = np.where(color==0)
i_1 = np.where(color==1)
i_2 = np.where(color==2)
i_3 = np.where(color == 3)
XX_new = np.concatenate((XX[i_0],XX[i_1],XX[i_2],XX[i_3]),0)

color_new = np.concatenate((color[i_0],color[i_1],color[i_2],color[i_3]))

XX = XX_new
color= color_new


# In[6]:


def mean_centering(data):

  #Standardize features by removing the mean and scaling to unit variance.
  scaler = preprocessing.StandardScaler().fit(data)
  X_scaled = scaler.transform(data)
  print('colume_mean', np.mean(X_scaled,0))
  return X_scaled


# In[12]:


def pca_to_d(data,reduced_dim):

  pca = PCA(n_components=reduced_dim)
  data_embed = pca.fit(data).transform(data)

  return data_embed


# # Gradient compuation
# 

# In[17]:



def  batch_generator(data,batch_size):
    
    input_data = shuffle(data)
    batch_data=[]
    for ind in range(0, len(input_data), batch_size):

            input_batch = input_data[ind:ind + batch_size]
            batch_data.append(input_batch)
    
    return jnp.array(batch_data)


# In[18]:


def thresolding_rhs(vec,cutoff):
  vec_np = np.array(vec)
  vec_np[vec_np>cutoff]=cutoff
  vec_np[vec_np<-cutoff]=-cutoff
  
  return jnp.array(vec_np)


# In[19]:



def sol_ode(beta,X_batch,Xi):
    
    
    Y_pre = X_batch
    sol_t = []
    sol_t.append(Y_pre)
        
    for i in range(len(t)-1):
       
        rhs = Xi(Y_pre)@beta
       
       
        
        Y_new = Y_pre + dt*rhs
        
        Y_new = thresolding_rhs(Y_new, 100)
        
        
        sol_t.append(Y_new)
        Y_pre = Y_new
        
    sol_t= jnp.array(sol_t) 
    
    return sol_t


# In[20]:



def sol_adjoint(beta,Q,X_batch,Xi,sol_t):
    
    (N_batch,d) = X_batch.shape
    
    
    sol_at_fT = sol_t[-1]
    
    dn = Xi(sol_at_fT).shape[1]
    lam = -2*sol_at_fT@(jnp.eye(d)-2*Q.T@Q +  Q.T@Q@Q.T@Q)
    lam = thresolding_rhs(lam,10)
    

    
    J_beta_1 = np.zeros(beta.shape)
    J_beta_2 = np.zeros(beta.shape)

    
    J2 = 0 
    
    for i in range(len(t)-1):
      
        
        sol_at_t = sol_t[len(t)-i-1]
        
       
        Xi_at_t = Xi(sol_at_t)
        
        J2 =  J2 +  jnp.sum((Xi_at_t@beta)**2)*dt
 
        
        J_beta_1 = J_beta_1  + (2*Xi_at_t.T@Xi_at_t@beta)*dt
        J_beta_2 =  J_beta_2 - (Xi_at_t.T@lam)*dt

        grad_x_trans = grad_ft_true(sol_at_t,dn)
        
        linear =   - jnp.einsum('ij,ijk->ik',lam@beta.T, grad_x_trans)
        forcing =  2*mu*jnp.einsum('ij,ijk->ik',  Xi_at_t@beta@beta.T, grad_x_trans)
        rhs = linear +  forcing


        lam =lam  - dt*rhs 
        lam = thresolding_rhs(lam,100)

      
   
        
        
    J_beta  =  ( mu*J_beta_1 + J_beta_2 )/N_batch
    
    
    
    return J_beta, mu*J2/N_batch


# In[21]:


def J_Q(sol_t,k,Q):
    sol_fT = sol_t[-1]
    N_batch = sol_fT.shape[0]
    
    
    J1 =   jnp.sum((sol_fT@(jnp.eye(d)-Q.T@Q))**2)/N_batch
    
    return  J1
    
    


# # Setting for iteration
# 

# In[26]:


def approx_eps(mu):

  e = np.exp(1)
  c1 = e/(e**2-1)
  c2 = (e**2 + 3*e**4)/(2*(e**2-1)**2)

  ep_app = (-c1+ np.sqrt(c1**2+c2*mu))/(2*c2) + 1/e
  M = jnp.log(ep_app)

  return M


# In[27]:





def bigxi_true(x):
    
    rhs = jnp.concatenate([jnp.power(x,3)],1)
    return rhs
    

def initialize_para(XX,size, key, scale,mu):
    (dn,d),(k,d) = size
   
    M = approx_eps(mu)
    u,sigma,v  = jnp.linalg.svd(XX.T,full_matrices=True)
    Q = u[:,:k].T
    
   
    beta= scale* random.normal(key, (dn, d))
    
    
    return (beta,Q)

def grad_ft_true(sol_at_t, dn):
    
    # gradient : N x dn x d
    (N,d) = sol_at_t.shape
    gradient = np.zeros((N, dn, d))

    
    for i in range(d):
    
      gradient[:,i ,i ] = jnp.multiply(jnp.power(sol_at_t[:,i],2),3)
   
   
    return jnp.array(gradient)
 
  


# In[28]:


def draw_Q_space(Q):
    q1 = Q[0,:]
    q2 = Q[1,:]
    normal = jnp.cross(q1,q2)
    
    xx, yy = jnp.meshgrid(jnp.linspace(-1.1,1.1,110), jnp.linspace(-1.1,1.1,110))

    # calculate corresponding z
    z = (-normal[0]*xx - normal[1]* yy )/normal[2]
    
    return xx,yy,z


# In[29]:


def visualization(data,color,beta,Q):
    sol_t = sol_ode(beta,data,Xi)
    sol_at_fT_opt = sol_t[-1]
    embed = sol_at_fT_opt@Q.T

    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(embed[:,0], embed[:,1],c=color,cmap=plt.cm.jet,s = 15 , label = color)
    return sol_t , sol_at_fT_opt,embed


# In[35]:



def main(beta_ini, Q_ini,n_epoch,step_size,batch_size,Beta, QQ, JJ1, JJ2, Loss):
    
   
    
    opt_init, opt_update, get_params = optimizers.adam(step_size)

    opt_state = opt_init(beta_ini)
    
    Q = Q_ini


    for epoch in range(n_epoch):
      
        start_time = time.time()
        Batch = batch_generator(XX,train_batch_size)


        for n_batch in range(len(Batch)):
            
            batch_x=  Batch[n_batch]
          
            # get parameters
            beta=  get_params(opt_state)
            

          
            # forward ODE
            sol_t = sol_ode(beta,batch_x,Xi)
            sol_fT = sol_t[-1]

            # update Q
            u_T,sigma_T,v_T  = jnp.linalg.svd(sol_fT.T,full_matrices=True)
            Q = u_T[:,:k].T
          
            # adjoint 
            J_beta,  J2 = sol_adjoint(beta,Q, batch_x,Xi,sol_t)
            J1 =  J_Q(sol_t,k,Q)



            # save beta and Q
            Beta.append(beta)
            QQ.append(Q)
            

            # see optax later
            # update
            opt_state = opt_update(epoch,J_beta, opt_state)
            


            total_loss  = J1+J2
            print('epoch',epoch,'J1:',J1,'J2:',J2,'Loss:', total_loss)
            epoch_time = time.time()-start_time
            print(epoch_time)
            Epoch_time.append(epoch_time)
            print('----------------------------------------------')
            
            
            
            
            
            
        JJ1.append(J1)
        JJ2.append(J2)
        epoch_time = time.time()-start_time
        print()

        Loss.append(total_loss)
        
    return Beta[-1],QQ[-1]


# In[32]:


def schedule_learning(n_epoch_i, step_size_i, beta_i, Q_i,train_batch_size_i):

    
 
  beta_next,Q_next  = main(beta_i, Q_i,n_epoch_i,step_size_i,train_batch_size_i,Beta, QQ, JJ1, JJ2, Loss)
 

  return beta_next, Q_next


# # Customize
# 

# In[ ]:


# downsize the initial data dimension by applying the PCA
XX = pca_to_d(XX,10)
(N,d) = XX.shape


# In[33]:


# set dictionary

Xi  = bigxi_true
Xi_XX = Xi(XX)

dn = Xi_XX.shape[1]

k = 2


# In[36]:



Epoch_time=[]
(N,d) = XX.shape
train_batch_size  = len(XX)

# hyper-parameters 


mu = 0.009 # hyper-parameter


fT  = 1
t= jnp.linspace(0,fT, 100)
dt = t[1]-t[0]




# schedule leanring plan
N_epoch = [200,200,500]
Step_size = [0.01,0.05,0.001]

Beta = []
QQ=[]
JJ1 =[]
JJ2 = []
Loss= []


# initializer
scale = 0.0 #perturb initial beta
key = random.PRNGKey(0)

(beta_ini,Q_ini)= initialize_para(XX,((dn,d), (k,d) ), key, scale,mu)  
Beta.append(beta_ini)
QQ.append(Q_ini)


start_time = time.time()


for i in range(len(N_epoch)):

  
  beta_next, Q_next = schedule_learning(N_epoch[i], Step_size[i], beta_ini, Q_ini,train_batch_size)
  beta_ini = beta_next
  Q_ini = Q_next  


N_epoch = np.sum(N_epoch)
epoch_time = time.time()-start_time    
print(N_epoch,'th computing time : ',epoch_time,'------------------')
 


# In[37]:



import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d


fig, axes = plt.subplots(3, sharex=True, figsize=(8, 5))
fig.suptitle('Training Metric')

axes[0].set_ylabel("J", fontsize=14)

#axes[0].plot(JJ1,'r')
#axes[0].plot(JJ2,'g')
axes[0].plot(Loss[:])

axes[1].set_ylabel("J1", fontsize=14)
axes[1].plot(JJ1[:],'r')

axes[2].set_ylabel("J2", fontsize=14)
axes[2].plot(JJ2[:],'r')


# In[38]:


beta_ini = Beta[0]
Q_ini = QQ[0]


beta_opt = Beta[-1]
Q_opt = QQ[-1]


# In[43]:



sol_t, sol_fT, embed = visualization(XX,color,beta_opt,Q_opt)


#plt.savefig('MNIST4_d_10_mu_0_01.eps')

