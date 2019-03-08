# Robust QP problem:
# f(x)=x1^2-8*x1+4*x2^2-16*x2
# g(x,u)=(a+Pu)'*x-b<=0
# h(u)=u'*u-1<=0

# importing
import numpy as np
import time
import matplotlib.pyplot as plt

# Optimization problem parameters
n=2

if n==1:
    a=np.array([1])
    b=1
    P=np.array([[1]])
elif n==2:
    a=np.array([[1],[-7]])
    b=5
    P=np.array([[10,15],[15,0.1]])

###############################################################################
### Sub-gradient method
# initialization
iter_no=40000
init=iter_no+1
Lx=np.zeros((n,init))
Llam=np.zeros((1,init))
Liu=np.zeros((n,init))
Liv=np.zeros((1,init))
norm_T=np.zeros((1,init))
alpha=np.zeros((1,init))
gamma=np.zeros((1,init))
c=0
x=c*np.ones((n,init))
u=c*np.ones((n,init))
lam=c*np.ones((1,init))
v=0*np.ones((1,init))

# Subgradient algorithm iterations
t=time.time()
for k in range(iter_no):
    proj=1 # new projections active or not active
    pu=2 # power of 2-norm in uncertainty set

    hu=np.linalg.norm(u[:,k])**pu-1**pu
    dhu=pu*u[:,k,None]
    if proj==1 and hu<=0:
        hu=0
        dhu=np.zeros((n,1))

    gxu=np.dot(np.transpose(a+np.dot(P,u[:,k,None])),x[:,k,None])-b
    dgx=(a+np.dot(P,u[:,k,None]))
    dgu=np.dot(np.transpose(P),x[:,k,None])
#    if proj==1 and gxu<=0:
#        gxu=0
#        dgu=np.zeros((n,1))
#        dgx=np.zeros((n,1))

    rho=0 # for augmentation term
    # subgradients
    if n==1:
        Lx[:,k,None]=np.array([[-8+2*x[0,k]]])+lam[0,k]*dgx
    elif n==2:
        Lx[:,k,None]=np.array([[-8+2*x[0,k]],[-16+8*x[1,k]]])+lam[0,k]*dgx
    Llam[:,k,None]=-(gxu-v[0,k]*hu-(rho/2)*(np.linalg.norm(hu))**2)
    Liu[:,k,None]=-(dgu-dhu*v[0,k]-rho*hu*dhu)
    Liv[:,k,None]=-(hu*lam[0,k])

    # stepsize
    if k>0:
        gamma[0,k]=10/k
    norm_T[0,k]=np.linalg.norm(np.vstack((Lx[:,k,None],Llam[0,k],Liu[:,k,None],Liv[:,k,None])))
    alpha[0,k]=gamma[0,k]/norm_T[0,k]
    #alpha[0,k]=0.005

    # iterations
    x[:,k+1,None]=x[:,k,None]-alpha[0,k]*Lx[:,k,None]
    lam[0,k+1]=lam[0,k]-alpha[0,k]*Llam[0,k]
    if lam[0,k+1]<=0:# and proj==0:
        lam[0,k+1]=0
    u[:,k+1,None]=u[:,k,None]-alpha[0,k]*Liu[:,k,None]
    v[0,k+1]=v[0,k]-alpha[0,k]*Liv[0,k]
#    if proj==0 and v[0,k+1]<=0:
#        v[0,k+1]=0

elapsed=time.time()-t
print('elapsed time =',elapsed)
print('uncertainty set =',np.linalg.norm(u[:,k])**pu-1**pu)
print('constraint =',float(np.dot(np.transpose(a+np.dot(P,u[:,k,None])),x[:,k,None])-b))
print('x =',x[:,-1])

# plot
K=np.linspace(0,init,init)
plt.figure(1)
plt.plot(K,x[0,:],label=r"$x_1$")
if n==2:
    plt.plot(K,x[1,:],label=r"$x_2$")
plt.plot(K,lam[0,:],label=r"$\lambda$")
if n==1:
    plt.plot(K,u[0,:],label=r"$u$")
if n==2:
    uu=[]
    for i in range(len(u[0,:])):
        uu.append(np.linalg.norm(u[:,i]))
    plt.plot(K,uu[:],label=r"$\Vert u \Vert$")
plt.plot(K,v[0,:],label=r"$v$")
plt.legend()
plt.show()
