# Robust SVM for image classification
###############################################################################
# number of iterations
iter_no=5*10**5
plot_seconds=10
# parameters
gamma_c=10 # penalty parameter of the error term
# beta=2 # augmented term, not working now
init0=0 # initial conditions
iter_type='new'
step_size='con' # con (constant) or dim (diminishing) step size'
save_show='show'
constant_step=0.001
gamma_num=3*10**2 # numinator of gamma in the stepsize
proj=0 # projections active or not active
with_xi=0 # xi exists?
with_multipliers=1 # plotting lambda and v?
###############################################################################
eta_w=1 # binary parameter for w
eta_b=1 # binary parameter for b
eta_lam=1 # binary parameter for lambda
eta_u=1 # binary parameter for u
eta_v=1 # binary parameter for v

acc_w=1 # acceleration parameter for w
acc_b=1 # acceleration parameter for b
acc_lam=1 # acceleration parameter for lambda
acc_u=1 # acceleration parameter for u
acc_v=1 # acceleration parameter for v

if with_xi==0:
    eta_xi=0 # binary parameter for xi
    eta_mu=0 # binary parameter for mu
    acc_xi=0 # acceleration parameter for xi
    acc_mu=0 # acceleration parameter for mu
elif with_xi==1:
    eta_xi=1 # binary parameter for xi
    eta_mu=1 # binary parameter for mu
    acc_xi=1 # acceleration parameter for xi
    acc_mu=1 # acceleration parameter for mu
"""
hint for choosing alpha and gamma_c:
    blobs dataset, 50 samples, std 1: alpha=0.005, gamma_c=10 / 20 samples, std 1: alpha=0.01, gamma_c=10
    digits dataset, test  0.3: alpha=0.0001~0.0002, gamma_c=10 / test  0.98: alpha=0.001, gamma_c=10
"""
###############################################################################
# standard scientific Python imports
import numpy as np
from math import atan,degrees
import time
import random
#import matplotlib
#matplotlib.use('Agg') # for c9 run
import matplotlib.pyplot as plt
from matplotlib import patches
# import dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
###############################################################################
# plotting different norms for uncertainty sets
def norm_plot(p,i,d,rho):
    xx,yy=np.meshgrid(np.linspace(d[i,0]-2*rho[i],d[i,0]+2*rho[i],num=201),
                      np.linspace(d[i,1]-2*rho[i],d[i,1]+2*rho[i],num=201))
    fig=plt.figure()
    ax=fig.gca()
    if p[i]==0:
        zz=(xx!=0).astype(int)+(yy!=0).astype(int)
        ax.imshow(zz,cmap='bwr',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect="auto")
    elif np.isinf(p[i]):
        zz=np.maximum(np.abs((xx-d[i,0])/rho[i]),np.abs((yy-d[i,1])/rho[i]))
        ax.contour(xx,yy,zz,[1],colors='red',linewidths=1.5)
    else:
        zz=((np.abs((xx-d[i,0])/rho[i])**p[i])+(np.abs((yy-d[i,1])/rho[i])**p[i]))**(1/p[i])
        ax.contour(xx,yy,zz,[1],colors='red',linewidths=1.5)
    plt.axis('equal')
    fig.suptitle(("p=%1.2f"%p[i]),fontsize=12,y=.93)
    plt.show()
###############################################################################
# Dataset
dataset=1 # 1 for blobs dataset, 2 for simple dataset, 3 for digits datasets
if dataset==1:
    # sklearn blobs dataset
    from sklearn.datasets.samples_generator import make_blobs
    d,yy_train=make_blobs(n_samples=20,centers=2,random_state=1,cluster_std=3.2)
    d*=2
    x_train=[None]*d.shape[0]
    for i in range(np.shape(x_train)[0]):
        x_train[i]=d[i].T.flatten()[:,None]
    y_train=yy_train.copy()
    y_train[yy_train==0]=-1
    y_train=y_train[:,None]
    plotting='on'
elif dataset==2:
    # simple dataset
    d=np.array([[1,0],[0,-1],[0.5,-0.5],[0,1],[-1,0],[-0.5,0.5]])
    yy_train=np.array([[1],[1],[1],[0],[0],[0]])
    y_train=yy_train.copy()
    y_train[yy_train==0]=-1
    plotting='on'
elif dataset==3:
    # digits dataset (8x8 images of digits)
    # attributes of the dataset: images, target
    digits=datasets.load_digits()
    n1,n2=np.shape(digits.images[0])
    classified_digit_1=3
    classified_digit_2=8
    digits.images=[item for counter,item in enumerate(digits.images) if
                   digits.target[counter]==classified_digit_1 or digits.target[counter]==classified_digit_2]
    digits.target=[item for item in digits.target if item==classified_digit_1 or item==classified_digit_2]
    image_label=list(zip(digits.images,digits.target))
    no_images=np.shape(digits.images)[0]
    vec_image_data=[None]*no_images
    for index,(image,label) in enumerate(image_label[:no_images]):
        # plot some of the images
        if no_images<=20:
            fig1=plt.figure(1)
            plt.subplot(no_images/4,4,index+1)
            plt.axis('off')
            plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
            plt.title('Digit: %i' % label)
        # flatten the pixel matrix into a column vector
        vec_image_data[index]=digits.images[index].flatten()[:,None]
    #d=np.array([vec_image_data[index].T for index in range(no_images)]).reshape(no_images,n1*n2)
    x_train,x_test,target_train,target_test=train_test_split(vec_image_data,digits.target,test_size=0.98,random_state=109)

    # creating labels for two digits (binary classification)
    yy_train=np.array([0]*np.shape(target_train)[0])[:,None]
    for counter,item in enumerate(target_train):
        if item==classified_digit_1:
            yy_train[counter]=1
        elif item==classified_digit_2:
            yy_train[counter]=0

    y_train=yy_train.copy()
    y_train[yy_train==0]=-1
    print(y_train)
    plotting='off'
    # scale the data in the range [0,1] # this is decreasing the accuracy
    #from sklearn.preprocessing import MinMaxScaler
    #scaling=MinMaxScaler(feature_range=(0,1)).fit(x_train)
    #x_train=scaling.transform(x_train)
    #x_test=scaling.transform(x_test)
    for i in range(np.shape(x_train)[0]):
        x_train[i]=x_train[i]/16

M=np.shape(x_train)[0] # number of datapoints
n=np.shape(x_train)[1] # dimension of each datapoint
pu=random.choices([1.1,1.5,2,2.5,3,4,5,8,10],k=M) # lp-norm of uncertainty set for each datapoint
rho=random.choices([1,2],k=M) # radius of uncertainty set for each datapoint
# ellipses
pu[0]='ell'
pu[2]='ell'
pu[-1]='ell'
rho[0]=3
rho[2]=3
rho[-1]=4
Q1=np.array([[0.8,0.6],[0.6,0.8]])
Q2=np.array([[6,-3],[-3,3]])
Q3=np.array([[5,-3],[-3,4]])
Q4=np.array([[7,3],[5,5]])
Q5=np.array([[5,0],[0,8]])
Q6=np.array([[5,3],[3,5]])
Q7=np.array([[5,1],[1,5]])
Q8=np.array([[9,5],[5,4]])
Q9=np.array([[3,1],[1,2]])
Q10=np.array([[4,4],[4,6]])
# Q=random.choices([Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10],k=3)
Q=np.array([Q6,Q2,Q5])
###############################################################################
### Sub-gradient method
# initialization
init=iter_no+1
w=init0*np.ones((n,init))
b=init0*np.ones((1,1,init))
xi=0*np.ones((M,1,init)) # should be zero so that new iterations=old iterations
mu=init0*np.ones((M,1,init))
u=0.001*np.ones((M,n,init)) # should be non-zero where we have 
lam=init0*np.ones((M,1,init))
v=init0*np.ones((M,1,init))

hu=np.zeros((M,1,init))
dhu=np.zeros((M,n,init))

norm_T=np.zeros((1,init))
gamma=np.zeros((1,init))
alpha=np.zeros((1,init))

L_w=np.zeros((n,1,init))
L_b=np.zeros((1,1,init))
L_xi=np.zeros((M,1,init))
L_mu=np.zeros((M,1,init))
L_u=np.zeros((M,n,init))
L_lam=np.zeros((M,1,init))
L_v=np.zeros((M,1,init))

# Subgradient algorithm iterations
t=time.time()
t2=0
error=1
accuracy=[0]
counter2=0
for k in range(iter_no):
    if k!=0 and (k==iter_no-1 or time.time()-t2>=plot_seconds):
        counter2+=1
        t2=time.time()
        # plotting trajectories
        start_plt=0
        K=np.linspace(start_plt+1,k,num=k-start_plt)
        fig2=plt.figure(counter2,figsize=(5.5,5))
        fig2.subplots_adjust(left=.15,bottom=.16,right=.99,top=.97,wspace=0.2,hspace=0.28)
        ax=fig2.gca()
        ax.grid(True,which='both',linestyle='dotted')
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.rc('font',family='serif')
        plt.rc('xtick',labelsize=10)
        plt.rc('ytick',labelsize=10)
        plt.rc('text',usetex=False) # Because of the slower performance, we recommend only enabling this option at the last minute, once you are ready to make the final plot
        sub1=3 if with_xi else (2 if with_multipliers else 1)
        ax=plt.subplot(sub1,2,1)
        for i in range(n):
            ax.plot(K,w[i,start_plt:k].T,label=(r'$w_%d$' %i))
        ax.plot(K,b[0,0,start_plt:k].T,label=r"$b$",linewidth=1.5)
        ax.set_title(r"$w,b$",y=1)
        ax.set_xlabel('iterations')
        ax.grid(True,which='both',linestyle='dotted')
        #ax.legend(loc='upper left')
        if dataset!=3:
            plt.legend()
        ax=plt.subplot(sub1,2,2)
        u_norm=np.zeros(k)
        jj=0
        for i in range(M):
            if pu[i]==np.inf:
                u_norm=eta_u*np.array([np.linalg.norm(u[i,:,0:k,None],ord=np.inf,axis=0)]).reshape(k)
            elif pu[i]=='ell': # ellipse
                for j in range(k):
                    u_norm[j]=eta_u*np.array([np.dot(np.dot(u[i,:,j,None].T,Q[jj]),u[i,:,j,None])])**.5
                jj+=1
            elif int(pu[i])==pu[i]:
                u_norm=eta_u*np.array([np.linalg.norm(u[i,:,0:k],ord=pu[i],axis=0)]).reshape(k)
            else:
                u_norm=eta_u*np.array([np.sum(np.abs(u[i,:,0:k,None])**pu[i],axis=0)**(1/pu[i])]).reshape(k)
            ax.plot(K,u_norm,linewidth=1.5)#,label=r"$$")
        ax.set_title(r"$\Vert u \Vert$",y=1)
        ax.set_xlabel('iterations')
        ax.grid(True,which='both',linestyle='dotted')
        if with_multipliers:
            ax=plt.subplot(sub1,2,3)
            for i in range(M):
                ax.plot(K,lam[i,0,start_plt:k].T,linewidth=1.5)#,label=r"$\lambda$")     
            ax.set_title(r"$\lambda$",y=1)
            ax.set_xlabel('iterations')
            ax.grid(True,which='both',linestyle='dotted')
            ax=plt.subplot(sub1,2,4)
            for i in range(M):
                ax.plot(K,v[i,0,start_plt:k].T,linewidth=1.5)#,label=r"$v$")
            ax.set_title(r"$v$",y=1)
            ax.set_xlabel('iterations')
            ax.grid(True,which='both',linestyle='dotted')
        if with_xi:
            ax=plt.subplot(sub1,2,5)
            for i in range(M):
                ax.plot(K,xi[i,0,start_plt:k].T,linewidth=1.5)#,label=r"$\xi$ of d1")
            ax.set_title(r"$\xi$",y=1)
            ax.set_xlabel('iterations')
            ax=plt.subplot(sub1,2,6)
            for i in range(M):
                #plt.plot(K,alpha[0,0:k].T)
                plt.plot(K,mu[i,0,0:k].T,linewidth=1.5)#,label=r"$\mu$ of d1")
            ax.set_title(r"$\mu$",y=1)
            ax.set_xlabel('iterations')
            ax.grid(True,which='both',linestyle='dotted')
            #plt.ylim([0,3*alpha[0,k-1]])
        if save_show=='show':
            plt.show()
        elif save_show=='save' and k==iter_no-1:
            fig2.savefig('fig2.png',bbox_inches='tight')
        wd=w[:,k-1]
        bd=b[0,0,k-1]
        #######################################################################
        # plotting datapoints and separating boundary only for 2-dimensional data
        if plotting=='on':
            counter2+=1
            fig3=plt.figure(counter2,figsize=(6,6))
            ax=fig3.gca()
            # uncertainty set
            jj=0
            for i in range(M):
                if pu[i]=='ell':
                    e1=patches.Ellipse((d[i,0],d[i,1]),rho[i]*2/(np.linalg.eig(Q[jj])[0][0]**.5),rho[i]*2/(np.linalg.eig(Q[jj])[0][1]**.5),
                    angle=degrees(atan(np.linalg.eig(Q[jj])[1][1,0]/np.linalg.eig(Q[jj])[1][0,0])),fill=False,
                                  color="magenta" if yy_train[i] else "cyan",linewidth=1.5)
                    ax.add_patch(e1)
                    jj+=1
                    #circle=plt.Circle((d[i,0],d[i,1]),radius=rho[i],color="magenta" if yy_train[i] else "cyan",fill=False)
                    #ax.add_patch(circle)
                else:
                    xx,yy=np.meshgrid(np.linspace(d[i,0]-2*rho[i],d[i,0]+2*rho[i],num=201),
                                      np.linspace(d[i,1]-2*rho[i],d[i,1]+2*rho[i],num=201))
                    if np.isinf(pu[i]):
                        zz=np.maximum(np.abs((xx-d[i,0])/rho[i]),np.abs((yy-d[i,1])/rho[i]))
                        ax.contour(xx,yy,zz,[1],colors=("magenta" if yy_train[i] else "cyan"),linewidths=1.5,zorder=1)
                    else:
                        zz=((np.abs((xx-d[i,0])/rho[i])**pu[i])+(np.abs((yy-d[i,1])/rho[i])**pu[i]))**(1/pu[i])
                        ax.contour(xx,yy,zz,[1],colors=("magenta" if yy_train[i] else "cyan"),linewidths=1.5,zorder=1)
                # support vectors
                if lam[i,0,k,None]!=0: 
                    X=np.arange(np.min(d,axis=0)[0]-5,np.max(d,axis=0)[0]+5,0.01)
                    Y=-(wd[0]/wd[1])*(X-d[i,0]-+u[i,0,k])+d[i,1]+u[i,1,k]
                    plt.plot(X,Y,linewidth=2.5,linestyle='--',color='yellow',zorder=1)
                ax.text(d[i,0],d[i,1],('%d' %(i)))
                plt.scatter(d[i,0],d[i,1],s=50 if yy_train[i] else 55,color="magenta" if yy_train[i] else "cyan",
                            marker="+" if yy_train[i] else (5,1),label=('nominal (label %d)'%y_train[i]) 
                            if i==next(i for i, v in enumerate(yy_train) if v==1) or i==next(i for i, v in enumerate(yy_train) if v==0) else None,zorder=2)
                plt.scatter(d[i,0]+u[i,0,k],d[i,1]+u[i,1,k],s=50 if yy_train[i] else 55,color="blueviolet" if yy_train[i] else "c",
                            marker="+" if yy_train[i] else (5,1),label=('perturbed (label %d)'%y_train[i])
                            if i==next(i for i, v in enumerate(yy_train) if v==1) or i==next(i for i, v in enumerate(yy_train) if v==0) else None,zorder=2)
            X=np.arange(np.min(d,axis=0)[0]-5,np.max(d,axis=0)[0]+5,0.01)
            Y=-bd/wd[1]-(wd[0]/wd[1])*X
            plt.plot(X,Y,linewidth=2.5,linestyle='--',color='lime')
            plt.rc('font',family='serif')
            plt.rc('xtick',labelsize=10)
            plt.rc('ytick',labelsize=10)
            plt.rc('text',usetex=False) # Because of the slower performance, we recommend only enabling this option at the last minute, once you are ready to make the final plot
            plt.legend(loc='upper center',ncol=2)
            #fig3.suptitle("Datapoints and separating hyperplane",fontsize=12,y=.93)
            ax.set_aspect('equal')
            ax.grid(True,which='both',linestyle='dotted',linewidth=1.5)
            #ax.axhline(y=0,color='grey',linestyle='--',linewidth=2)
            #ax.axvline(x=0,color='grey',linestyle='--',linewidth=2)
            ax.axis('equal')
            plt.xlim([min(d[:,0])-1.5*max(rho),max(d[:,0])+1.5*max(rho)])
            plt.ylim([min(d[:,1])-1.5*max(rho),max(d[:,1])+2.5*max(rho)])
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            if save_show=='show':
                plt.show()
            elif save_show=='save' and k==iter_no-1:
                fig3.savefig('fig3.png',bbox_inches='tight')
        print(k+1,"iterations done,",iter_no-k-1,"iterations left")
        print("error =",error,", accuracy =",accuracy[-1],"%")
        elapsed=time.time()-t
        print('elapsed time =',elapsed)
        #print("alpha =",alpha[0,k-1])
        #print("lambda =",lam[:,0,k,None])
    ###########################################################################
    # subgradients
    if iter_type=='new':
        temp=np.sum((np.repeat(lam[:,0,k,None]*y_train,n)[:,None]*((x_train+u[:,:,k,None]).reshape(M*n,1))).reshape(M,n,1),axis=0)
        L_w[:,0,k,None]=eta_w*(gamma_c*w[:,k,None]-temp)
        L_b[0,0,k,None]=eta_b*(-np.sum(np.array([lam[:,0,k,None]*y_train])))
        # uncertainty set
        jj=0
        for i in range(M):
            if pu[i]==np.inf:
                hu[i,0,k,None]=eta_u*np.array([np.linalg.norm(u[i,:,k,None],ord=pu[i])-rho[i]])
                # dhu?
            elif pu[i]=='ell': # ellipse
                hu[i,0,k,None]=eta_u*np.array([np.dot(np.dot(u[i,:,k,None].T,Q[jj]),u[i,:,k,None])-rho[i]**2])
                dhu[i,:,k,None]=eta_u*np.dot((Q[jj]+Q[jj].T),u[i,:,k,None])
                jj+=1
            elif int(pu[i])==pu[i]:
                hu[i,0,k,None]=eta_u*np.array([np.linalg.norm(u[i,:,k],ord=pu[i])**pu[i]-rho[i]**pu[i]])
                dhu[i,:,k,None]=eta_u*u[i,:,k,None]*(abs(u[i,:,k,None])**(pu[i]-2))
            else:
                hu[i,0,k,None]=eta_u*np.array([np.sum(np.abs(u[i,:,k,None])**pu[i])-rho[i]**pu[i]])
                dhu[i,:,k,None]=eta_u*u[i,:,k,None]*(abs(u[i,:,k,None])**(pu[i]-2))
        
        # positive projection for hu and dhu
        if proj==1:
            temp2=dhu[:,:,k,None].copy().reshape(M*n,1)
            temp2[np.repeat(hu[:,0,k,None],n)[:,None]<0]=0
            dhu[:,:,k,None]=temp2.reshape(M,n,1)
            hu[:,0,k,None][hu[:,0,k,None]<0]=0
        L_u[:,:,k,None]=eta_u*(-np.repeat(y_train,n)[:,None]*np.array([w[:,k,None]]*M).reshape(M*n,1)
        -np.repeat(v[:,0,k,None],n)[:,None]*dhu[:,:,k,None].reshape(M*n,1)).reshape(M,n,1)#-beta*hu[i,0,k,None]*dhu[i,:,k,None]
        L_v[:,0,k,None]=eta_v*(-hu[:,0,k,None])#*lam[:,0,k,None]
        if with_xi:
            L_xi[:,0,k,None]=eta_xi*(np.ones((M,1))-lam[:,0,k,None]-mu[:,0,k,None])
            L_mu[:,0,k,None]=eta_mu*(-xi[:,0,k,None])
        else:
            L_xi[:,0,k,None]=np.array([0]*M)[:,None]
            L_mu[:,0,k,None]=np.array([0]*M)[:,None]
        L_lam[:,0,k,None]=eta_lam*(np.ones((M,1))-y_train*(np.sum(np.array([w[:,k,None]]*M)*(x_train[:]+u[:,:,k,None]),axis=1)
        +np.repeat(b[0,0,k],M)[:,None])-eta_u*v[:,0,k,None]*hu[:,0,k,None]-eta_xi*xi[:,0,k,None])#-(beta/2)*hu[i,0,k,None]**2
        norm_T[0,k]=np.linalg.norm(np.concatenate((eta_w*L_w[:,0,k,None],eta_b*L_b[:,0,k,None],eta_u*L_u[:,:,k,None].reshape(n*M,1),
        eta_xi*L_xi[:,0,k,None],eta_lam*L_lam[:,0,k,None],eta_v*L_v[:,0,k,None],eta_mu*L_mu[:,0,k,None]),axis=0))
                
    elif iter_type=='old':
        temp=np.sum(np.array([lam[i,0,k,None]*y_train[i]*(x_train[i]+u[i,:,k,None]) for i in range(M)]),axis=0)
        L_w[:,0,k,None]=eta_w*(gamma_c*w[:,k,None]-temp)
        L_b[0,0,k,None]=eta_b*(-np.sum(np.array([lam[i,0,k,None]*y_train[i] for i in range(M)]),axis=0))
        norm_T[0,k]=eta_w*np.linalg.norm(L_w[:,0,k,None])**2+eta_b*L_b[0,0,k,None]**2
        for i in range(M):
            hu[i,0,k,None]=eta_u*np.array([np.linalg.norm(u[i,:,k,None])**pu-rho[i]**pu])
            dhu[i,:,k,None]=eta_u*np.array([pu*u[i,:,k,None]])
            if proj==1:
                if hu[i,0,k,None]<0:
                    hu[i,0,k,None]=0
                    dhu[i,:,k,None]=np.array([0]*n)[:,None]
            L_u[i,:,k,None]=eta_u*(-y_train[i]*w[:,k,None]-v[i,0,k,None]*dhu[i,:,k,None])#-beta*hu[i,0,k,None]*dhu[i,:,k,None]  
            L_v[i,0,k,None]=eta_v*(-hu[i,0,k,None])#*lam[i,0,k,None]
            if with_xi:
                L_lam[i,0,k,None]=eta_lam*(1-y_train[i]*(np.dot(w[:,k,None].T,(x_train[i]+u[i,:,k,None]))
                +b[0,0,k])-eta_u*v[i,0,k,None]*hu[i,0,k,None]-eta_xi*xi[i,0,k,None])#-(beta/2)*hu[i,0,k,None]**2
                L_xi[i,0,k,None]=eta_xi*(1-lam[i,0,k,None]-mu[i,0,k,None])
                L_mu[i,0,k,None]=eta_mu*(-xi[i,0,k,None])
            else:
                L_lam[i,0,k,None]=eta_lam*(1-y_train[i]*(np.dot(w[:,k,None].T,(x_train[i]+u[i,:,k,None]))
                +b[0,0,k])-eta_u*v[i,0,k,None]*hu[i,0,k,None])#-(beta/2)*hu[i,0,k,None]**2
                L_xi[i,0,k,None]=0
                L_mu[i,0,k,None]=0
            norm_T[0,k]+=eta_xi*L_xi[i,0,k,None]**2+eta_lam*L_lam[i,0,k,None]**2+eta_u*np.linalg.norm(L_u[i,:,k,None])**2
            +eta_v*L_v[i,0,k,None]**2+eta_mu*L_mu[i,0,k,None]**2
        norm_T[0,k]=norm_T[0,k]**.5
    ######################### stepsize
    if step_size=='dim':
        if k>0:
            gamma[0,k]=gamma_num/(k**1)
        alpha[0,k]=gamma[0,k]/norm_T[0,k]
    elif step_size=='con':
        alpha[0,k]=constant_step
    
    w[:,k+1,None]=w[:,k,None]-alpha[0,k]*acc_w*(L_w[:,0,k,None])
    b[0,0,k+1,None]=b[0,0,k,None]-alpha[0,k]*acc_b*(L_b[0,0,k,None])
    lam[:,0,k+1,None]=lam[:,0,k,None]-alpha[0,k]*acc_lam*(-1)*(L_lam[:,0,k,None])
    xi[:,0,k+1,None]=xi[:,0,k,None]-alpha[0,k]*acc_xi*(L_xi[:,0,k,None])
    mu[:,0,k+1,None]=mu[:,0,k,None]-alpha[0,k]*acc_mu*(-1)*(L_mu[:,0,k,None])
    u[:,:,k+1,None]=u[:,:,k,None]-alpha[0,k]*acc_u*(-1)*(L_u[:,:,k,None])
    v[:,0,k+1,None]=v[:,0,k,None]-alpha[0,k]*acc_v*(L_v[:,0,k,None])

    # positive projections
    lam[:,0,k+1,None][lam[:,0,k+1,None]<0]=0 # positive projection for lambda
    if with_xi:
        mu[:,0,k+1,None][mu[:,0,k+1,None]<0]=0 # positive projection for mu
    if proj==0:
        v[:,0,k+1,None][v[:,0,k+1,None]<0]=0  # positive projection for v
 
    # stopping criterion 1
    if 0: #  !!! time consuming, ignore for the time being
        if with_xi:
            error=np.sum(np.array([
            np.linalg.norm(w[:,kk+1,None]-w[:,kk,None])
            +np.linalg.norm(b[0,0,kk+1,None]-b[0,0,kk,None])
            +np.linalg.norm(xi[:,0,kk+1,None]-xi[:,0,kk,None])
            +np.linalg.norm(lam[:,0,kk+1,None]-lam[:,0,kk,None])
            +np.linalg.norm(u[:,:,kk+1,None]-u[:,:,kk,None])
            +np.linalg.norm(v[:,0,kk+1,None]-v[:,0,kk,None])
            +np.linalg.norm(mu[:,0,kk+1,None]-mu[:,0,kk,None])
            for kk in range(k-10,k)]), axis=0)
        else:
            error=np.sum(np.array([
            np.linalg.norm(w[:,kk+1,None]-w[:,kk,None])
            +np.linalg.norm(b[0,0,kk+1,None]-b[0,0,kk,None])
            +np.linalg.norm(lam[:,0,kk+1,None]-lam[:,0,kk,None])
            +np.linalg.norm(u[:,:,kk+1,None]-u[:,:,kk,None])
            +np.linalg.norm(v[:,0,kk+1,None]-v[:,0,kk,None])
            +np.linalg.norm(mu[:,0,kk+1,None]-mu[:,0,kk,None])
            for kk in range(k-10,k)]), axis=0)
    # stopping criterion 2
    if dataset==3:
        res_train=np.array([np.dot(w[:,k-1].T,x_train[i])+b[0,0,k-1,None] for i in range(M)]) # !!! time consuming
        y_res_train=np.array([0]*np.shape(y_train)[0])[:,None]
        y_res_train[res_train>0]=1
        res_err_train=y_res_train-yy_train
        accuracy.append(100*(1-len(res_err_train[np.nonzero(res_err_train)])/np.shape(y_train)[0]))
        if np.sum(accuracy[-5:])>100*5:
            print("break by high accuracy")
            break
###############################################################################
# printing results
elapsed=time.time()-t
print('elapsed time =',elapsed)
print('w =',w[:,k-1])
print('b =',b[0,0,k-1])
if with_xi:
    print('xi_0 =',xi[0,:,-1]) # just for data 1
best_index=np.argmax(accuracy)
print("last error =",error," , best accuracy =",accuracy[best_index],"%")

# testing the model
if dataset==3:
    #w_best=w[:,best_index] # max training accuracy as best 
    #b_best=b[0,best_index] # max training accuracy as best 
    w_best=w[:,k] # last as best
    b_best=b[0,0,k] # last as best
    M_test=np.shape(x_test)[0]
    yy_test=np.array([0]*np.shape(target_test)[0])[:,None]
    for counter,item in enumerate(target_test):
        if item==classified_digit_1:
            yy_test[counter]=1
        elif item==classified_digit_2:
            yy_test[counter]=0
    y_test=yy_test.copy()
    y_test[yy_test==0]=-1
    res_test=[np.dot(w_best.T,x_test[i])+b_best for i in range(M_test)]
    yy_res_test=np.array([0]*np.shape(y_test)[0])[:,None]
    for i in range(M_test):
        if res_test[i]>0:
            yy_res_test[i]=1
    res_err_test=yy_res_test-yy_test
    accuracy_test=100*(1-len(res_err_test[np.nonzero(res_err_test)])/np.shape(y_test)[0])
    print("Test accuracy =",accuracy_test,"%")

    # plotting the perturbed images
    ud=u[:,:,k]
    #for counter,i in enumerate([246,247,248]):
    for i in range(M):
        xx=16*x_train[i].reshape(n1,n2)
        uu=16*ud[i,:].reshape(n1,n2)
        fig4=plt.figure(1)
        fig4.suptitle('Main image      Worst Uncertainty       Perturbed image')
        ax=plt.subplot(8,3,1+i*3)
        ax.imshow(xx,cmap=plt.cm.gray_r,interpolation='nearest',vmin=0,vmax=16)
        ax.axis('off')
        ax=plt.subplot(8,3,2+i*3)
        ax.imshow(uu,cmap=plt.cm.gray_r,interpolation='nearest',vmin=0,vmax=16)
        ax.axis('off')
        ax=plt.subplot(8,3,3+i*3)
        ax.imshow(xx+uu,cmap=plt.cm.gray_r,interpolation='nearest',vmin=0,vmax=16)
        ax.axis('off')
        fig4.savefig('fig4.pdf',bbox_inches='tight')
    
        


