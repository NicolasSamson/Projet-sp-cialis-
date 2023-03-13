from filterpy.kalman  import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import numpy as np

import affichage_graphique as ag
### Variable model 1 D 2 états (positions,vitesses)
delta_t = 0.1 
sigma_laser = 4
sigma_encodeur = 4

sigma_process_noise = 0.01

##
#delta_t = 0.1 
#sigma_laser = 0.5
#sigma_encodeur = 0.5

#sigma_process_noise = 1
##
x0 = np.array([[0.],    # position
    [0.]]) # velocity


f = KalmanFilter(dim_x=2, dim_z=2,dim_u =2)
f.x = np.array(x0)  

#state transition matrix Phi
phi = np.array([[1.,delta_t],
                [0.,0.]])
f.F = phi
# measurement function: alpha
alpha = np.array([[1.,0.],
                [0.,1.]])
f.H = alpha

# the covariance matrix
f.P = np.array([[1000.,    0.],
                [   0., 1000.] ])
#  measurement noise cW
f.R = np.array([[sigma_laser,    0.],
                [   0., sigma_encodeur] ])
#  Process noise cW
process_noise = np.array([[sigma_process_noise,    0.],
                [   0., sigma_process_noise] ])
#process_noise[1,1] = 1*sigma_process_noise
f.Q = process_noise

# command tansition Matrix
gamma = np.array([[0,    0],
                [   0., 1] ])
f.B = gamma

u = np.array([[0.],    # position
            [5.]])   # velocity



### Variable to store 
size = 400
x_array = np.zeros((size,x0.shape[0]))

z_array = np.zeros((size,x0.shape[0]))


x_reel = np.zeros((size,x0.shape[0]))


p_reel = np.zeros((size,x0.shape[0]))

# Sauvegarde le gain à chaque étape
k_reel = np.zeros((size,x0.shape[0]))
# Innovation
y_reel = np.zeros((size,x0.shape[0]))


x = x0
x_real= x0
time_axe = np.zeros(size)

for i in range(size):


    x_real = np.dot(phi,x_real)+ np.dot(gamma,u) #+ np.reshape(np.random.normal(0, [sigma_process_noise,sigma_process_noise*100]),x.shape)
    
    z = np.dot(alpha,x_real) + np.reshape(np.random.normal(0, [np.sqrt(sigma_laser),np.sqrt(sigma_encodeur)]),x.shape)
    
    
    u_now = u 
    #u_now[1] = u_now[1] + np.array([0, np.random.normal(0,sigma_process_noise*1000)])
    #print(u_now)
    f.predict(u_now)
    f.update(z)

    var_estimated = f.P

    time = i*delta_t

    #print(f.K)
    # Stock les variables
    x = f.x
    x_array[i,:] = np.squeeze(f.x)
    #x_array[i,1] = f.x[1]
    z_array[i,:] = np.squeeze(f.z)
    #z_array[i,1] = f.z[1]

    x_reel[i,:] = np.squeeze(x_real)
    #x_reel[i,1] = x_real[1]
    time_axe[i] = time

    #print(np.diag(var_estimated))
    p_reel[i] = np.diag(var_estimated)
    k_reel[i] = np.diag(f.K)
    
    y_reel[i,:] = np.squeeze(f.y)

error_bar_capteur = np.sqrt(np.ones(size)*sigma_laser)


ag.dashboard(time_axe,z_array,x_array,x_reel,p_reel,error_bar_capteur,y_reel,k_reel)