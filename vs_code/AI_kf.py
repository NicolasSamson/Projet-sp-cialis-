from filterpy.kalman  import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import numpy as np

### Variable model 1 D 2 états (positions,vitesses)
delta_t = 0.1 
sigma_laser = 5
sigma_encodeur = 5

sigma_process_noise = 1

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
process_noise = Q_discrete_white_noise(dim=2, dt=delta_t, var=sigma_process_noise)
f.Q = process_noise

# command tansition Matrix
gamma = np.array([[1,    0.],
                [   0., 1] ])
f.B = gamma

u = np.array([[0.],    # position
            [5.]])   # velocity



### Variable to store 
size = 800
x0_array = np.zeros(size)
x1_array = np.zeros(size)
z0_array = np.zeros(size)
z1_array = np.zeros(size)

x_0_reel = np.zeros(size)
x_1_reel = np.zeros(size)

x = x0
x_real= x0
time_axe = np.zeros(size)

for i in range(size):


    x_real = np.dot(phi,x_real)+ np.dot(gamma,u) #+ np.reshape(np.random.normal(0, [sigma_process_noise,sigma_process_noise]),x.shape)
    
    z = np.dot(alpha,x_real) + np.reshape(np.random.normal(0, [sigma_laser,sigma_encodeur]),x.shape)
    f.predict(u)
    f.update(z)
    
    time = i*delta_t
    # Stock les variables
    x = f.x
    x0_array[i] = f.x[0]
    x1_array[i] = f.x[1]
    z0_array[i] = f.z[0]
    z1_array[i] = f.z[1]

    x_0_reel[i] = x_real[0]
    x_1_reel[i] = x_real[1]
    time_axe[i] = time

# Affichage graphique
plt.figure(figsize=(15,15))
plt.plot(time_axe,x_0_reel,label="x_reel" )
plt.scatter(time_axe,z0_array,label="x_mesure" )
plt.scatter(time_axe,x0_array,label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(15,15))
plt.plot(time_axe,x_0_reel-x_0_reel,label="x_reel" )
plt.scatter(time_axe,z0_array-x_0_reel,label="x_mesure" )
plt.scatter(time_axe,x0_array-x_0_reel,label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()


plt.figure(figsize=(15,15))
plt.plot(time_axe,x_1_reel,label="v_reel" )
plt.scatter(time_axe,z1_array,label="v_mesure" )
plt.scatter(time_axe,x1_array,label="vKalman")
plt.title("vitesse estimé en  fonction du temps")
plt.ylabel("Vitesse [m/s]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(15,15))
plt.plot(time_axe,x_1_reel-x_1_reel,label="v_reel" )
plt.scatter(time_axe,z1_array-x_1_reel,label="v_mesure" )
plt.scatter(time_axe,x1_array-x_1_reel,label="vKalman")
plt.title("Erreur sur la vitesse estimé en  fonction du temps")
plt.ylabel("vitesse [m/s]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()