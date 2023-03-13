from filterpy.kalman  import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import numpy as np
import test_cinematique as tc
### Variable model 1 D 2 états (positions,vitesses)
delta_t = 0.1
scale = 10**-1
sigma_encodeur_droit = 0.05 *scale
sigma_encodeur_gauche = 0.05 * scale
sigma_vitesse_laterale = 0.05 * scale
sigma_process_noise = 1
b = 2 #m
r = 1 #m

r_circle = 100
t_max = 60 # sec
lin_speed = np.pi*2*r_circle/t_max
turning_point = np.pi*2/t_max 

u = np.array([[lin_speed],[0],[turning_point]])



x0 = np.array([[0.],    # vx
    [0.], # vy
    [0]]) # yaw


f = KalmanFilter(dim_x=3, dim_z=3,dim_u =2)
f.x = np.array(x0)  

#state transition matrix Phi
phi = np.array([[0,0.,0.],
                [0.,0,0.],
                [0.,0.,0]])
f.F = phi
# measurement function: alpha
alpha = np.array([[b**2/(r*(b**2+4)),0.,-2*b/(r*(b**2+4))],
                [0,1,0],
                [b**2/(r*(b**2+4)),0.,-2*b/(r*(b**2+4))]])
#alpha = np.array([[1,0,0],[0,1,0],[0,0,1]])
f.H = alpha

# the covariance matrix
f.P = np.array([[1000.,    0.,0.],
                [   0., 1000.,0.],
                [   0., 0., 1000.]])
#  measurement noise cW
f.R = np.array([[sigma_encodeur_droit,    0.,0.],
                [0.,sigma_vitesse_laterale,0.],
                [ 0.,  0., sigma_encodeur_gauche] ])
#  Process noise cW
process_noise = Q_discrete_white_noise(dim=3, dt=delta_t, var=sigma_process_noise)
f.Q = process_noise

# command tansition Matrix
gamma = np.array([[1,    0.,    0],
                [   0., 1., 0],
                [   0., 0, 1.] ])
f.B = gamma




### Variable to store 
size = 600
x0_array = np.zeros(size)
x1_array = np.zeros(size)
z0_array = np.zeros(size)
z1_array = np.zeros(size)

x_0_reel = np.zeros(size)
x_1_reel = np.zeros(size)

x = x0
x_real= x0
time_axe = np.zeros(size)

pos_abso_kalman_array = np.zeros((2,size))
pos_abso_array_reel = np.zeros((2,size))
theta_pos_abso = 0 
pos_abso_reel = np.array([[0],[0]])

theta_pos_abso_kalman = 0 
pos_abso_reel_kalman = np.array([[0],[0]])
for i in range(size):
    x_real = np.dot(phi,x_real)+ np.dot(gamma,u) #+ np.reshape(np.random.normal(0, [sigma_process_noise,sigma_process_noise]),x.shape)
    
    z = np.dot(alpha,x_real) + np.reshape(np.random.normal(0, [sigma_encodeur_droit,0, sigma_encodeur_gauche]),(3,1))
    print(z)
    f.predict(u)
    f.update(z)
    
    time = i*delta_t
    # Calcule la position absolue
    pos_abso_reel,theta_pos_abso = tc.pos_from_twist(x_real,delta_t,theta_pos_abso,pos_abso_reel)
    pos_abso_reel_kalman,theta_pos_abso_kalman = tc.pos_from_twist(f.x,delta_t,theta_pos_abso_kalman,pos_abso_reel_kalman)
    # Stock les variables
    x = f.x
    x0_array[i] = f.x[0]
    x1_array[i] = f.x[2]
    z0_array[i] = f.z[0]
    z1_array[i] = f.z[2]

    x_0_reel[i] = x_real[0]
    x_1_reel[i] = x_real[2]
    time_axe[i] = time

    pos_abso_array_reel[:,i] = np.squeeze(pos_abso_reel)
    pos_abso_kalman_array[:,i] = np.squeeze(pos_abso_reel_kalman)

plt.figure(figsize=(8,8))
plt.plot(pos_abso_array_reel[0,:],pos_abso_array_reel[1,:],label="x_reel" )
plt.scatter(pos_abso_kalman_array[0,:],pos_abso_kalman_array[1,:],label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(8,8))
plt.plot(pos_abso_array_reel[0,:]-pos_abso_array_reel[0,:],pos_abso_array_reel[1,:]-pos_abso_array_reel[1,:],label="x_reel" )
plt.scatter(pos_abso_kalman_array[0,:]-pos_abso_array_reel[0,:],pos_abso_kalman_array[1,:]-pos_abso_array_reel[1,:],label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()
"""
# Affichage graphique
plt.figure(figsize=(15,15))
plt.plot(time_axe,x_0_reel,label="v_x" )
plt.scatter(time_axe,z0_array,label="v_x_mesure" )
plt.scatter(time_axe,x0_array,label="v_x_kalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(15,15))
plt.plot(time_axe,x_0_reel-x_0_reel,label="v_x" )
plt.scatter(time_axe,z0_array-x_0_reel,label="v_x_mesure" )
plt.scatter(time_axe,x0_array-x_0_reel,label="v_x_kalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()


plt.figure(figsize=(15,15))
plt.plot(time_axe,x_1_reel,label="v angulaire" )
#plt.scatter(time_axe,z1_array,label="angulaire_mesure?" )
plt.scatter(time_axe,x1_array,label="v angulaire_kalman")
plt.title("vitesse estimé en  fonction du temps")
plt.ylabel("Vitesse [m/s]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(15,15))
plt.plot(time_axe,x_1_reel-x_1_reel,label="v angulaire" )
#plt.scatter(time_axe,z1_array-x_1_reel,label="v_mesure" )
plt.scatter(time_axe,x1_array-x_1_reel,label="v angulaire_kalman")
plt.title("Erreur sur la vitesse estimé en  fonction du temps")
plt.ylabel("vitesse [m/s]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()
"""