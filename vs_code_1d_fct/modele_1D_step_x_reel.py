from filterpy.kalman  import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import numpy as np

import affichage_graphique as ag
### Variable model 1 D 2 états (positions,vitesses)
delta_t = 0.1 
sigma_laser = 0.0001
sigma_encodeur = 0.4#0.0000002

sigma_process_noise = 1


affichage = 1
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
                [0.,0]])
f.F = phi
# measurement function: alpha
alpha = np.array([[1.,0.],
                [0,1.]])
f.H = alpha

# the covariance matrix
f.P = np.array([[1000.,    0.],
                [   0., 1000.] ])
#  measurement noise cW
f.R = np.array([[sigma_laser,    0.],
                [   0., sigma_encodeur] ])
#  Process noise cv
process_noise = np.array([[1,    0.],
                [   0., 1] ])

#Q_discrete_white_noise(dim=2, dt=delta_t, var=sigma_process_noise)
f.Q = process_noise

# command transition Matrix
gamma = np.array([[0,    0.],
                [   0., 1] ])
f.B = gamma

u_ = np.array([[0.],    # position
            [5.]])   # velocity




### Variable to store 
size = 1000
x_array = np.zeros((size,x0.shape[0]))

z_array = np.zeros((size,x0.shape[0]))


x_reel = np.zeros((size,x0.shape[0]))

x_predict = np.zeros((size,x0.shape[0]))


p_reel = np.zeros((size,x0.shape[0]))

# Sauvegarde le gain à chaque étape
k_reel = np.zeros((size,x0.shape[0]))
# Innovation
y_reel = np.zeros((size,x0.shape[0]))

# Added to x 
x_correction = np.zeros((size,x0.shape[0]))


x = x0
x_real= x0
time_axe = np.zeros(size)

def selection_cas(cas,i,delta_t,x_real):
    if cas == 1: # Step sur la position seulement 
        step_x = np.array([[-10],[0]])
        step_x_2 = np.array([[0],[0]])

    elif cas == 2: # Step seulement sur la vitesse
        # Le véhicule ne bouge plus immobile.  
        # Les résultats sont cohérents, car on contrôle en vitesse
        # La vitesse précédente n'est pas enregistré dans le modèle. 
        # Le modèle, dans sa prédiction, assume que le véhicule avance. 
        # Conséquemment, le filtre de Kalman donne toujours une vitesse d'avance 
        # de 5m/s peut importe la modulation de la réalitées. 

        # Par contre, la position est enregistrée dans le modèle ainsi que la mesure de la vitesse.
        # Donc après un certain temps, l'Kalman réussi à comprendre qu'il y a eu un drop de
        # vitesse dans sa position, mais pas dans sa vitesse. La position suit donc la bonne 
        # valeur avec une erreur de biais, mais pas la vitesse. Plus le step est grand, plus le biais est grand. 
        #   
        # 
        # 
        deceleration = -1 #m/s^2
        
        
        step_x = np.array([[0],[(i - 25/delta_t) * deceleration * delta_t]])
        
        if step_x[1]<-5:
            step_x[1] = -5
        #print(step_x)
        
    
    elif cas == 3: # Step seulement sur la vitesse
        # Même chose qu'au cas 2. sauf que cosntante de temps = 60. Donc environ
        # La même constante de temps. 

        step_x = np.array([[0],[-0.5]])
        step_x_2 = np.array([[0],[-0.5]])
    return step_x
 
flag = 1
for i in range(size):

    x_real = np.dot(phi,x_real)+ np.dot(gamma,u_) #+ np.reshape(np.random.normal(0, [sigma_process_noise,sigma_process_noise]),x.shape)
    
    
    step_x = selection_cas(2,i,delta_t,x_real)

    if (i*delta_t) >=38:
        x_real+= step_x
        #if flag == 30:
        #    flag  = 1
        #flag = 1
        if flag ==1:
            flag = 0
            f.P[1,1] = 100

    elif (i*delta_t) >=25:
        x_real+= step_x
        print(step_x)

        if flag == 1:
            flag=0
            f.P[1,1] = 100
            print(f.P)
        
    
    
        
    #elif (i*delta_t) >25:
    #    x_real+= step_x_2
    z = np.dot(alpha,x_real) + np.reshape(np.random.normal(0, [np.sqrt(sigma_laser),np.sqrt(sigma_encodeur)]),x.shape)
    #f.predict(u)
    f.predict(u=u_)
    x_predict[i,:] = np.squeeze(f.x)
    f.update(z)

    var_estimated = f.P

    time = i*delta_t
    # Stock les variables
    x = f.x
    var_estimated = f.P

    time = i*delta_t

    #print(f.K)
    # Stock les variables
    x = f.x
    x_array[i,:] = np.squeeze(f.x)
    #x_array[i,1] = f.x[1]
    z_array[i,:] = np.squeeze(f.z)
    #z_array[i,1] = f.z[1]
    x_correction[i,:] = np.squeeze(f.K @ f.y)

    x_reel[i,:] = np.squeeze(x_real)
    #x_reel[i,1] = x_real[1]
    time_axe[i] = time

    #print(np.diag(var_estimated))
    p_reel[i] = np.diag(var_estimated)
    k_reel[i] = np.diag(f.K)
    
    y_reel[i,:] = np.squeeze(f.y)

print(f.P)
error_bar_capteur = np.sqrt(np.ones(size)*sigma_laser)


ag.dashboard(time_axe,z_array,x_array,x_reel,p_reel,error_bar_capteur,y_reel,k_reel,x_predict,x_correction,affichage)