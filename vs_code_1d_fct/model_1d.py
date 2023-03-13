import numpy as np
import matplotlib.pyplot as plt
from filtre_kalman import *
# État initial
x_k_initial=np.array([[0],[0]])
x_reel = x_k_initial
print("x_k_initial",x_k_initial.shape)

# Paramètre supplémentaire
Rroue = 0.25 #m

var_laser = 1
var_encodeur = 1

pas_de_temps = 1 # s

var_propag_x = 2
var_propag_speed = 2

etalo_laser = 1#0.005 # v/m
etalo_encodeur = 1#1/5000 *1/(2*np.pi*Rroue) *1/pas_de_temps*(2*np.pi) # 1m/5000pulse *1tr/(2 pi r [m]) / pas_de_temps [s] * 2 PI [rad]/[tr]

# Commande Initial
u_k_initial = np.array([[2],[0.1]])
#print(u_k_initial.shape)

# Matrice de transformation de la commande
gamma = np.array([[1,0],[0,1]])

# Matrice de la propagation d'état
pas_iteration_temps = 0.1

phi = np.array([[1,pas_iteration_temps],[0,0]])

# Matrice de covariance
initial_mat_var = 20
p_k_initial = np.array([[initial_mat_var,0],[initial_mat_var,0]])

# Incertitude sur la propagation d'état
c_v_k_init = np.array([[var_propag_x,0],[0,var_propag_speed]])

lambda_ = np.array([[etalo_laser,0],[0,etalo_encodeur]])

c_w_k_init = np.array([[var_laser,0],[0,var_encodeur]])



x_k= x_k_initial
p_k = p_k_initial
u_k = u_k_initial.T

# Définit le vecteur de commande

u_k_array = np.array([[0,0,0,0],[0,5,10,20]])

size = 10000
pos_changement_commande = np.array([0,50,100,150])

# Vecteur pour sauver les données

x_k_array = np.zeros((size,2))
p_k_array = np.zeros((size,2))

u_k_array_save = np.zeros((size,2))
x_reel_save = np.zeros((size,2))

x_k_pred_array = np.zeros((size,2))
p_k_pred_array = np.zeros((size,2))
x_k_1_reel_array = np.zeros((size,2))

K_k_1_array = np.zeros((size,2))
z_array = np.zeros((size,2))
# Exécution du code.

print("x_k",x_k)
print("p_k",p_k)
print("u_k",u_k)


x_k =np.array([[0],[0]])
n_iter =0
numero_commande = 0


while x_k[0][0] < 150 and n_iter<60:
    #print(x_k)
    #print(pos_changement_commande[numero_commande])
    
    print(x_k[0][0])
    if x_k[0][0] >= pos_changement_commande[numero_commande] or n_iter==0:
        numero_commande+=1
        u_k = np.array([u_k_array[:,numero_commande]]).T
        
    
    
        
    
    x_k_1_pred,p_k_1_pred,x_k,p_k,x_k_1_reel,K_k_1,zk1_mesure = calcul_iteration_filtre_kalman(x_k,p_k,u_k,phi,gamma,c_v_k_init,lambda_,c_w_k_init)
    #x_k = x_k_1_pred
    #p_k = p_k_1_pred
    
    if x_k.shape != np.array([[1],[1]]).shape:
        raise ValueError("check_dimension_x")
    
    # Log data
    
    x_k_array[n_iter,:] = np.squeeze(x_k)
    #print("test",x_k_1_pred.shape)
    x_k_pred_array[n_iter,:] = np.squeeze(x_k_1_pred)
    x_k_1_reel_array[n_iter,:] = np.squeeze(x_k_1_reel)
    
    p_k_array[n_iter,:] = np.diag(p_k)
    p_k_pred_array[n_iter,:] = np.diag(p_k_1_pred)
    
    u_k_array_save[n_iter,:] = np.squeeze(u_k)
    
    K_k_1_array[n_iter,:] = np.diag(K_k_1)
    # ACtualise les variables 
    z_array[n_iter,:] = np.squeeze(zk1_mesure)
    
    #print(p_k)
    #print(x_k_array)
    n_iter += 1



# Extract logged_data
start = 0 
x_k_array = x_k_array[start:n_iter,:]
x_k_pred_array = x_k_pred_array[start:n_iter,:]
x_k_1_reel_array = x_k_1_reel_array[start:n_iter,:]

p_k_array = p_k_array[start:n_iter,:]
p_k_pred_array= p_k_pred_array[start:n_iter,:]

u_k_array_save = u_k_array_save[start:n_iter,:]

K_k_1_array = K_k_1_array[start:n_iter,:]

z_array= z_array[start:n_iter,:]

# Affichage graphique
dumb_x_axe = np.array([i for i in range(x_k_array.shape[0])])
#print(dumb_x_axe)

# Create time vectors
time_axe = np.array([i for i in range(x_k_array.shape[0])])*pas_iteration_temps

# Affichage graphique
plt.figure(figsize=(15,15))
plt.plot(time_axe,x_k_1_reel_array[:,0],label="x_reel" )
plt.scatter(time_axe,z_array[:,0],label="x_mesure" )
plt.errorbar(time_axe,x_k_array[:,0],label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

plt.figure(figsize=(15,15))
plt.plot(time_axe,x_k_1_reel_array[:,0]-x_k_1_reel_array[:,0],label="x_reel" )
plt.scatter(time_axe,z_array[:,0]-x_k_1_reel_array[:,0],label="x_mesure" )
plt.errorbar(time_axe,x_k_array[:,0]-x_k_1_reel_array[:,0],label="xKalman")
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 
plt.show()

"""
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
plt.errorbar(time_axe,x_k_pred_array[:,0],yerr=p_k_pred_array[:,0],label="prediction")

plt.errorbar(time_axe,x_k_array[:,0],yerr=p_k_array[:,0],label="corrigée")

plt.plot(time_axe,x_k_1_reel_array[:,0],label="x_reel" )
plt.scatter(time_axe,z_array[:,0],label="x_mesure" )
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 



# Graph vitesse
plt.subplot(2,2,2)
plt.errorbar(time_axe,x_k_pred_array[:,1],yerr=p_k_pred_array[:,1],label="prediction")

plt.errorbar(time_axe,x_k_array[:,1],yerr=p_k_array[:,1],label="corrigée")

plt.plot(time_axe,x_k_1_reel_array[:,1],label="x_reel" )
plt.scatter(time_axe,z_array[:,1],label="vitesse_mesure" )
plt.title("Vitesse estimée en  fonction du temps")
plt.ylabel("Vitesse [m/s]")
plt.xlabel("temps [s]")
plt.legend()

# GRaphique de l'erreur position

plt.subplot(2,2,3)
plt.errorbar(time_axe,x_k_pred_array[:,0]-x_k_1_reel_array[:,0] ,yerr=p_k_pred_array[:,0],label="prediction")

plt.errorbar(time_axe,x_k_array[:,0]-x_k_1_reel_array[:,0],yerr=p_k_array[:,0],label="corrigée")

plt.plot(time_axe,x_k_1_reel_array[:,0]-x_k_1_reel_array[:,0],label="x_reel" )

plt.title("Erreur en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 

# GRaphique de l'erreur position

plt.subplot(2,2,4)
plt.errorbar(time_axe,x_k_pred_array[:,1]-x_k_1_reel_array[:,1] ,yerr=p_k_pred_array[:,1],label="prediction")

plt.errorbar(time_axe,x_k_array[:,1]-x_k_1_reel_array[:,1],yerr=p_k_array[:,1],label="corrigée")

plt.plot(time_axe,x_k_1_reel_array[:,1]-x_k_1_reel_array[:,1],label="x_reel" )

plt.title("Erreur en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.legend() 

plt.show()

plt.plot(time_axe,K_k_1_array[:,0])
plt.plot(time_axe,K_k_1_array[:,1])
plt.show()

plt.figure(figsize=(15,15))

plt.subplot(2,3,1)
plt.plot(dumb_x_axe*pas_de_temps,x_k_array[:,0])
plt.title("Position estimé en  fonction du temps")
plt.ylabel("Position [m]")
plt.xlabel("temps [s]")
plt.subplot(2,3,4)
plt.plot(dumb_x_axe*pas_de_temps,p_k_array[:,0])
plt.title("Variance sur la position estimé en fonction du temps")
plt.ylabel("variance [m]")
plt.xlabel("temps [s]")


# Graph la vitesse 
plt.subplot(2,3,2)
plt.plot(dumb_x_axe*pas_de_temps,x_k_array[:,1])
plt.title("Vitesse estimée en fonction du temps")
plt.ylabel("Vitesse [m/s]")
plt.xlabel("temps [s]")
plt.subplot(2,3,5)
plt.plot(dumb_x_axe*pas_de_temps,p_k_array[:,1])
plt.title("Variance sur la Vitesse estimée \n  en fonction du temps")
plt.ylabel("variance [m]")
plt.xlabel("temps [s]")

# GRaph la comande 
plt.subplot(2,3,3)
plt.plot(dumb_x_axe*pas_de_temps,u_k_array_save[:,0])
plt.title("Commande en position \n en fonction du temps")
plt.ylabel("Position [m/s]")
plt.xlabel("temps [s]")
plt.subplot(2,3,6)
plt.plot(dumb_x_axe*pas_de_temps,u_k_array_save[:,1])
plt.title("commande en vitesse \n en fonction du temps")
plt.ylabel("Vitesse [m/s]")
plt.xlabel("temps [s]")

plt.show()
"""