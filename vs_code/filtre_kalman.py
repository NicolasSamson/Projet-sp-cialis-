import numpy as np
import matplotlib.pyplot as plt



def filtre_kalman_1(x_k,u_k,phi,gamma):
    """Prédiction du prochain état. """
    x_k_1 = (phi@x_k)+ (gamma@u_k)
    #print(x_k.shape)
    
    return x_k_1

def filtre_kalman_2(p_k,c_v_k_1,phi_k):
    """Calcul de l'estimateur de la variance deuxième etat """
    p_k_1 = phi_k@p_k@phi_k.T+ c_v_k_1
    
    return p_k_1

def filtre_kalman_3(lambda_,x_k_1):
    """prédiction de la mesure"""
    z_k_1_estim = lambda_@x_k_1
    
    return z_k_1_estim

def filtre_kalman_4(z_k_1,z_k_1_estim):
    """calcul erreur entre mesure et prédiction de la mesure (innovation)"""
    r_k_1 = z_k_1-z_k_1_estim
    
    return r_k_1
def filtre_kalman_5(p_k_1,lambda_,c_w_k_1):
    """Calcul le "gain" du terme correctif """
    terme_to_inverse = lambda_@p_k_1@lambda_.T+c_w_k_1
    
    K_k_1 = p_k_1@lambda_.T@np.linalg.inv(terme_to_inverse)
    return K_k_1

def filtre_kalman_6(x_k_1, K_k_1, r_k_1):
    """Corrige l'estimation d'état"""
    x_k_1_corrige = x_k_1 + K_k_1@r_k_1
    
    return x_k_1_corrige

def filtre_kalman_7(K_k_1,lambda_,P_k_1):
    """ Corrige l,estimation de la variance du prochain état"""
    #print(K_k_1.shape[0])
    P_k_1_corrige = (np.identity(K_k_1.shape[0])- (K_k_1@lambda_))@P_k_1
    return P_k_1_corrige


def calcul_prediction_etat(x_k,p_k,u_k,phi,gamma,c_v_k_init):
    """Calcule la prédiction"""
    # 1.0 Estime le prochain état
    x_k_1 = filtre_kalman_1(x_k,u_k,phi,gamma)
    # 2.0 Estimation de la variance du prochain état
    p_k_1 = filtre_kalman_2(p_k,c_v_k_init,phi)
    
    return x_k_1,p_k_1

def calcul_etat_corrige(x_k_1,p_k_1,zk1_mesure,lambda_,c_w_k_init):
    #3.0 Estime la mesure
    z_k_1_estim = filtre_kalman_3(lambda_,x_k_1)


    # 4.0 Calcule l'erreur sur la mesure
    r_k_1 = filtre_kalman_4(zk1_mesure,z_k_1_estim)

    # 5.0 Calcule du gain sur l'erreur
    K_k_1 = filtre_kalman_5(p_k_1,lambda_,c_w_k_init)


    #6.0 Correction de l'état
    x_k_1_corrige = filtre_kalman_6(x_k_1, K_k_1, r_k_1)

    p_k_1_corrige = filtre_kalman_7(K_k_1,lambda_,p_k_1)
    
    return x_k_1_corrige,p_k_1_corrige,K_k_1

def calcul_iteration_filtre_kalman(x_k,p_k,u_k,phi,gamma,c_v_k_init,lambda_,c_w_k_init,mesure=True,):
    # Calcul le state réel du prochain pas
    x_k_1_reel = np.dot(phi,x_k)+ np.dot(gamma,u_k) #+ np.reshape(np.random.normal(0, np.diag(c_v_k_init)),x_k.shape)
    
    
    x_k_1_pred,p_k_1_pred = calcul_prediction_etat(x_k,p_k,u_k,phi,gamma,c_v_k_init)
    #print("x_k_1",x_k_1)
    #print("p_k_1",p_k_1)
    #x_k_1_pred += np.reshape(np.random.normal(0, np.diag(c_v_k_init)),x_k.shape)
    # Si pas de correction alors la correction indique avec 0.
    x_k_1_c = 0
    p_k_1_c = 0
    
    
    if mesure==True:
        test = np.random.normal(0, np.diag(c_w_k_init))
        zk1_mesure = np.dot(lambda_,x_k_1_reel) + np.reshape(np.random.normal(0, np.diag(c_w_k_init)),x_k.shape)
        if zk1_mesure.shape != np.array([[1],[1]]).shape:
            raise ValueError("check_dimension_z")
        
        x_k_1_c,p_k_1_c,K_k_1 = calcul_etat_corrige(x_k_1_pred,p_k_1_pred,zk1_mesure,lambda_,c_w_k_init)
        #print("x_k_corrige",x_k_1)
        #print("p_k_corrige",p_k_1)
        
    
    
    return x_k_1_pred,p_k_1_pred,x_k_1_c,p_k_1_c,x_k_1_reel,K_k_1,zk1_mesure