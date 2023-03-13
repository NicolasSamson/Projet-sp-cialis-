import matplotlib.pyplot as plt
import numpy as np


def dashboard(time_axe,z_array,x_array,x_reel,p_reel,error_bar_capteur,y_reel,k_reel,x_predict,x_correction,affichage):

    # Affichage graphique

    plt.figure(figsize=(15,15))



    # Subplot x centrée
    plt.subplot(2,3,1)

    plt.scatter(time_axe,z_array[:,0]-x_reel[:,0],label="x_mesure",c="green")
    plt.scatter(time_axe,x_array[:,0]-x_reel[:,0],label="xKalman",c="orange")

    plt.errorbar(time_axe,z_array[:,0]-x_reel[:,0],yerr=3*error_bar_capteur,fmt="o",c="green")
    plt.errorbar(time_axe,x_array[:,0]-x_reel[:,0],yerr=3*np.sqrt(p_reel[:,0]),fmt="o",c="orange")

    plt.plot(time_axe,x_reel[:,0]-x_reel[:,0],label="x_reel" )
    
    plt.plot(time_axe,x_predict[:,0]-x_reel[:,0],label="x_predict",c="red")

    plt.title("Erreur sur la position en  fonction du temps")
    plt.ylabel("Erreur [m]")
    plt.xlabel("temps [s]")
    plt.legend() 

    # Plot l'innovation
    plt.subplot(2,3,2)
    plt.plot(time_axe,y_reel[:,0],label="x_innovation" )
    plt.plot(time_axe,y_reel[:,1],label="vitesse innovation" )

    plt.title("Innovation en fonction du temps")
    plt.ylabel("Innovation [SI]")
    plt.xlabel("temps [s]")
    plt.legend() 

    plt.subplot(2,3,4)
    #plt.errorbar(time_axe,x_array[:,1],yerr=3*np.sqrt(p_reel[:,1]),fmt="o",c="orange")

    plt.plot(time_axe,x_reel[:,1],label="v_reel" )
    plt.scatter(time_axe,z_array[:,1],label="v_mesure" )
    plt.scatter(time_axe,x_array[:,1],label="vKalman")
    plt.plot(time_axe,x_predict[:,1],label="x_predict",c="red")

    #plt.errorbar(time_axe,z_array[:,1],yerr=3*error_bar_capteur,fmt="o",c="green")
    
    plt.title("vitesse estimé en  fonction du temps")
    plt.ylabel("Vitesse [m/s]")
    plt.xlabel("temps [s]")
    plt.legend() 


    plt.subplot(2,3,5)
    plt.scatter(time_axe,k_reel[:,0],label="gain inovation x" )
    plt.scatter(time_axe,k_reel[:,1],label="gain_innovation speed" )

    plt.title("Gain des états en  fonction du temps")
    plt.ylabel("Gain [SI]")
    plt.xlabel("temps [s]")
    plt.legend() 

    plt.subplot(2,3,3)
    plt.scatter(time_axe,p_reel[:,0],label="std  x" )
    plt.scatter(time_axe,p_reel[:,1],label="std speed" )

    plt.title("STD des états en  fonction du temps")
    plt.ylabel("STD [SI]")
    plt.xlabel("temps [s]")
    plt.legend() 



    plt.subplot(2,3,6)

    plt.scatter(time_axe,z_array[:,0],label="x_mesure",c="green")
    plt.scatter(time_axe,x_array[:,0],label="xKalman",c="orange")

    plt.errorbar(time_axe,z_array[:,0],yerr=3*error_bar_capteur,fmt="o",c="green")
    plt.errorbar(time_axe,x_array[:,0],yerr=3*np.sqrt(p_reel[:,0]),fmt="o",c="orange")

    plt.plot(time_axe,x_reel[:,0],label="x_reel" )
    plt.plot(time_axe,x_predict[:,0],label="x_predict",c="red")

    plt.title("Position en  fonction du temps")
    plt.ylabel("Position [m]")
    plt.xlabel("temps [s]")
    plt.legend() 

    if affichage == 1:
        plt.show()
    else:
        plt.savefig("dashboard.png")
    #plt.show()




    #plt.show()

    plt.subplot(2,2,1)
    plt.plot(time_axe, x_correction[:,0],label="x_correction")
    
    
    plt.title("Correction de la position en  fonction du temps")
    plt.ylabel("Correction [m]")
    plt.xlabel("temps [s]")
    plt.legend() 

    plt.subplot(2,2,2)
    plt.plot(time_axe, x_correction[:,1],color="orange",label="x_correction")
    
    
    plt.title("Correction de la vitesse en  fonction du temps")
    plt.ylabel("Correction [m/s]")
    plt.xlabel("temps [s]")
    plt.legend() 

    plt.subplot(2,2,3)

    plt.scatter(time_axe,z_array[:,0]-x_reel[:,0],label="x_mesure",c="green")
    plt.scatter(time_axe,x_array[:,0]-x_reel[:,0],label="xKalman",c="orange")

    plt.errorbar(time_axe,z_array[:,0]-x_reel[:,0],yerr=3*error_bar_capteur,fmt="o",c="green")
    plt.errorbar(time_axe,x_array[:,0]-x_reel[:,0],yerr=3*np.sqrt(p_reel[:,0]),fmt="o",c="orange")

    plt.plot(time_axe,x_reel[:,0]-x_reel[:,0],label="x_reel" )
    
    plt.plot(time_axe,x_predict[:,0]-x_reel[:,0],label="x_predict",c="red")

    plt.title("Erreur sur la position en  fonction du temps")
    plt.ylabel("Erreur [m]")
    plt.xlabel("temps [s]")
    plt.legend() 

    plt.subplot(2,2,4)

    
    plt.plot(time_axe,x_reel[:,1]-x_reel[:,1],label="v_reel" )
    plt.scatter(time_axe,z_array[:,1]-x_reel[:,1],label="v_mesure" )
    plt.scatter(time_axe,x_array[:,1]-x_reel[:,1],label="vKalman")

    plt.plot(time_axe,x_predict[:,1]-x_reel[:,1],label="v_predict",c="red")
    

    plt.errorbar(time_axe,z_array[:,1],yerr=3*error_bar_capteur,fmt="o",c="green")
    
    plt.title("Erreur de la vitesse estimé en  fonction du temps")
    plt.ylabel("Erreur [m/s]")
    plt.xlabel("temps [s]")
    plt.legend()
    plt.show()
    """
    # PLot position fct(t)

    """