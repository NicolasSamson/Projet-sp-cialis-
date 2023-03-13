
import numpy as np 
import matplotlib.pyplot as plt

def pos_from_twist(vitesse,delta_t,theta,x):
    phi = vitesse[2] * delta_t
    if vitesse[2] == 0:
        Q = np.squeeze(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))
        x_abs = Q @ (vitesse[0:2]) * delta_t+ x
    else:
        r = vitesse[0]/vitesse[2]
        x_rel = np.reshape(np.array([[np.abs(r)*np.sin(phi)],[r*(1-np.cos(phi))]]),(2,1))
        #print(x_rel)
        Q = np.squeeze(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))
        #print(x_rel.shape)
        #print(Q.shape)
        theta += phi
        x_abs = Q @ x_rel+ x 
    return x_abs,theta

def test_pos_from_twist():
    r_circle = 100
    t_max = 60 # sec

    lin_speed = np.pi*2*r_circle/t_max
    turning_point = np.pi*2/t_max
    vitesse = np.array([[lin_speed],[0],[turning_point]])

    delta_t = 1 #s 

    theta = 0
    size = 101
    time_v = np.linspace(0,100,size)
    x_array = np.zeros((2,size))


    x = np.array([[0],[0]])
    i=0
    theta = 0
    for time in time_v:

        x_abs,theta= pos_from_twist(vitesse,delta_t,theta,x)
        
        x = x_abs
        print("test",x)
        x_array[:,i] = np.squeeze(x) 
        i+= 1

    plt.scatter(x_array[0,:],x_array[1,:])
    plt.axis('equal')
    plt.show()
    print(x_array)

#test_pos_from_twist()