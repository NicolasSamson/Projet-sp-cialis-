import numpy as np



def calculate_speed(vx_r,vx_l,yICR_r,yICR_l,x_ICRv):
    v_x = (vx_r * yICR_l- vx_l *yICR_r)/(yICR_l-yICR_r)

    v_y = (vx_l - vx_r)*x_ICRv / (yICR_l-yICR_r)

    omega_z = -(vx_l-vx_r)/(yICR_l-yICR_r)

    return v_x,v_y,omega_z

def calculate_ICR(vx_l,vx,vx_r,omega_z,vy):
    yICR_l = -(vx_l-vx)/omega_z
    yICR_r = -(vx_r-vx)/omega_z
    x_ICRv = -vy/omega_z 

    return yICR_l,yICR_r,x_ICRv
