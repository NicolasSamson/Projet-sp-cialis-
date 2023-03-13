from filterpy.kalman import ExtendedKalmanFilter
from numpy import dot 
import numpy as np 

class EKF_general(ExtendedKalmanFilter):
    """ This class was created, because the EKF built-in the parent class 
        assume an exponantial linearization for the process matrix. This
        class only change the prediction to the general equations. 

         
    """
    def predict_x(self, u=0):
        raise ValueError("IT CAN'T BE USED BECAUSE OF THE MODIFICATION")


    def predict(self, u, process_function, pf_dx, pf_du, pf_args=(),pf_dx_args=(),pf_du_args=(),noise_on_command=False):
        """
        Predict next state (prior) using the Kalman filter state propagation
        fonction.
        Parameters
        ----------
        u : np.array
            Control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        process_function : function
            Fonction of process to calculate the new state. Takes x and u as pos args and 
            eventually pf_args. 

        pf_dx: 
            Function that calculates the jacobians of the process function by derivation of 
            the process function by x (state_vector).

        pf_du : 
            Function that calculates the jacobians of the process function by derivation of 
            the process function by u (command_vector).

        pf_args : tuple, optional, default (,)
            arguments to be passed into process function after the required state
            variable.
        """
        ##################################
        ##### Validation of the args #####
        ##################################
        
        

        if not isinstance(pf_args, tuple):
            pf_args = (pf_args,)

        if not isinstance(pf_dx_args, tuple):
            pf_dx_args = (pf_dx_args,)

        if not isinstance(pf_du_args, tuple):
            pf_du_args = (pf_du_args,)
        ##################################
        #### Caculate the predictions ####
        ##################################

        

        self.x = process_function(self.x,u,pf_args)

        self.F = pf_dx(self.x,u,pf_dx_args)
        self.B = pf_du(self.x,u,pf_du_args)
        
        if noise_on_command ==False:
            self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        elif noise_on_command == True:
            self.P = dot(self.F, self.P).dot(self.F.T) + dot(self.B, self.Q).dot(self.B.T) 
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)