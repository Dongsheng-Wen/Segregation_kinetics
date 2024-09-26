import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import constants


class binary_model:

    def __init__(self,
                D0,
               Q, # 1d array, same length as nd
               E_seg,# 1d array, same length as nd
               T,
               c0,
               d,
               nd, # 1d array
               dt,
               nt,
               sublattice_fraction = 1):

        """
        A class to calculate the segregation profiles as a function of time for a ternary system. 
        The ternary system is represented as A-B-C, where A is solvent, B is solute #1, and C is solute #2.

        ...

        Attributes
        ----------
        D0: 1d array
            diffusion coefficients of solute, has the same length as the number of layers $nd$
        Q: 1d array
            diffusion energy barriers of solute, has the same length as the number of layers $nd$
        E_seg: 1d array
            segregation energies of solute, has the same length as the number of layers $nd$
        c0: floats 
            initial solute concentrations between [0,1]

        T: float
            temperature
        d: float
            inter-layer distance, unit in meter
        nd: int
            total number of layers 
        dt: float
            time increment to simulate the segregation process, unit in second
        nt: int
            number of time steps, nt*dt = total amount of time 

        sublattice_fraction: float
            fraction of the sublattice: c_l
            if c_l = 1, its a complete solid solution. 
            if c_l varies between (0,1), sublattice site fraction is taken into account.
        Methods
        -------
        diffusivity(D0,Q,T):
            Calculate the diffusion coefficient given the diffusion data and temperature T.

        dX1_dt_sublattice: 
            Calculate the concentration as a function of time. This is the main function to calculate the properties. 

        tabulate_calc_res:
            Start the calculation for dX1_dt and tabulate the results into a pandas.DataFrame

        """


        self.D0 = D0
        self.Q = Q
        self.E_seg = E_seg
        self.c0 = c0
        self.T = T
        self.d = d
        self.nd = nd
        self.dt = dt
        self.nt = nt
        self.sublattice_fraction = sublattice_fraction


    def diffusivity(self,D0,Q,T):
        # D0: m^2/s
        # Q: kJ/mole
        # RT: kJ/mole/K * K
        # T: Kelvin
        
        RT = constants.gas_constant *T/1000
        return D0*np.exp(-Q/RT)
        


    def dX1_dt_sublattice(self):
        # E_seg: segregation energy for every layer in the system
        # Q: energy barrier for every layer in the system: jump from i+1 to i layer
        
        
        Delta_G_ij = np.array([self.E_seg[i] - self.E_seg[i+1] for i in range(len(self.E_seg)-1) ] + [0] )
        #print(Delta_G_ij)
        Eb_ji = self.Q
        #print(Eb_ji)
        Dji = self.diffusivity(self.D0,Eb_ji,self.T)# diffusion coefficient 
        #print(Dji)
        prefactor = Dji/self.d**4
        kBT = 8.617333262145e-5 * self.T #eV
        # transfer out
        exp_term_ij = np.exp(Delta_G_ij/kBT)
        #print(exp_term_ij)
        # initial condition, all layers have the same concentrations
        X_layers = np.zeros(self.nd)+self.c0
        #print(X_layers)
        X_layers_vs_t = []
        X_layers_vs_t.append(X_layers)
        t = np.insert(np.cumsum(np.zeros(self.nt)+self.dt),0,0)
        #t_all = 
        for ti in range(self.nt):
            Wij = np.array([self.sublattice_fraction - X_layers[i+1] for i in range(len(X_layers)-1) ] +[self.sublattice_fraction-self.c0]) 
            #print(Wij)
            Wji = (self.sublattice_fraction-X_layers) 
            #print(Wji)
            Xi = np.array([x for x in X_layers])
            Xj = np.array([X_layers[i+1] for i in range(len(X_layers)-1) ] +[self.c0])
            
            Jij = prefactor  * exp_term_ij * Wij * Xi # out (right)
            Jji = prefactor  * Wji * Xj               # in  (left)
            #print(Jij)
            dX1dt = [self.d**2 * (Jji[0] - Jij[0])]
            
            dXidt = [self.d**2 * (Jji[i] + Jij[i-1] - Jji[i-1] - Jij[i] ) for i in range(1,len(Jij)-1) ]
            
            dX_layers = np.array(dX1dt+dXidt+[0])
            #print(len(dX_layers))
            #print(dX_layers)
            X_layers = X_layers + dX_layers*self.dt
            
            
            X_layers_vs_t.append(X_layers)
            
        return np.array(X_layers_vs_t),t


    def tabulate_calc_res(self):

        X_layers_vs_t,t = self.dX1_dt_sublattice()

        self.X_layers_vs_t = X_layers_vs_t
        self.t = t

        self.calc_data = pd.DataFrame()
        self.calc_data['time(s)'] = self.t
        
        for i in range(self.nd):
            self.calc_data[f'x_layer_{i}'] = self.X_layers_vs_t.T[i]




























