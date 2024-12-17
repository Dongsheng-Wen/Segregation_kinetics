

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import constants


class ternary_seg_profile:

    def __init__(self,
                D0_C,Q_C, # 1d array, same length as nd
                E_seg_C,# 1d array, same length as nd
                D0_B,Q_B, # 1d array, same length as nd
                E_seg_B,# 1d array, same length as nd
                c0_C,c0_B,
                L_AB,L_AC,L_ABC,L_BC,
                T,  # temperature
                d,  # layer distance
                nd, # number of layers
                dt,
                nt):

        """
        A class to calculate the segregation profiles as a function of time for a ternary system. 
        The ternary system is represented as A-B-C, where A is solvent, B is solute #1, and C is solute #2.

        ...

        Attributes
        ----------
        D0_B and D0_C: 1d array
            diffusion coefficients of solute B and C, have the same length as the number of layers $nd$
        Q_B and Q_C : 1d array
            diffusion energy barriers of solute B and C, have the same length as the number of layers $nd$
        E_seg_B and E_seg_C: 1d array
            segregation energies of solute B and C, have the same length as the number of layers $nd$
        c0_C and c0_B: floats 
            initial solute concentrations between [0,1]
        L_AB,L_AC,L_ABC, and L_BC: floats
            solute-solute interaction energies


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
        Methods
        -------
        diffusivity(D0,Q,T):
            Calculate the diffusion coefficient given the diffusion data and temperature T.

        dX1_dt: 
            Calculate the concentration as a function of time. This is the main function to calculate the properties. 

        tabulate_calc_res:
            Start the calculation for dX1_dt and tabulate the results into a pandas.DataFrame

        """


        self.D0_C = D0_C
        self.Q_C = Q_C
        self.E_seg_C = E_seg_C
        self.D0_B = D0_B
        self.Q_B = Q_B
        self.E_seg_B = E_seg_B
        self.c0_C = c0_C
        self.c0_B = c0_B
        self.L_AB = L_AB
        self.L_AC = L_AC
        self.L_ABC = L_ABC
        self.L_BC = L_BC
        self.T = T
        self.d = d
        self.nd = nd
        self.dt = dt
        self.nt = nt


    def diffusivity(self,D0,Q,T):
        # D0: m^2/s
        # Q: kJ/mole
        # RT: kJ/mole/K * K
        # T: Kelvin
        
        RT = constants.gas_constant *T/1000
        return D0*np.exp(-Q/RT)
        

    def dX1_dt(self):
        
        # E_seg: segregation energy for every layer in the system
        # Q: energy barrier for every layer in the system: jump from i+1 to i layer
        # A: solvent-A
        # B: solute-B
        # C: solute-C
        kBT = 8.617333262145e-5 * self.T #eV
        
        
        Delta_G_ij_C_0 = np.array([self.E_seg_C[i] - self.E_seg_C[i+1] for i in range(len(self.E_seg_C)-1) ] + [0] )
        Delta_G_ij_B_0 = np.array([self.E_seg_B[i] - self.E_seg_B[i+1] for i in range(len(self.E_seg_B)-1) ] + [0] )
        #print(Delta_G_ij)
        Eb_ji_C = self.Q_C
        Eb_ji_B = self.Q_B
        #print(Eb_ji)
        Dji_C = self.diffusivity(self.D0_C,Eb_ji_C,self.T)# diffusion coefficient 
        Dji_B = self.diffusivity(self.D0_B,Eb_ji_B,self.T)# diffusion coefficient 
        #print(Dji)
        prefactor_C = Dji_C/self.d**4
        prefactor_B = Dji_B/self.d**4
        
        
        #print(exp_term_ij)
        # initial condition, all layers have the same concentrations
        X_layers_C = np.zeros(self.nd)+self.c0_C
        X_layers_B = np.zeros(self.nd)+self.c0_B
        #print(X_layers)
        X_layers_C_vs_t = []
        X_layers_C_vs_t.append(X_layers_C)
        X_layers_B_vs_t = []
        X_layers_B_vs_t.append(X_layers_B)
        t = np.insert(np.cumsum(np.zeros(self.nt)+self.dt),0,0)
        excess_term_C = np.zeros(self.nd)
        excess_term_B = np.zeros(self.nd)
        
        Delta_G_ij_C_vs_t = []
        Delta_G_ij_B_vs_t = []
        
        
        # excess interaction terms
        A_term_B = self.L_AB + X_layers_C*self.L_ABC
        A_term_B0 = self.L_AB + self.c0_C*self.L_ABC
        A_term_C = self.L_AC + X_layers_B*self.L_ABC
        A_term_C0 = self.L_AC + self.c0_B*self.L_ABC
        self.c0_A = 1-self.c0_B-self.c0_C
        X_layers_A = 1-X_layers_B-X_layers_C
        mu_ex_B_x0 = (self.c0_A-self.c0_B)*A_term_B0 + self.c0_C * (self.L_BC - self.L_AC)
        mu_ex_B_x1 = (X_layers_A-X_layers_B)*A_term_B + X_layers_C * (self.L_BC - self.L_AC)

        #
        mu_ex_C_x0 = (self.c0_A-self.c0_C)*A_term_C0 + self.c0_B * (self.L_BC - self.L_AB)
        mu_ex_C_x1 = (X_layers_A-X_layers_C)*A_term_C + X_layers_B * (self.L_BC - self.L_AB)
        
        non_zero_terms = Delta_G_ij_C_0.nonzero()
        for li in non_zero_terms:
            excess_term_B[li] = mu_ex_B_x1[li] - mu_ex_B_x0
            excess_term_C[li] = mu_ex_C_x1[li] - mu_ex_C_x0
            
        Delta_excess_term_B = np.hstack([excess_term_B[:-1]-excess_term_B[1:],excess_term_B[-1]])
        Delta_excess_term_C = np.hstack([excess_term_C[:-1]-excess_term_C[1:],excess_term_C[-1]])

        
        Delta_G_ij_C = Delta_G_ij_C_0 + Delta_excess_term_C
        Delta_G_ij_B = Delta_G_ij_B_0 + Delta_excess_term_B
        Delta_G_ij_C_vs_t.append(Delta_G_ij_C)
        Delta_G_ij_B_vs_t.append(Delta_G_ij_B)
        
        
        dX_layers_C_vs_t = []
        dX_layers_B_vs_t = []
        for ti in range(self.nt):
            
            exp_term_ij_C = np.exp(Delta_G_ij_C/kBT)
            exp_term_ij_B = np.exp(Delta_G_ij_B/kBT)
            
            Wij_C = np.array([1 - X_layers_C[i+1] for i in range(len(X_layers_C)-1) ] +[1-self.c0_C])
            Wij_B = np.array([1 - X_layers_B[i+1] for i in range(len(X_layers_B)-1) ] +[1-self.c0_B])
            
            #print(Wij)
            Wji_C = 1-X_layers_C
            Wji_B = 1-X_layers_B
            #print(Wji)
            Xi_C = np.array([x for x in X_layers_C])
            Xi_B = np.array([x for x in X_layers_B])
            Xj_C = np.array([X_layers_C[i+1] for i in range(len(X_layers_C)-1) ] +[self.c0_C])
            Xj_B = np.array([X_layers_B[i+1] for i in range(len(X_layers_B)-1) ] +[self.c0_B])
            
            Jij_C = prefactor_C  * exp_term_ij_C * Wij_C * Xi_C # out (right)
            Jji_C = prefactor_C  * Wji_C * Xj_C               # in  (left)
            
            Jij_B = prefactor_B  * exp_term_ij_B * Wij_B * Xi_B # out (right)
            Jji_B = prefactor_B  * Wji_B * Xj_B               # in  (left)
            
            #print(Jij)
            dX1dt_C = [self.d**2 * (Jji_C[0] - Jij_C[0])]
            dX1dt_B = [self.d**2 * (Jji_B[0] - Jij_B[0])]
            
            dXidt_C = [self.d**2 * (Jji_C[i] + Jij_C[i-1] - Jji_C[i-1] - Jij_C[i] ) for i in range(1,len(Jij_C)-1) ]
            dXidt_B = [self.d**2 * (Jji_B[i] + Jij_B[i-1] - Jji_B[i-1] - Jij_B[i] ) for i in range(1,len(Jij_B)-1) ]
            
            dX_layers_C = np.array(dX1dt_C+dXidt_C+[0])
            dX_layers_B = np.array(dX1dt_B+dXidt_B+[0])
            #print(len(dX_layers))
            #print(dX_layers)
            X_layers_C = X_layers_C + dX_layers_C*self.dt
            X_layers_B = X_layers_B + dX_layers_B*self.dt
            dX_layers_C_vs_t.append(dX_layers_C*self.dt)
            dX_layers_B_vs_t.append(dX_layers_B*self.dt)
            
            X_layers_C_vs_t.append(X_layers_C)
            X_layers_B_vs_t.append(X_layers_B)
           
            # excess interaction terms
            A_term_B = self.L_AB + X_layers_C*self.L_ABC
            A_term_B0 = self.L_AB + self.c0_C*self.L_ABC
            A_term_C = self.L_AC + X_layers_B*self.L_ABC
            A_term_C0 = self.L_AC + self.c0_B*self.L_ABC
            
            X_layers_A = 1-X_layers_B-X_layers_C
            mu_ex_B_x0 = (self.c0_A-self.c0_B)*A_term_B0 + self.c0_C * (self.L_BC - self.L_AC)
            mu_ex_B_x1 = (X_layers_A-X_layers_B)*A_term_B + X_layers_C * (self.L_BC - self.L_AC)

            #
            mu_ex_C_x0 = (self.c0_A-self.c0_C)*A_term_C0 + self.c0_B * (self.L_BC - self.L_AB)
            mu_ex_C_x1 = (X_layers_A-X_layers_C)*A_term_C + X_layers_B * (self.L_BC - self.L_AB)

            
            non_zero_terms = Delta_G_ij_C_0.nonzero()
            for li in non_zero_terms:
                excess_term_B[li] = mu_ex_B_x1[li] - mu_ex_B_x0
                excess_term_C[li] = mu_ex_C_x1[li] - mu_ex_C_x0
                
            Delta_excess_term_B = np.hstack([excess_term_B[:-1]-excess_term_B[1:],excess_term_B[-1]])
            Delta_excess_term_C = np.hstack([excess_term_C[:-1]-excess_term_C[1:],excess_term_C[-1]])

            
            Delta_G_ij_C = Delta_G_ij_C_0 + Delta_excess_term_C
            Delta_G_ij_B = Delta_G_ij_B_0 + Delta_excess_term_B
            Delta_G_ij_C_vs_t.append(Delta_G_ij_C)
            Delta_G_ij_B_vs_t.append(Delta_G_ij_B)
            
        return np.array(X_layers_C_vs_t),np.array(X_layers_B_vs_t),np.array(Delta_G_ij_C_vs_t),np.array(Delta_G_ij_B_vs_t),np.array(dX_layers_C_vs_t),np.array(dX_layers_B_vs_t),t


    def tabulate_calc_res(self):

        X_layers_C_vs_t,X_layers_B_vs_t,Delta_G_ij_C_vs_t,Delta_G_ij_B_vs_t,dX_layers_C_vs_t,dX_layers_B_vs_t,t = self.dX1_dt()

        self.X_layers_C_vs_t = X_layers_C_vs_t
        self.X_layers_B_vs_t = X_layers_B_vs_t
        self.Delta_G_ij_C_vs_t = Delta_G_ij_C_vs_t
        self.Delta_G_ij_B_vs_t = Delta_G_ij_B_vs_t
        self.dX_layers_C_vs_t = dX_layers_C_vs_t
        self.dX_layers_B_vs_t = dX_layers_B_vs_t
        self.t = t

        self.calc_data = pd.DataFrame()


        self.calc_data['time(s)'] = self.t
        
        for i in range(self.nd):
            self.calc_data[f'x_B_layer_{i}'] = self.X_layers_B_vs_t.T[i]

        for i in range(self.nd):
            self.calc_data[f'x_C_layer_{i}'] = self.X_layers_C_vs_t.T[i]


