from copy import deepcopy
import scipy.constants as constants
import numpy as np
import pandas as pd

class CMSX4_seg_profile:

    def __init__(self,
                D0_Co,Q_Co, # 1d array, same length as nd
                E_seg_Co,   # 1d array, same length as nd
                D0_Cr,Q_Cr, # 1d array, same length as nd
                
                E_seg_Cr,# 1d array, same length as nd
                c0_Co,c0_Cr,
                L_NiCr,L_NiCo,L_NiCrCo,L_CrCo,

                L_AlCr,
                L_AlCrNi,
                L_AlCrCo,
                L_NiCoAl,
                
                Lg_NiCo_Cr,
                Lg_NiCo_Al,
                Lg_Ni_AlCr,
                Lg_Co_AlCr,
                
                T,  # temperature
                d,  # layer distance
                nd, # number of layers
                dt,
                nt):

        """
        A class to calculate the segregation profiles as a function of time for a the CMSX-4 system.
        Longsheng Feng's paper for Co-Cr co-segregation on the SISF. 
        The quaternary system is (Co,Ni)3(Al,Cr) where Co and Ni take the Ni-sublattice sites and the Al-sublattice sites are occupied by Cr and Al. Therefore, the diffusion processes of Co and Cr in different sublattice sites are considered separately. 

        ...

        Attributes
        ----------
        D0_Cr and D0_Co: 1d array
            diffusion coefficients of solute Cr and Co, have the same length as the number of layers $nd$
        Q_Cr and Q_Co : 1d array
            diffusion energy barriers of solute Cr and Co, have the same length as the number of layers $nd$
        E_seg_Cr and E_seg_Co: 1d array
            segregation energies of solute Cr and Co, have the same length as the number of layers $nd$
        c0_Co and c0_Cr: floats 
            initial solute concentrations between [0,1]
        L_NiCr,L_NiCo,L_NiCrCo, L_CrCo, L_AlCr, L_AlCrNi, L_AlCrCo, L_NiCoAl, Lg_NiCo_Cr, Lg_NiCo_Al, Lg_Ni_AlCr, and Lg_Co_AlCr,: floats
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


        self.D0_Co = D0_Co
        self.Q_Co = Q_Co
        self.E_seg_Co = E_seg_Co
        self.D0_Cr = D0_Cr
        self.Q_Cr = Q_Cr

        self.E_seg_Cr = E_seg_Cr
        self.E_seg_Cr2 = deepcopy(E_seg_Cr)
        
        self.c0_Co = c0_Co
        self.c0_Ni = 0.75 - c0_Co
        self.c0_Cr = c0_Cr
        self.c0_Al = 0.25 - c0_Cr 
        self.L_NiCr = L_NiCr
        self.L_NiCo = L_NiCo
        self.L_NiCrCo = L_NiCrCo
        self.L_CrCo = L_CrCo
        
        self.L_AlCr = L_AlCr
        self.L_AlCrNi = L_AlCrNi
        self.L_AlCrCo = L_AlCrCo
        self.L_NiCoAl = L_NiCoAl
        self.Lg_NiCo_Cr = Lg_NiCo_Cr
        self.Lg_NiCo_Al = Lg_NiCo_Al
        self.Lg_Ni_AlCr = Lg_Ni_AlCr
        self.Lg_Co_AlCr = Lg_Co_AlCr

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
        
        kBT = 8.617333262145e-5 * self.T #eV
        
        Eb_ji_Co = self.Q_Co
        Eb_ji_Cr = self.Q_Cr
        Dji_Co = self.diffusivity(self.D0_Co,Eb_ji_Co,self.T)# diffusion coefficient 
        Dji_Cr = self.diffusivity(self.D0_Cr,Eb_ji_Cr,self.T)# diffusion coefficient 

        prefactor_Co = Dji_Co/self.d**4
        prefactor_Cr = Dji_Cr/self.d**4
        
        
        # initial condition, all layers have the same concentrations
        X_layers_Co = np.zeros(self.nd)+self.c0_Co
        X_layers_Cr = np.zeros(self.nd)+self.c0_Cr
        #print(X_layers)
        X_layers_Co_vs_t = []
        X_layers_Co_vs_t.append(X_layers_Co)
        X_layers_Cr_vs_t = []
        X_layers_Cr_vs_t.append(X_layers_Cr)
        t = np.insert(np.cumsum(np.zeros(self.nt)+self.dt),0,0)
        excess_term_Co = np.zeros(self.nd)
        excess_term_Cr = np.zeros(self.nd)
        
        Delta_G_ij_Co_vs_t = []
        Delta_G_ij_Cr_vs_t = []
        # add excess term 
        
        
        X_layers_Ni = 0.75-X_layers_Co
        X_layers_Al = 0.25-X_layers_Cr
        A_term_Cr = self.L_AlCr + X_layers_Co*self.L_AlCrCo + X_layers_Ni*self.L_AlCrNi + 61/3*(X_layers_Co*self.Lg_Co_AlCr + X_layers_Ni*self.Lg_Ni_AlCr) 
        A_term_Cr0 = self.L_AlCr + self.c0_Co*self.L_AlCrCo + self.c0_Ni*self.L_AlCrNi + 61/3*(self.c0_Co*self.Lg_Co_AlCr + self.c0_Ni*self.Lg_Ni_AlCr) 
        
        A_term_Co = self.L_NiCo + X_layers_Cr*self.L_NiCrCo + X_layers_Al*self.L_NiCoAl + 55/9*(X_layers_Cr*self.Lg_NiCo_Cr + X_layers_Al*self.Lg_NiCo_Al)
        A_term_Co0 = self.L_NiCo + self.c0_Cr*self.L_NiCrCo + self.c0_Al*self.L_NiCoAl + 55/9*(self.c0_Cr*self.Lg_NiCo_Cr + self.c0_Al*self.Lg_NiCo_Al)
        
        #X_layers_Ni = 0.75-X_layers_Co
        #X_layers_Al = 0.25-X_layers_Cr
        mu_ex_Cr_x0 = (0.25-2*self.c0_Cr)*A_term_Cr0 
        mu_ex_Cr_x1 = (0.25-2*X_layers_Cr)*A_term_Cr 

        #
        mu_ex_Co_x0 = (0.75-2*self.c0_Co)*A_term_Co0 
        mu_ex_Co_x1 = (0.75-2*X_layers_Co)*A_term_Co 
        
        
        Delta_G_ij_Co_0 = np.array([self.E_seg_Co[i] - self.E_seg_Co[i+1] for i in range(len(self.E_seg_Co)-1) ] + [0] )
        cluster_fractions = X_layers_Co/0.75
        ##########################################################
        # This segregation energy assumes segregation of Cr near Co-clustering but no desegregation of Cr in the Co-lean
        self.E_seg_Cr = cluster_fractions*self.E_seg_Cr2 
        ##########################################################
        Delta_G_ij_Cr_0 = np.array([self.E_seg_Cr[i] - self.E_seg_Cr[i+1] for i in range(len(self.E_seg_Cr)-1) ] + [0] )
        
        non_zero_terms = Delta_G_ij_Cr_0.nonzero()
        for li in non_zero_terms:
            excess_term_Cr[li] = mu_ex_Cr_x1[li] - mu_ex_Cr_x0
            excess_term_Co[li] = mu_ex_Co_x1[li] - mu_ex_Co_x0
            
        Delta_excess_term_Cr = np.hstack([excess_term_Cr[:-1]-excess_term_Cr[1:],excess_term_Cr[-1]])
        Delta_excess_term_Co = np.hstack([excess_term_Co[:-1]-excess_term_Co[1:],excess_term_Co[-1]])
        # Delta_G_ij is Delta_E_s_ij in Eq.1 for Co and Cr
        Delta_G_ij_Co = Delta_G_ij_Co_0 + Delta_excess_term_Co
        Delta_G_ij_Cr = Delta_G_ij_Cr_0 + Delta_excess_term_Cr
        Delta_G_ij_Co_vs_t.append(Delta_G_ij_Co)
        Delta_G_ij_Cr_vs_t.append(Delta_G_ij_Cr)
        
        
        dX_layers_Co_vs_t = []
        dX_layers_Cr_vs_t = []
        for ti in range(self.nt):
            
            exp_term_ij_Co = np.exp(Delta_G_ij_Co/kBT)
            exp_term_ij_Cr = np.exp(Delta_G_ij_Cr/kBT)
            
            Wij_Co = np.array([0.75 - X_layers_Co[i+1] for i in range(len(X_layers_Co)-1) ] +[0.75-self.c0_Co])
            Wij_Cr = np.array([0.25 - X_layers_Cr[i+1] for i in range(len(X_layers_Cr)-1) ] +[0.25-self.c0_Cr])
            
            #print(Wij)
            Wji_Co = 0.75-X_layers_Co
            Wji_Cr = 0.25-X_layers_Cr
            #print(Wji)
            Xi_Co = np.array([x for x in X_layers_Co])
            Xi_Cr = np.array([x for x in X_layers_Cr])
            Xj_Co = np.array([X_layers_Co[i+1] for i in range(len(X_layers_Co)-1) ] +[self.c0_Co])
            Xj_Cr = np.array([X_layers_Cr[i+1] for i in range(len(X_layers_Cr)-1) ] +[self.c0_Cr])
            
            Jij_Co = prefactor_Co  * exp_term_ij_Co * Wij_Co * Xi_Co # out (right)
            Jji_Co = prefactor_Co  * Wji_Co * Xj_Co               # in  (left)
            
            Jij_Cr = prefactor_Cr  * exp_term_ij_Cr * Wij_Cr * Xi_Cr # out (right)
            Jji_Cr = prefactor_Cr  * Wji_Cr * Xj_Cr               # in  (left)
            
            #print(Jij)
            dX1dt_Co = [self.d**2 * (Jji_Co[0] - Jij_Co[0])]
            dX1dt_Cr = [self.d**2 * (Jji_Cr[0] - Jij_Cr[0])]
            
            dXidt_Co = [self.d**2 * (Jji_Co[i] + Jij_Co[i-1] - Jji_Co[i-1] - Jij_Co[i] ) for i in range(1,len(Jij_Co)-1) ]
            dXidt_Cr = [self.d**2 * (Jji_Cr[i] + Jij_Cr[i-1] - Jji_Cr[i-1] - Jij_Cr[i] ) for i in range(1,len(Jij_Cr)-1) ]
            
            dX_layers_Co = np.array(dX1dt_Co+dXidt_Co+[0])
            dX_layers_Cr = np.array(dX1dt_Cr+dXidt_Cr+[0])
            #print(len(dX_layers))
            #print(dX_layers)
            X_layers_Co = X_layers_Co + dX_layers_Co*self.dt
            X_layers_Cr = X_layers_Cr + dX_layers_Cr*self.dt
            dX_layers_Co_vs_t.append(dX_layers_Co*self.dt)
            dX_layers_Cr_vs_t.append(dX_layers_Cr*self.dt)
            
            X_layers_Co_vs_t.append(X_layers_Co)
            X_layers_Cr_vs_t.append(X_layers_Cr)
            
            # excess interaction terms
            X_layers_Ni = 0.75-X_layers_Co
            X_layers_Al = 0.25-X_layers_Cr
            
            A_term_Co = self.L_NiCo + X_layers_Cr*self.L_NiCrCo + X_layers_Al*self.L_NiCoAl + 55/9*(X_layers_Cr*self.Lg_NiCo_Cr + X_layers_Al*self.Lg_NiCo_Al)
            A_term_Co0 = self.L_NiCo + self.c0_Cr*self.L_NiCrCo + self.c0_Al*self.L_NiCoAl + 55/9*(self.c0_Cr*self.Lg_NiCo_Cr + self.c0_Al*self.Lg_NiCo_Al)
            
            A_term_Cr = self.L_AlCr + X_layers_Co*self.L_AlCrCo + X_layers_Ni*self.L_AlCrNi + 61/3*(X_layers_Co*self.Lg_Co_AlCr + X_layers_Ni*self.Lg_Ni_AlCr) 
            A_term_Cr0 = self.L_AlCr + self.c0_Co*self.L_AlCrCo + self.c0_Ni*self.L_AlCrNi + 61/3*(self.c0_Co*self.Lg_Co_AlCr + self.c0_Ni*self.Lg_Ni_AlCr) 
            
            
            mu_ex_Cr_x0 = (0.25-2*self.c0_Cr)*A_term_Cr0 
            mu_ex_Cr_x1 = (0.25-2*X_layers_Cr)*A_term_Cr 

            #
            mu_ex_Co_x0 = (0.75-2*self.c0_Co)*A_term_Co0 
            mu_ex_Co_x1 = (0.75-2*X_layers_Co)*A_term_Co 

            self.E_seg_Cr = cluster_fractions*self.E_seg_Cr2 
            Delta_G_ij_Cr_0 = np.array([self.E_seg_Cr[i] - self.E_seg_Cr[i+1] for i in range(len(self.E_seg_Cr)-1) ] + [0] )
            non_zero_terms = Delta_G_ij_Cr_0.nonzero()
            
            for li in non_zero_terms:
                excess_term_Cr[li] = mu_ex_Cr_x1[li] - mu_ex_Cr_x0
                excess_term_Co[li] = mu_ex_Co_x1[li] - mu_ex_Co_x0

            Delta_excess_term_Cr = np.hstack([excess_term_Cr[:-1]-excess_term_Cr[1:],excess_term_Cr[-1]])
            Delta_excess_term_Co = np.hstack([excess_term_Co[:-1]-excess_term_Co[1:],excess_term_Co[-1]])
            
            
            Delta_G_ij_Co = Delta_G_ij_Co_0 + Delta_excess_term_Co
            Delta_G_ij_Cr = Delta_G_ij_Cr_0 + Delta_excess_term_Cr
            Delta_G_ij_Co_vs_t.append(Delta_G_ij_Co)
            Delta_G_ij_Cr_vs_t.append(Delta_G_ij_Cr)
            
        return np.array(X_layers_Co_vs_t),np.array(X_layers_Cr_vs_t),np.array(Delta_G_ij_Co_vs_t),np.array(Delta_G_ij_Cr_vs_t),np.array(dX_layers_Co_vs_t),np.array(dX_layers_Cr_vs_t),t


    def tabulate_calc_res(self):
        
        X_layers_Co_vs_t,X_layers_Cr_vs_t,Delta_G_ij_Co_vs_t,Delta_G_ij_Cr_vs_t,dX_layers_Co_vs_t,dX_layers_Cr_vs_t,t = self.dX1_dt()
        
        
        
        self.X_layers_Co_vs_t = X_layers_Co_vs_t
        self.X_layers_Cr_vs_t = X_layers_Cr_vs_t
        self.Delta_G_ij_Co_vs_t = Delta_G_ij_Co_vs_t
        self.Delta_G_ij_Cr_vs_t = Delta_G_ij_Cr_vs_t
        self.dX_layers_Co_vs_t = dX_layers_Co_vs_t
        self.dX_layers_Cr_vs_t = dX_layers_Cr_vs_t
        
        self.t = t

        self.calc_data = pd.DataFrame()


        self.calc_data['time(s)'] = self.t
        
        for i in range(self.nd):
            self.calc_data[f'x_Cr_layer_{i}'] = self.X_layers_Cr_vs_t.T[i]

        for i in range(self.nd):
            self.calc_data[f'x_Co_layer_{i}'] = self.X_layers_Co_vs_t.T[i]


