import numpy as np
import numba

class Model(object):

    def __init__(self, steps = 200, skip = 8, dt = 0.01, sigma_obs = 0.1):
        # Number of time steps
        self.steps = steps
        # Skip steps for observations?
        self.skip = skip
        # State vector length
        self.n = 3
        # Length of output vector
        self.m = 2*int(steps / skip)
        # Time step
        self.dt = dt
        
        # Physical constants and whatnot
        pcs = {}
        pcs['r'] = 0.01
        pcs['m'] = 8e3 * (4./3.*np.pi*pcs['r']**3)
        pcs['A'] = np.pi* pcs['r']**2 
        pcs['rho_fluid'] = 1000.
        pcs['g'] = 9.81
        self.pcs = pcs

        

    # Run the simulation with state vector u
    def F(self, u):
        u0, v0, ln_cd = u
        dt = self.dt 
        m = self.pcs['m']
        rho_fluid = self.pcs['rho_fluid']
        A = self.pcs['A']
        g = self.pcs['g']
        
        c = 0.5*rho_fluid*np.exp(ln_cd)*A
        X = np.array([0.,0.,u0,v0])
        vals = [X.copy()]

        def f(X):
            X_prev = X.copy()
            U2 = X_prev[2]**2 + X_prev[3]**2
        
            Fd_scalar = c*U2
            Fd_x = -Fd_scalar*X_prev[2]/(np.sqrt(U2) + 1e-10)
            Fd_y = -Fd_scalar*X_prev[3]/(np.sqrt(U2) + 1e-10)
            Fg_x = 0
            Fg_y = -m*g
            F_x = Fg_x + Fd_x
            F_y = Fg_y + Fd_y

            X[0] += X_prev[2]*dt + 0.5*dt**2*F_x/m
            X[1] += X_prev[3]*dt + 0.5*dt**2*F_y/m
            X[2] += F_x/m*dt
            X[3] += F_y/m*dt

            return X

        
        for i in range(self.steps-1):
            X = f(X)
            vals.append(X.copy())

        return np.array(vals)[::self.skip,:2].ravel()

