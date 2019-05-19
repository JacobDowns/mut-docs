import numpy as np
import numba


@numba.jit
def run_sim(init_vel):
    """ 
    Simulate a projectile with quadratic drag. Returns the x position
    where the projectile hits the ground. 
    """

    # Initial x velocity
    x_init = init_vel[0]
    # Initial y velocity
    y_init = init_vel[1]
    # State vector with position and velocity [x, y, v_x, v_y] 
    X = np.array([0.,0.,u0,v0])
    
    t = 0
    while t<500.:
        X += rhs(X_0)
        t += dt

        if X[1] <= 0.0
          break

    return X[1]


# RHS of ODE
@numba.jit 
def rhs(X_prev,c):
    Uw = np.random.randn(2)*sigma_w
    U2 = (X_prev[2]-Uw[0])**2 + (X_prev[3]-Uw[1])**2

    Fd_scalar = c*U2
    Fd_x = -Fd_scalar*(X_prev[2]-Uw[0])/(np.sqrt(U2) + 1e-10)
    Fd_y = -Fd_scalar*(X_prev[3]-Uw[1])/(np.sqrt(U2) + 1e-10)
    Fg_x = 0
    Fg_y = -m*g
    F_x = Fg_x + Fd_x
    F_y = Fg_y + Fd_y

    return np.array([X_prev[2]*dt + 0.5*dt**2*F_x/m, X_prev[3]*dt + 0.5*dt**2*F_y/m, F_x/m*dt, F_y/m*dt])
