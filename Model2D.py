import numpy as np
import sympy as sp


class Model2D:
    xdof = 6  # number of states
    udof = 2  # number of inputs

    # Vehicle parameters
    length = 60.  # Vehicle length [m]
    width = 9.    # Vehicle width [m]
    r_b = 14.     # Arm between engine and vehicle's center of mass [m]
    m = 120E3     # Wet mass [kg]
    I = (1/12) * m * length**2 # Moment of inertia [kgm^2]
    T_max = 2.3E6  # Maximum thrust [N]

    # Constraints
    min_throttle = 0.4                # Minimum throttle level [-]
    max_gimbal = np.deg2rad(20)       # Maximum gimbal angle [rad]
    tan_cone = np.tan(np.deg2rad(60)) # Glide cone gradient
    max_pitchRate = np.deg2rad(25)    # Maximum angular velocity [rad/s]

    # Initial conditions
    x_init = np.array([0., 1000.])   # [m]
    v_init = np.array([0., -80.])    # [m/s]
    t_init = np.array([-np.pi/2, 0]) # [rad, rad/s]

    # Boundary conditions
    x_final = np.array([0., 0.])  # [m]
    v_final = np.array([0., 0.])  # [m/s]
    t_final = np.array([0., 0.])  # [rad, rad/s]

    g = 9.80665 # [m/s^2]

    def __init__(self):
        self.x_scale = np.linalg.norm(self.x_init)
        self.v_scale = np.linalg.norm(self.v_init)
        self.t_scale = np.linalg.norm(self.t_init)

        self.x_init = np.concatenate((self.x_init, self.v_init, self.t_init))
        self.x_final = np.concatenate((self.x_final, self.v_final, self.t_final))

    def stateFunc(self, x, u):
        xdot = x[2]
        ydot = x[3]
        vxdot = self.T_max * u[0] * np.sin(u[1] + x[4]) / self.m
        vydot = self.T_max * u[0] * np.cos(u[1] + x[4]) / self.m - self.g
        thetadot = x[5]
        omegadot = -self.T_max * u[0, 0] * np.sin(u[1]) * self.r_b / self.I

        return [xdot, ydot, vxdot, vydot, thetadot, omegadot]

    def initTrajectory(self, N):
        x_init = np.zeros((N, self.xdof))
        u_init = np.zeros((N, self.udof))

        dxdi = (self.x_final - self.x_init) / N

        for i in range(N - 1):
            x_init[i, :] = self.x_init + i * dxdi

        u_init[:, 0] = (1 + self.min_throttle) / 2

        return x_init, u_init



if __name__ == "__main__":
    vehicle = Model2D()
    print(vehicle.x_init)
