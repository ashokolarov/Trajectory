import casadi
import numpy as np
from Model2D import Model2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    vehicle = Model2D()

    T = 18.
    dt = 0.1
    N = int(T / dt)

    opti = casadi.Opti()

    x = opti.variable(N, vehicle.xdof)
    u = opti.variable(N, vehicle.udof)

    x_init, u_init = vehicle.initTrajectory(N)

    opti.set_initial(x, x_init)
    opti.set_initial(u, u_init)

    x[0, :] = vehicle.x_init
    x[N - 1, :] = vehicle.x_final
    u[N - 1, 1] = 0

    for i in range(0, N - 1):
        x_dot = vehicle.stateFunc(x[i, :], u[i, :])
        x_dot_next = vehicle.stateFunc(x[i + 1, :], u[i + 1, :])

        xk12 = opti.variable(vehicle.xdof)

        for j in range(vehicle.xdof):
            opti.subject_to(xk12[j] == 0.5 * (x[i, j] + x[i + 1, j]) + (dt / 8) * (x_dot[j] - x_dot_next[j]))

        uk12 = opti.variable(vehicle.udof)

        for j in range(vehicle.udof):
            opti.subject_to(uk12[j] == u[i, j] + (u[i, j] + u[i + 1, j]) / 2)

        x_dot_half = vehicle.stateFunc(xk12, uk12)

        opti.subject_to(x[i + 1, 0] - x[i, 0] == (x_dot[0] + 4 * x_dot_half[0] + x_dot_next[0]) * (dt / 6))
        opti.subject_to(x[i + 1, 1] - x[i, 1] == (x_dot[1] + 4 * x_dot_half[1] + x_dot_next[1]) * (dt / 6))

        opti.subject_to(x[i + 1, 2] - x[i, 2] == (x_dot[2] + 4 * x_dot_half[2] + x_dot_next[2]) * (dt / 6))
        opti.subject_to(x[i + 1, 3] - x[i, 3] == (x_dot[3] + 4 * x_dot_half[3] + x_dot_next[3]) * (dt / 6))

        opti.subject_to(x[i + 1, 4] - x[i, 4] == (x_dot[4] + 4 * x_dot_half[4] + x_dot_next[4]) * (dt / 6))
        opti.subject_to(x[i + 1, 5] - x[i, 5] == (x_dot[5] + 4 * x_dot_half[5] + x_dot_next[5]) * (dt / 6))

        opti.subject_to(x[i, 0] / x[i, 1] <= vehicle.tan_cone)
        opti.subject_to(opti.bounded(-vehicle.max_pitchRate, x[i, 5], vehicle.max_pitchRate))

    for i in range(0, N):
        opti.subject_to(opti.bounded(vehicle.min_throttle, u[i, 0], 1))
        opti.subject_to(opti.bounded(-vehicle.max_gimbal, u[i, 1], vehicle.max_gimbal))

    opti.minimize(casadi.sumsqr(u[:, 0]) + casadi.sumsqr(u[:, 1]) + 2 * casadi.sumsqr(x[:, 5]))

    opti.solver('ipopt')
    sol = opti.solve()

    x = sol.value(x)
    u = sol.value(u)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 12), gridspec_kw={'width_ratios': [30, 1, 1]})

    ax1.set_xlim(-250, 200)
    ax1.set_ylim(-60, 1000)

    ax1.set_xlabel('Crossrange [m]', fontsize=20)
    ax1.set_ylabel('Altitude [m]', fontsize=20)

    ax1.plot(x[:, 0], x[:, 1], linestyle='--', color='orange', linewidth=3)

    rocket = ax1.plot([], [], color='blue', linewidth=6)[0]
    thrust = ax1.plot([], [], color='red', linewidth=4)[0]

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_ticks([])

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Throttle [-]', fontsize=14)

    throttle = ax2.bar([], [], color='red')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_xaxis().set_ticks([])

    mg = np.rad2deg(vehicle.max_gimbal)

    ax3.set_ylim(-mg, mg)
    ax3.set_ylabel(r'Gimbal angle [$^{\circ}$]', fontsize=14)

    gimbal = ax3.bar([], [], color='green')

    plt.tight_layout()


    def animate(i):
        rx = [x[i, 0] + vehicle.length / 2 * np.sin(x[i, 4]), x[i, 0] - vehicle.length / 2 * np.sin(x[i, 4])]
        ry = [x[i, 1] + vehicle.length / 2 * np.cos(x[i, 4]), x[i, 1] - vehicle.length / 2 * np.cos(x[i, 4])]

        rocket.set_data(rx, ry)

        flame_length = u[i, 0] * 50

        gx = [rx[1], rx[1] + flame_length * np.sin(-u[i, 1] - x[i, 4])]
        gy = [ry[1], ry[1] - flame_length * np.cos(-u[i, 1] - x[i, 4])]

        thrust.set_data(gx, gy)

        global throttle
        throttle.remove()
        throttle = ax2.bar([0], [u[i, 0]], color='red')

        global gimbal
        gimbal.remove()
        gimbal = ax3.bar([0], [np.rad2deg(u[i, 1])], color='green')


    anim = FuncAnimation(fig, animate, frames=np.arange(0, N - 1, 1), interval=dt * 100)
    anim.save('starship.gif', writer='imagemagick', fps=30)
    #plt.show()
