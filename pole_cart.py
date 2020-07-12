import numpy as np
from numpy import sin, cos
from utils import sgn


class PoleCart:
    def __init__(self, init_state, dt=0.0001):
        self.init_state = init_state
        self.state = init_state  # [x, xd, theta, thetad]
        self.f = 0
        # Parameters from Barto et al
        self.m = 0.1
        self.m_c = 1.0
        self.l = 0.5
        self.mu_c = 0.0005
        self.mu_p = 0.00002
        self.dt = dt

    def step(self):
        # The following 4 lines perform a Heun integration step
        fval = self.derivative(self.state)
        y1 = self.state + self.dt * fval
        fval += self.derivative(y1)
        self.state = self.state + 0.5 * self.dt * fval
        return self.state

    def derivative(self, y):  # computed yd
        g = 9.81    # gravity constant

        # Compute thetadd by splitting the computation of the numerator and denominator for clarity
        temp_num = g * sin(y[2]) + cos(y[2]) * (
                -self.f - self.m * self.l * y[3] * y[3] * sin(y[2]) + self.mu_c * sgn(y[1])) / (
                           self.mu_c + self.m) - self.mu_p * y[3] / (self.m * self.l)
        temp_denom = self.l * (4 / 3 - (self.m * cos(y[2]) ** 2) / (self.mu_c + self.m))
        thetadd = temp_num / temp_denom

        # Compute xdd by splitting the computation of the numerator and denominator for clarity
        temp_num = self.f + self.m * self.l * (y[3] ** 2 * sin(y[2]) - thetadd * cos(y[2])) - self.mu_c * sgn(y[1])
        temp_denom = self.mu_c + self.m
        xdd = temp_num / temp_denom

        return np.array([y[1], xdd, y[3], thetadd])

    def reset(self):
        self.state = self.init_state

    def sim_and_plot(self, duration=2.0):
        import matplotlib.pyplot as plt
        t_arr = np.arange(0, duration, self.dt)
        self.f = 0.15
        states = np.array([self.step() for _ in range(len(t_arr))])
        fig, ax = plt.subplots()
        ax.plot(t_arr, states[:, 0], 'r-', label="x")
        ax.plot(t_arr, states[:, 1], 'r--', label="x_dot")
        ax.set_ylabel("distance or linear speed [m or m/s]", color='r')
        ax.set_xlabel("Time [s]")
        ax_deg = ax.twinx()
        ax_deg.plot(t_arr, np.rad2deg(states[:, 2]), 'k-', label="theta")
        ax_deg.plot(t_arr, np.rad2deg(states[:, 3]), 'k--', label="theta_dot")
        ax_deg.set_ylabel("angle or angular speed [deg or deg/s]")
        ax_deg.grid()
        fig.legend()
        plt.show()

