
from custom_cartpole_delay import CartPoleEnv
import gym
import numpy as np
import scipy.linalg as linalg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

lqr = linalg.solve_continuous_are
scale_factors = []  # scaling factors for sensitivity analysis
t_durations = []  # survival times for sensitivity analysis
for Epsilon in range(100):
    # Model parameters
    Eps = 0.1+Epsilon/100*10  # sensitivity analysis parameter
    scale_factors.append(Eps)
    rho = 1
    env = gym.make('CartPole-v1')
    gravity = 9.8
    masscart = rho*1.0
    masspole = rho*0.1
    total_mass = Eps*(masspole + masscart)
    length = Eps*0.5  # actually half the pole's length
    polemass_length = (masspole * length)
    env.force_mag = 0
    env.tau = 0.02  # step size 20ms

    # Potential energy
    def E(x):
        return 1 / 2 * masspole * (2 * length) ** 2 / 3 * x[3] ** 2 + np.cos(x[2]) * polemass_length * gravity


    def u(x):
        return 1.0 * (E(x) - Ed) * x[3] * np.cos(x[2])

    # State Space

    H = np.array([
        [1, 0, 0, 0],
        [0, total_mass, 0, - polemass_length],
        [0, 0, 1, 0],
        [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
    ])

    Hinv = np.linalg.inv(H)

    A = Hinv @ np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, - polemass_length * gravity, 0]
    ])
    B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))

    # Weight matrices
    Q = np.diag([6, 10.0, 510.0, 10.0])
    R = np.array([[10]])

    # LQR controller - definition of K_cont
    P = lqr(A, B, Q, R)
    Rinv = np.linalg.inv(R)
    K = Rinv @ B.T @ P
    print(K)


    def ulqr(x):
        x1 = np.copy(x)
        x1[2] = np.sin(x1[2])
        return np.dot(K, x1)

    # Initializations
    Ed = E([0, 0, 0, 0])
    init_state = [0, 0.0, 0.01, 0]
    max_trial = 1
    trial_durations = []
    its = 0
    for trial in range(max_trial):
        states = []
        control_force = []
        Eps_done = False
        env.reset()
        observation = init_state

        while not Eps_done:
            observation = np.copy(observation)
            e = np.random.rand(1)
            observation[0] += observation[1] * env.tau
            observation[2] += observation[3] * env.tau
            observation[3] += np.sin(observation[2]) * env.tau * gravity / (length * 2) / 2
            a = u(observation) - 0.3 * observation[0] - 0.1 * observation[1]
            if abs(E(observation) - Ed) < 0.1 and np.cos(observation[2]) > 0.6:
                a = ulqr(observation)
                env.force_mag = min(abs(a[0]), 10)
                print(a)
            else:
                env.force_mag = 10.0
            if a < 0:
                action = 0
            else:
                action = 1
            observation, reward, done, info = env.step(action)
            states.append(observation)
            control_force.append(a)
            its += 1
            Eps_done = np.rad2deg(abs(observation[2])) > 12 or abs(observation[0]) > 2.4 or its > 10000
        trial_durations.append(len(control_force))
        states = np.array(states)
        t_durations.append(its)
        if Eps == 1:
            #plot of unscaled results
            fig, axes = plt.subplots(3, 1, figsize=(5, 8))
            fig.canvas.set_window_title(f"Trial {trial}, total time steps: {len(control_force)}")

            color = 'tab:red'
            axes[0].plot(np.arange(len(control_force)), np.rad2deg(states[:, 2]), color=color)
            axes[0].set_xlabel('time step (20 ms)')
            axes[0].set_ylabel('theta [deg]', color=color)
            axes[0].set_ylim([-1.5, 1.5])
            axes[0].grid()

            color = 'tab:blue'
            axes[1].plot(np.arange(len(control_force)), states[:, 0], color=color)
            axes[1].set_xlabel('time step (20 ms)')
            axes[1].set_ylabel('x [m]', color=color)
            axes[1].set_ylim([-0.05, 0.05])
            axes[1].grid()

            axes[2].plot(np.arange(len(control_force)), control_force, color='k')
            axes[2].set_xlabel('Time steps  (20 ms)')
            axes[2].set_ylabel('control input  [N]')
            axes[2].set_ylim([-0.5, 0.5])
            axes[2].grid()
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig('plots.png', dpi=150)
#plot sensitivity analysis
plt.figure()
plt.plot(scale_factors, t_durations, color='k')
plt.title('Survival times for different system scales')
plt.xlabel('Scale factor')
plt.plot([0.1, 10], [10000, 10000], 'g--')
plt.ylabel('Time steps  (20 ms)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
plt.grid()
plt.savefig('sensitivity_plots.png', dpi=150)