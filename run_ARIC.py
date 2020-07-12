from fuzzy_set import FuzzySet
from ARIC_model import ARICModel
import membership_functions
from utils import sgn, plot_fuzzy_memberships
import matplotlib.pyplot as plt
import numpy as np
import gym
from pole_cart import PoleCart

# Rule set: logic statements of the form IF [C1] AND [C2] THEN [C3]
# The conditions (C1 and C2) are composed of the index of the input
# variable and a fuzzy set. The conclusions (C3) are simply fuzzy sets.
rule_set = [
    [(2, "NE"), (3, "NE"), "NL"],
    [(2, "NE"), (3, "ZE"), "NS"],
    [(2, "NE"), (3, "PO"), "0"],
    [(2, "ZE"), (3, "NE"), "NM"],
    [(2, "ZE"), (3, "ZE"), "0"],
    [(2, "ZE"), (3, "PO"), "PM"],
    [(2, "PO"), (3, "NE"), "0"],
    [(2, "PO"), (3, "ZE"), "PS"],
    [(2, "PO"), (3, "PO"), "PL"],
    [(0, "NE"), (1, "NE"), "NS"],
    [(0, "VS"), (1, "NE"), "NVS"],
    [(0, "VS"), (1, "PO"), "PVS"],
    [(0, "PO"), (1, "PO"), "PS"]
]


def o_func_cart_pole(v, p):
    import random
    q = (p + 1) / 2

    return v if random.random() < q else -v


def k_func_cart_pole(v, v_prev, p):
    return 1 - p if sgn(v) != sgn(v_prev) else -p


aric = ARICModel(4, membership_functions.berenji_quantization_inputs, membership_functions.berenji_quantization_outputs, rule_set,
                 o_func_cart_pole, k_func_cart_pole, discount_rate=0.9, beta=0.2, beta_h=0.05, rho=1.0, rho_h=0.2)

# <START DEBUG CODE>
# aric.show_imf()
# aric.show_omf()
# env = PoleCart([0, 0.0, 0.0, 0], dt=0.001)
# env.sim_and_plot(10)
# exit()
# <END DEBUG CODE>

max_trials = 1000
show_pole_cart = False
show_plots = True
use_gym = False  # Gym does not work very well, prefer the own implementation
init_state = [0, 0.0, 0.01, 0]  # [x, xd, theta, thetad]
dt = 0.02

if use_gym:
    env = gym.make('CartPole-v1')
else:
    env = PoleCart(init_state, dt=dt)

print("\nNOTE: a plot window opens after each trial showing the progress. To proceed to the next trial, CLOSE the window.\n")

trial_durations = []
for trial in range(max_trials):
    env.reset()
    states = []
    control_force = []

    if use_gym:
        env.env.force_mag = 0
        env.env.tau = dt
        env.env.state = init_state

    done = False
    print(f"Trial #{trial}/{max_trials}")
    while not done:
        if use_gym:
            if show_pole_cart:
                env.render()
            y, reward, done, info = env.step(1)
            control_input = aric.process_state_input(y, status="operating" if not done else "fail")
            env.env.force_mag = control_input
        else:
            y = env.step()  # comes out as [x, x_dot, theta, theta_dot]
            done = np.rad2deg(abs(y[2])) > 12 or abs(y[0]) > 2.4
            control_input = 0.1 * aric.process_state_input(y, status="operating" if not done else "fail")
            env.f = control_input

        states.append(y)  # save state for plotting
        control_force.append(control_input)
        if len(states) > 100000:
            done = True
    trial_durations.append(len(control_force))

    print(f'Finished trial {trial} at t={len(control_force)}s')

    if show_plots:  # and len(trial_durations) % 50 == 0:
        states = np.array(states)
        fig, axes = plt.subplots(2, 2, figsize=(10, 4))
        fig.canvas.set_window_title(f"Trial {trial}, total time steps: {len(control_force)}")

        color = 'tab:red'
        axes[0, 0].plot(np.arange(len(control_force)), np.rad2deg(states[:, 2]), color=color)
        axes[0, 0].set_xlabel('Time steps  (20 ms)')
        axes[0, 0].set_ylabel('theta  [deg]', color=color)
        axes[0, 0].set_ylim([-12, 12])
        axes[0, 0].grid()

        color = 'tab:blue'
        axes[0, 1].plot(np.arange(len(control_force)), states[:, 0], color=color)
        axes[0, 1].set_xlabel('Time steps  (20 ms)')
        axes[0, 1].set_ylabel('x  [m]', color=color)
        axes[0, 1].set_ylim([-2.4, 2.4])
        axes[0, 1].grid()

        axes[1, 0].plot(np.arange(len(control_force)), control_force, color='k')
        axes[1, 0].set_xlabel('Time steps  (20 ms)')
        axes[1, 0].set_ylabel('control input  [N]')
        axes[1, 0].set_ylim([-10, 10])
        axes[1, 0].grid()

        axes[1, 1].plot(np.arange(1, len(trial_durations) + 1), trial_durations, color='k')
        axes[1, 1].set_xlabel('Trial number')
        axes[1, 1].set_ylabel('Time steps  (20 ms)')
        axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
        axes[1, 1].grid()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

plt.show()
