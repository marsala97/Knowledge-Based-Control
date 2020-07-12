from fuzzy_set import FuzzySet
from ARIC_model import ARICModel
import quantizations
from utils import sgn, plot_fuzzy_memberships
import matplotlib.pyplot as plt
import numpy as np
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
use_gym = False
init_state = [0, 0.0, 0.01, 0]  # [x, xd, theta, thetad]
dt = 0.02

env = PoleCart(init_state, dt=dt)

# STEP 1: TRAIN CONTROLLER IN NORMAL CONDITIONS

aric = ARICModel(4, quantizations.berenji_quantization_inputs, quantizations.berenji_quantization_outputs, rule_set,
                 o_func_cart_pole, k_func_cart_pole, discount_rate=0.9, beta=0.2, beta_h=0.05, rho=1.0, rho_h=0.2)

trial_durations = []
for i in range(200):
    done = False
    env.reset()
    its = 0
    while not done:
        y = env.step()  # comes out as [x, x_dot, theta, theta_dot]
        done = np.rad2deg(abs(y[2])) > 12 or abs(y[0]) > 2.4
        control_input = 0.1 * aric.process_state_input(y, status="operating" if not done else "fail")
        env.f = control_input

        its += 1

        if its > 500000:
            done = True
    print(f"Trial #{i}/{200} done in {its} steps")
    if its > 500000:
        break

# STEP 2: TEST CONTROLLER IN DIFFERENT CONDITIONS

trial_durations = []
scale_factors = [0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5]
for scale_factor in scale_factors:
    print(f"Sensitivity test for factor {scale_factor}")
    env.l = 0.5 * scale_factor
    env.m = 0.1 * scale_factor
    env.m_c = 1.0 * scale_factor

    env.reset()
    its = 0
    done = False
    while not done:
        y = env.step()  # comes out as [x, x_dot, theta, theta_dot]
        control_input = 0.1 * aric.process_state_input(y, status="operating" if not done else "fail")
        env.f = control_input
        its += 1
        done = np.rad2deg(abs(y[2])) > 12 or abs(y[0]) > 2.4 or its > 200000
    print(f'\tFailed in {its} steps')
    trial_durations.append(its)


plt.plot(scale_factors, trial_durations, color='k')

plt.xlabel('Scale factor')
plt.plot([0, 200], [100000, 100000], 'g--')
plt.ylabel('Time steps  (20 ms)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
plt.grid()
# plt.legend()
plt.show()
