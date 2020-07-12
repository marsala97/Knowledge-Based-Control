import numpy as np
import matplotlib.pyplot as plt


def sgn(x):
    return 1 if x >= 0 else -1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip(value, vmin, vmax):
    return min(max(value, vmin), vmax)


def nn_doubly_connected_forward_pass(w_inp_hid, w_hid_out, w_inp_out, x):
    i_count = w_inp_hid.shape[0]
    h_count = w_inp_hid.shape[1]
    h = np.zeros(h_count)  # values at hidden neurons
    for i in range(h_count):
        weighted_sum = 0
        for j in range(i_count):
            weighted_sum += w_inp_hid[j, i] * x[j]
        h[i] = sigmoid(weighted_sum)
    # value of output neuron
    o = sum([w_inp_out[i] * x[i] for i in range(i_count)]) + sum([w_hid_out[i] * h[i] for i in range(h_count)])
    return h, sigmoid(o)


def plot_fuzzy_memberships(membership_functions, title="", axis=None):
    most_positive = 0
    most_negative = 0
    for func in membership_functions.values():
        most_positive = max(most_positive, func.x_max)
        most_negative = min(most_negative, func.x_min)
    margin = (most_positive - most_negative) * 0.1
    most_positive += margin
    most_negative -= margin
    x = np.arange(most_negative, most_positive, (most_positive - most_negative) * 0.0005)

    for key, func in membership_functions.items():
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = func.fuzzify(x[i])
        if axis:
            axis.plot(x, y, label=key)
        else:
            plt.plot(x, y, label=key)

    if axis:
        # axis.legend()
        axis.grid()
        # axis.title.set_text(title)
        return axis
    else:
        # plt.legend()
        plt.grid()
        # plt.suptitle(title)
        plt.tight_layout(pad=1.5)
        plt.show()
