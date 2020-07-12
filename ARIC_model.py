import numpy as np
import matplotlib.pyplot as plt
from utils import sgn, nn_doubly_connected_forward_pass, plot_fuzzy_memberships, clip


class ARICModel:
    def __init__(self, input_len, input_membership_functions, output_membership_functions, rule_set, o_func, k_func, discount_rate, beta,
                 beta_h, rho, rho_h):
        self.imf = input_membership_functions
        self.omf = output_membership_functions
        self.rule_set = rule_set
        self.o_func = o_func
        self.k_func = k_func

        self.input_len = input_len
        self.h_aen = input_len + 1
        self.h_asn = len(rule_set)
        self.discount_rate = discount_rate
        self.beta = beta
        self.beta_h = beta_h
        self.rho = rho
        self.rho_h = rho_h
        self.it_count = 0

        # Weights
        np.random.seed(123)
        self.A = (np.random.random((input_len + 1, self.h_aen))*2-1)  # n + 1 because we also need a bias neuron
        self.B = (np.random.random(input_len + 1)*2-1)
        self.C = (np.random.random(self.h_aen)*2-1)
        self.D = np.zeros((input_len + 1, self.h_asn))  # initialised later based on io_map
        self.E = (np.random.random(input_len + 1)*2-1)
        self.F = (np.random.random(self.h_asn)*2-1)

        # Internal states
        self.x_prev = np.zeros(self.input_len+1)  # Input state (with bias) at previous time step
        self.y_prev = np.zeros(self.h_aen)  # Values of the hidden neurons of the AEN neural network at previous time step
        self.z_prev = np.zeros(self.h_asn)  # Values of the hidden neurons of the ASN neural network at previous time step
        self.y = np.zeros(self.h_aen)  # Values of the hidden neurons of the AEN neural network
        self.z = np.zeros(self.h_asn)  # Values of the hidden neurons of the ASN neural network

        for i, rule in enumerate(rule_set):
            self.D[rule[0][0], i] = 1.0  # Connect input for condition 1 of rule
            self.D[rule[1][0], i] = 1.0  # Connect input for condition 2 of rule

    def process_state_input(self, x, status="operating"):
        x_with_bias = np.array(list(x)+[1])  # add bias and convert to numpy array to use in vector operations

        # RUN AEN
        internal_reinforcement = self.forward_aen(x_with_bias, status)

        # RUN ASN
        control_input, action_modification = self.forward_asn(x_with_bias)

        # RUN LEARNING STEPS
        self.learn_aen(x_with_bias, internal_reinforcement)
        self.learn_asn(x_with_bias, internal_reinforcement, action_modification)

        # SAVE STATES
        self.x_prev = x_with_bias
        self.y_prev = self.y
        self.z_prev = self.z
        self.it_count += 1
        return control_input

    def forward_aen(self, x, status):
        # ACTION-STATE EVALUATION NETWORK
        self.y, v = nn_doubly_connected_forward_pass(self.A, self.C, self.B, x)
        if self.x_prev is not None:
            _, v_tt = nn_doubly_connected_forward_pass(self.A, self.C, self.B, self.x_prev)
        else:
            v_tt = 0

        if status == "start":
            ir = 0
        elif status == "fail":
            ir = -1 - v_tt
        else:
            ir = self.discount_rate * v - v_tt
        return ir

    def forward_asn(self, x):
        # ACTION SELECTION NETWORK - FUZZY INFERENCE
        w = np.zeros(self.h_asn)
        m = np.zeros(self.h_asn)
        for i, rule in enumerate(self.rule_set):
            j1, j2 = rule[0][0], rule[1][0]  # input indexes
            u1, u2 = self.imf[j1][rule[0][1]], self.imf[j2][rule[1][1]]  # input membership functions
            w[i] = min(self.D[j1, i] * u1.fuzzify(x[j1]), self.D[j2, i] * u2.fuzzify(x[j2]))
            w[i] = clip(w[i], 0, 1)  # Make sure the degree of satisfaction of the rule is between 0 and 1
            m[i] = self.omf[rule[2]].defuzzify(w[i])

        denominator_temp = sum([self.F[i] * w[i] for i in range(self.h_asn)])
        if abs(denominator_temp) < 0.00001:
            u = 0
        else:
            u = sum([self.F[i] * m[i] * w[i] for i in range(self.h_asn)]) / denominator_temp

        # ACTION SELECTION NETWORK - NEURAL NETWORK
        self.z, p = nn_doubly_connected_forward_pass(self.D, self.F, self.E, x)
        p = clip(p, 0, 1)
        up = self.o_func(u, p)
        s = self.k_func(u, up, p)
        return up, s  # NOTE: return u instead of up to bypass the stochastic modification as suggested in the report

    def learn_aen(self, x, ir):
        b_new = self.B + self.beta * ir * self.x_prev
        c_new = self.C + self.beta * ir * self.y_prev
        a_new = np.zeros_like(self.A)
        for i in range(self.h_aen):
            for j in range(self.input_len + 1):
                a_new[j, i] = self.A[j, i] + self.beta_h * ir * self.y_prev[i] * (1 - self.y_prev[i]) * sgn(self.C[i]) * self.x_prev[j]
        self.A = a_new
        self.B = b_new
        self.C = c_new

    def learn_asn(self, x, ir, s):
        e_new = self.E + self.rho * ir * s * self.x_prev
        f_new = self.F + self.rho * ir * s * self.z_prev
        d_new = np.zeros_like(self.D)
        for i in range(self.h_asn):
            for j in range(self.input_len + 1):
                d_new[j, i] = self.D[j, i] + self.rho_h * ir * self.z_prev[i] * (1 - self.z_prev[i]) * sgn(self.F[i]) * s * self.x_prev[j]
        self.D = d_new
        self.E = e_new
        self.F = f_new

    def show_imf(self):
        cols = np.ceil(len(self.imf) / 2)
        plt.figure(figsize=(9, 6))
        for i, imf in enumerate(self.imf):
            ax = plt.subplot(2, cols, i+1)
            plot_fuzzy_memberships(imf, title=f"Input Membership Functions for x{i+1}", axis=ax)
        plt.tight_layout()
        plt.show()

    def show_omf(self):
        plot_fuzzy_memberships(self.omf, title="Output membership functions")
