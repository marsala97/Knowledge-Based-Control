# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:53:09 2020

@author: marsala
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import radians
from pole_cart import PoleCart

# parameters taken from 'Comparison' paper

ALPHA = 0.5 # learning rate
GAMMA = 0.99 # discount factor
force = [-10, 10]  # Netwons

class Q_agent():    
    def __init__(self, FORCE_MAGNITUDE = 3, GAMMA = 0.7, ALPHA = 0.1, EPSILON = 0.3):
        
        # getting a table of bins for discretize states
        self.states_boxes = self.set_discrete_states()        
        self.FORCE_MAGNITUDE = FORCE_MAGNITUDE
        # parameters to tune
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.EPSILON = EPSILON        
        # x, xd, theta, thetad, action
        self.Q = np.zeros((3, 3, 6, 3, 2))

    def set_discrete_states(self):
        '''Set a table of discretize states. Values taken from function getBox of paper "Comparison of RL" '''
        x_box = [-24, -0.8, 0.8, 2.4] 
        xd_box = [-np.inf, -0.5, 0.5, np.inf] 
        theta_box = list(radians([-12, -6, -1, 0, 1, 6, 12]))
        thetad_box = list(radians([-np.inf, -50, 50, np.inf]))
        return [x_box, xd_box, theta_box, thetad_box] 
    
    def discretize(self, state):
        '''This function discretize a state'''  
        discretize_state = np.zeros_like(state)
        for i, box in enumerate(self.states_boxes):
            discretize_state[i] = min(max(0, np.digitize(state[i], box, right=True) -1), len(box)-2)
        return tuple(discretize_state.astype(int)) 
    
    def choose_action(self, state):
        return np.random.randint(2) if (np.random.random() <= self.EPSILON) else np.argmax(self.Q[state])
    
    def update_q(self, state_old, action, reward, state_new):
        # print('before:', self.Q[state_old][action])
        self.Q[state_old][action] += self.ALPHA * (reward + self.GAMMA * np.max(self.Q[state_new]) - self.Q[state_old][action])
        # print('maxQnexstate: ', self.Q[state_new])
        # print('after: ', self.Q[state_old][action])
    
    def get_epsilon(self, trial, max_trials):
        END_EPSILON_DECAYING = 1
        START_EPSILON_DECAYING = max_trials//2
        epsilon_decay_value = 1/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)
        if END_EPSILON_DECAYING >= trial >= START_EPSILON_DECAYING:
            self.EPSILON -= epsilon_decay_value
        return self.EPSILON
       
    
    def get_alpha(self, t):
        pass


init_state = [0, 0.0, 0.01, 0]  # [x, xd, theta, thetad]
dt = 0.001
#dt = 0.02  
env = PoleCart(init_state, dt=dt) 

agent = Q_agent()
#setting of Comparison paper
#env.m = 0.209
#env.m_c = 0.711
#env.l = 0.326
#env.mu_c = 0.0
#env.mu_p = 0.0

max_trials = 1000 # original paper claims 420

trial_durations = []
learn = False
    
for trial in range(max_trials):
    if not learn:
        env.reset() 
        states = []
        control_force = []
        
        new_state = agent.discretize(init_state)
        old_state = new_state
        done = False

        while not done:
            
            agent.get_epsilon(trial, max_trials)
            action = agent.choose_action(old_state)
    
            control_input = agent.FORCE_MAGNITUDE if action == 1 else -agent.FORCE_MAGNITUDE
            control_force.append(control_input)
            env.f = control_input     
            
            new_state = env.step()
            states.append(new_state)         
            done = np.rad2deg(abs(new_state[2])) > 12 or abs(new_state[0]) > 2.4
            reward = -1 if done else 0
    
            new_state = agent.discretize(new_state)
            # print(new_state)
            agent.update_q(old_state, action, reward, new_state)
            old_state = new_state
    
            if len(states) > 100000:
                done = True
                learn = True
                print(f"The agent has learned after {trial} episodes")
                    
        trial_durations.append(len(control_force))
        print(f"Trial #{trial}/{max_trials} — time steps: {len(control_force)}")

            
if learn:
    states = np.array(states)
    fig, axes = plt.subplots(4, 1, figsize=(5, 8))
    fig.canvas.set_window_title(f"Trial {trial}, total time steps: {len(control_force)}")

    color = 'tab:red'
    axes[0].plot(np.arange(len(control_force)), np.rad2deg(states[:, 2]), color=color)
    axes[0].set_xlabel('time step (20 ms)')
    axes[0].set_ylabel('theta [deg]', color=color)
    axes[0].set_ylim([-12, 12])
    axes[0].grid()

    color = 'tab:blue'
    axes[1].plot(np.arange(len(control_force)), states[:, 0], color=color)
    axes[1].set_xlabel('time step (20 ms)')
    axes[1].set_ylabel('x [m]', color=color)
    axes[1].set_ylim([-2.4, 2.4])
    axes[1].grid()

    axes[2].plot(np.arange(len(control_force)), control_force, color='k')
    axes[2].set_xlabel('Time steps  (20 ms)')
    axes[2].set_ylabel('control input  [N]')
    axes[2].set_ylim([-10, 10])
    axes[2].grid()
    
    axes[3].plot(np.arange(1, len(trial_durations) + 1), trial_durations, color='k')
    axes[3].set_xlabel('Trial number')
    axes[3].set_ylabel('Time steps  (20 ms)')
    axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    axes[3].grid()        

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('plots.png', dpi=150)
            
policy = agent.Q
#%% SENSITIVITY ANALYSIS

#init_state = [0, 0.0, 0.01, 0]
#dt = 0.001 
#env = PoleCart(init_state, dt=dt)
#eps_vector = np.linspace(0.2, 5, 100)
#succesfull = []
#
#agent = Q_agent() 
#agent.Q = policy
#for i,eps in enumerate(eps_vector):
#    # perturbed enviroment
#    env.m = env.m * eps
#    env.m_c = env.m_c * eps
#    env.l = env.m_c * eps    
#    
#    env.reset() 
#    states = []
#    control_force = []
#    
#    new_state = agent.discretize(init_state)
#    old_state = new_state
#    done = False
#    #print(f"Trial #{trial}/{max_trials}")
#    while not done:
#        
#        action = np.argmax(agent.Q[old_state])
#
#        control_input = agent.FORCE_MAGNITUDE if action == 1 else -agent.FORCE_MAGNITUDE
#        control_force.append(control_input)
#        env.f = control_input     
#        
#        new_state = env.step()
#        states.append(new_state)         
#        done = np.rad2deg(abs(new_state[2])) > 12 or abs(new_state[0]) > 2.4
#        new_state = agent.discretize(new_state)
#        old_state = new_state
#
#        if len(states) > 100000:
#            done = True  
#            succesfull.append([eps])
#            print(f"The agent has controlled the perturbed cart with eps = {eps}")
#
#if len(succesfull)==0:
#    print('Not really robust')


