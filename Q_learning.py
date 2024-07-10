
import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt


# Q-learning Agent for MoonWorldEnv
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_space.n)])
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.get_q_value(state, action)
        self.q_table[(state, action)] = self.get_q_value(state, action) + self.learning_rate * td_error

    def choose_action(self, state, env):
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()  # Explore action space
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_space.n)]
            return np.argmax(q_values)  # Exploit learned values

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, file_path):
        np.save(file_path, self.q_table)

    def get_q_table(self):
        return self.q_table

    def print_q_table(self):
        for state_action, value in self.q_table.items():
            print(f"State {state_action[0]} Action {state_action[1]}: Q-value = {value}")

    def get_optimal_policy(self):
        optimal_policy = {}
        for state_action in self.q_table.keys():
            state = state_action[0]
            action = state_action[1]
            if state not in optimal_policy:
                optimal_policy[state] = (action, self.q_table[state_action])
            else:
                if self.q_table[state_action] > optimal_policy[state][1]:
                    optimal_policy[state] = (action, self.q_table[state_action])
        return optimal_policy

    def print_optimal_policy(self):
        optimal_policy = self.get_optimal_policy()
        for state, action_value in optimal_policy.items():
            print(f"State {state}: Best Action = {action_value[0]}, Q-value = {action_value[1]}")
