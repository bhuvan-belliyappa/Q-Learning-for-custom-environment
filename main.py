import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from padm_env import MoonWorldEnv
from Q_learning import QLearningAgent

def main():
    env = MoonWorldEnv()
    agent = QLearningAgent(env.action_space)

    episodes = 10000
    for episode in range(episodes):
        obs, _ = env.reset()
        state = tuple(obs)
        total_reward = 0

        while True:
            action = agent.choose_action(state, env)
            next_obs, reward, done, _ = env.step(action)
            next_state = tuple(next_obs)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()  # Decay epsilon after each episode

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # Save Q-table
    agent.save_q_table('q_table.npy')

    # Print Q-table and optimal policy
    print("Q-table:")
    agent.print_q_table()
    print("\nOptimal Policy:")
    agent.print_optimal_policy()

    # Testing (10 episodes) with slower visualization
    for _ in range(10):
        obs, info = env.reset()
        while True:
            state = tuple(obs)
            action = agent.choose_action(state, env)
            obs, reward, done, info = env.step(action)
            env.render(tick_rate=5)  # Adjust tick_rate as needed for slower rendering

            if done:
                break

    env.close()

# Visualize the Q-table
def visualize_q_table(q_values_path):
    """
    Visualize the trained Q-table.

    Args:
        q_values_path (str): Path to the Q-table file.
    """
    q_table = np.load(q_values_path, allow_pickle=True).item()

    grid_size = 8
    action_space = 4
    Q = np.zeros((grid_size, grid_size, action_space))

    for (state, action), value in q_table.items():
        Q[state[0], state[1], action] = value

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    actions = ['Up', 'Down', 'Left', 'Right']

    for i in range(4):
        row = i // 2
        col = i % 2
        cax = ax[row, col].matshow(Q[:, :, i], cmap='viridis')

        for x in range(grid_size):
            for y in range(grid_size):
                ax[row, col].text(y, x, f'{Q[x, y, i]:.2f}', va='center', ha='center', color='white')

        fig.colorbar(cax, ax=ax[row, col])
        ax[row, col].set_title(f'Q-value for action: {actions[i]}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    visualize_q_table('q_table.npy')