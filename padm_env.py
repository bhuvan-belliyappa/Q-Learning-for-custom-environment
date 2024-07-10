
import math
import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Custom Environment
class MoonWorldEnv(gym.Env):
    def __init__(self):
        super(MoonWorldEnv, self).__init__()
        self.grid_size = 8
        self.moon_location = (7, 7)
        self.meteor_states = [(5, 5), (7, 5), (1, 5), (4, 3), (2, 3), (4, 7)]
        self.holes = [(0, 2), (1, 7), (3, 5), (4, 1), (6, 3)]
        self.meteors = self.meteor_states
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.window_size = 700
        self.cell_size = self.window_size // self.grid_size
        pygame.init()     
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('MoonWorld Environment')
        self.clock = pygame.time.Clock()
        self.background_img = pygame.image.load('background.jpg')
        self.background_img = pygame.transform.scale(self.background_img, (self.window_size, self.window_size))
        self.rocket_img = pygame.image.load('rocket.png')
        self.rocket_img = pygame.transform.scale(self.rocket_img, (self.cell_size, self.cell_size))
        self.moon_img = pygame.image.load('moon.png')
        self.moon_img = pygame.transform.scale(self.moon_img, (self.cell_size, self.cell_size))
        self.meteor_img = pygame.image.load('meteor.png')
        self.meteor_img = pygame.transform.scale(self.meteor_img, (self.cell_size, self.cell_size))

    def reset(self):
        self.agent_position = [0, 0]
        self.history = [self.agent_position.copy()]
        self.fuel = 20
        distance_to_moon = self.calculate_distance(self.agent_position, self.moon_location)
        info = {'distance_to_moon': distance_to_moon, 'fuel': self.fuel}
        return np.array(self.agent_position), info

    def step(self, action):
        prev_position = self.agent_position.copy()
        new_position = self.agent_position.copy()
        
        if action == 0 and self.agent_position[1] > 0:  # Up
            new_position[1] -= 1
        elif action == 1 and self.agent_position[1] < self.grid_size - 1:  # Down
            new_position[1] += 1
        elif action == 2 and self.agent_position[0] > 0:  # Left
            new_position[0] -= 1
        elif action == 3 and self.agent_position[0] < self.grid_size - 1:  # Right
            new_position[0] += 1

        if tuple(new_position) not in self.holes:
            self.agent_position = new_position

        self.history.append(self.agent_position.copy())
        self.fuel -= 1

        done = False
        reward = 0  # Default reward is 0, only specific rewards for reaching goals or hitting undesired states

        if tuple(self.agent_position) == self.moon_location:
            reward = 100  # High reward for reaching the moon
            done = True
        elif tuple(self.agent_position) in self.meteors:
            reward = -50  # High penalty (negative reward) for hitting a meteor
            done = True
        elif self.fuel <= 0:
            reward = -10  # Penalty (negative reward) for running out of fuel
            done = True

        distance_to_moon = self.calculate_distance(self.agent_position, self.moon_location)
        info = {'distance_to_moon': distance_to_moon, 'fuel': self.fuel}

        return np.array(self.agent_position), reward, done, info

    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def render(self, mode='human', tick_rate=10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background_img, (0, 0))

        for x in range(0, self.window_size, self.cell_size):
            for y in range(0, self.window_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        self.screen.blit(self.rocket_img, (self.agent_position[0] * self.cell_size, self.agent_position[1] * self.cell_size))
        self.screen.blit(self.moon_img, (self.moon_location[0] * self.cell_size, self.moon_location[1] * self.cell_size))

        for meteor in self.meteor_states:
            self.screen.blit(self.meteor_img, (meteor[0] * self.cell_size, meteor[1] * self.cell_size))

        for hole in self.holes:
            hole_rect = pygame.Rect(
                hole[0] * self.cell_size,
                hole[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 0, 0), hole_rect)

        pygame.display.flip()
        self.clock.tick(tick_rate)  # Adjusted tick rate for slower rendering

    def close(self):
        pygame.quit()