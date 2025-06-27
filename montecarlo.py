import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pygame

import constants
from game import Game

ACTIONS = [-1, 0, 1]  # left, stay, right


class MonteCarloAgent:
    def __init__(self):
        self.Q = {}  # Q[state, action] = value
        self.returns = {}  # returns[state, action] = list of returns
        self.policy = {}  # state -> action
        self.epsilon = 0.1  # exploration rate
        self.rewards = []  # Add in __init__

    def get_state(self, game):
        # Discretize state
        paddle_bin = game.paddle.rect.x // constants.GAME_UNIT
        ball_bin_x = game.ball.rect.x // constants.GAME_UNIT
        ball_bin_y = game.ball.rect.y // constants.GAME_UNIT
        ball_dx = int(np.sign(game.ball.dx))  # -1, 0, 1 (discretize more if you want)
        state = (paddle_bin, ball_bin_x, ball_bin_y, ball_dx)
        return state

    def choose_action(self, state):
        if random.random() < self.epsilon or state not in self.policy:
            return random.choice(ACTIONS)
        return self.policy[state]

    def update_policy(self):
        # Make greedy policy from current Q
        for (state, action), value in self.Q.items():
            if state not in self.policy or self.Q.get((state, action), float('-inf')) > self.Q.get(
                    (state, self.policy[state]), float('-inf')):
                self.policy[state] = action

    def play_episode(self, screen, layout="rectangle", rows=3, cols=3, ball_start_direction=0, max_steps=500):
        # Set up game
        game = Game(screen, ball_start_direction=ball_start_direction)
        game.create_bricks_layout(layout, num_rows=rows, num_cols=cols)
        episode = []
        total_reward = 0

        for t in range(max_steps):
            state = self.get_state(game)
            action = self.choose_action(state)
            # Apply action
            if action == -1:
                game.paddle.move_left()
            elif action == 1:
                game.paddle.move_right()
            # Environment step
            game.update()
            reward = -1  # per assignment: -1 per timestep
            done = game.game_over
            episode.append((state, action, reward))
            total_reward += reward
            if done:
                break
        return episode, total_reward

    def run(self, num_episodes=1000, layout="rectangle", rows=3, cols=3, ball_start_direction=0, print_every=100):
        # Pygame headless mode
        # pygame.display.iconify()  # Minimize window, you can also set dummy video driver
        screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
        for ep in range(num_episodes):
            episode, total_reward = self.play_episode(screen, layout, rows, cols, ball_start_direction)
            G = 0
            visited = set()
            for (state, action, reward) in reversed(episode):
                G += reward
                if (state, action) not in visited:
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])
                    visited.add((state, action))
            self.update_policy()
            self.rewards.append(total_reward)
            if (ep + 1) % print_every == 0:
                print(f"Episode {ep + 1}/{num_episodes}, last total reward: {total_reward}")

        with open("mc_agent.pkl", "wb") as f:
            pickle.dump(self, f)

        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Reward over Episodes ({layout})')
        plt.savefig(f'imgs/training_reward_{layout}.png')
        plt.show()

        pygame.quit()
        print("Training complete!")
