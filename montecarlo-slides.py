import random

import matplotlib.pyplot as plt
import numpy as np
import pygame

import constants
from game import Game

# Define possible paddle actions: -1 (left), 0 (stay), 1 (right)
ACTIONS = [-1, 0, 1]

class MonteCarloAgent:
    """
    First-visit Monte Carlo control agent with epsilon-soft policy.
    """
    def __init__(self, epsilon=0.1):
        # Exploration rate
        self.epsilon = epsilon
        # Action-value estimates: Q[s][a] = value
        self.Q = {}  # nested dict: state -> {action: value}
        # Returns for each state-action pair: list of returns
        self.returns = {}  # (state, action) -> [G1, G2, ...]
        # Policy: maps state to action
        self.policy = {}  # state -> action
        # Track total reward per episode
        self.rewards_history = []

    def get_state(self, game):
        """
        Discretize the game state for tabular learning.
        """
        # Bin positions by game unit
        paddle_bin = game.paddle.rect.x // constants.GAME_UNIT
        ball_bin_x = game.ball.rect.x // constants.GAME_UNIT
        ball_bin_y = game.ball.rect.y // constants.GAME_UNIT
        # Discretize ball direction
        ball_dx = int( game.ball.dx) # -1, 0, 1
        return paddle_bin, ball_bin_x, ball_bin_y, ball_dx

    def initialize_state(self, state):
        """
        Ensure Q and policy initialized for a given state.
        """
        if state not in self.Q:
            # Initialize each action's value to zero
            self.Q[state] = {a: 0.0 for a in ACTIONS}
            # Initialize returns list
            for a in ACTIONS:
                self.returns[(state, a)] = []
            # Initialize an epsilon-soft policy: random action
            self.policy[state] = random.choice(ACTIONS)

    def choose_action(self, state):
        """
        Epsilon-soft action selection.
        """
        self.initialize_state(state)
        # With probability epsilon choose random action
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        # Otherwise choose greedy action
        action_values = self.Q[state]
        max_val = max(action_values.values())
        # Break ties randomly
        best_actions = [a for a, v in action_values.items() if v == max_val]
        return random.choice(best_actions)

    def generate_episode(self, screen, layout, rows, cols, max_steps=1000, ball_start_direction=0):
        """
        Generate an episode following current epsilon-soft policy.
        Returns a list of (state, action, reward)."""
        # Initialize game
        game = Game(screen, ball_start_direction=ball_start_direction)
        game.create_bricks_layout(layout, num_rows=rows, num_cols=cols)

        episode = []
        total_reward = 0

        # Reset paddle and ball
        game.ball.reset()
        game.paddle.rect.x = (constants.SCREEN_WIDTH - constants.PADDLE_WIDTH) // 2

        for t in range(max_steps):
            state = self.get_state(game)
            action = self.choose_action(state)

            # Apply action to paddle
            if action == -1:
                game.paddle.move_left()
            elif action == 1:
                game.paddle.move_right()

            # Environment step
            game.update()

            # Reward: -1 per timestep until all bricks cleared
            reward = -1
            done = game.game_over

            episode.append((state, action, reward))
            total_reward += reward

            if done:
                break

        return episode, total_reward

    def run(self, num_episodes=1000, layout="rectangle", rows=5, cols=10, print_every=100, ball_start_direction=0, max_steps=1000):
        """
        Run Monte Carlo control for a number of episodes.
        """
        # Initialize Pygame (headless mode)
        pygame.init()
        # Create a screen surface (not displayed)
        screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))

        for ep in range(1, num_episodes + 1):
            episode, total_reward = self.generate_episode(screen, layout, rows, cols, ball_start_direction=ball_start_direction, max_steps=max_steps)
            self.rewards_history.append(total_reward)
            self.epsilon = max(self.epsilon * 0.995, 0.01)

            # Keep track of first visits
            visited = set()
            G = 0
            # Process episode in reverse to compute returns
            for state, action, reward in reversed(episode):
                G = reward + G
                if (state, action) not in visited:
                    visited.add((state, action))
                    # Append return
                    self.returns[(state, action)].append(G)
                    # Update Q to average of returns
                    self.Q[state][action] = np.mean(self.returns[(state, action)])

            # Policy improvement: for all states seen in this episode
            for state, _, _ in episode:
                # Choose greedy action for state
                action_values = self.Q[state]
                max_val = max(action_values.values())
                best_actions = [a for a, v in action_values.items() if v == max_val]
                best = random.choice(best_actions)
                self.policy[state] = best

            # if ep % print_every == 0:
            print(f"Episode {ep}/{num_episodes}, last reward: {total_reward}")

        # Plot learning curve
        plt.plot(self.rewards_history)
        min_return = min(self.rewards_history)
        plt.ylim(min_return, 0)  # bottom = worst episode, top = 0
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'MC Control learning ({layout})')
        plt.savefig(f'imgs/learning-curves/mc_learning_{layout}-trajectory_{ball_start_direction}.png')
        plt.show()



        rewards = np.array(self.rewards_history)
        window = 100
        running = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.scatter(range(len(rewards)), rewards, s=8, alpha=0.2)
        plt.plot(range(window - 1, len(rewards)), running, color='orange', lw=2)
        plt.ylim(min(rewards), 0)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'MC Control learning ({layout})')
        plt.savefig(f'imgs/average-curves/mc_learning_{layout}-trajectory_{ball_start_direction}.png')
        plt.show()

        pygame.quit()
        print("Training complete!")

    def greedy_action(self, state):
        """
        Return the action with the highest Q-value for `state`.
        Breaks ties at random.  Does *not* use ε.
        """
        self.initialize_state(state)          # make sure Q[state] exists
        qs = self.Q[state]
        max_val = max(qs.values())
        best_actions = [a for a, v in qs.items() if v == max_val]
        return random.choice(best_actions)

# Example usage:
if __name__ == "__main__":
    # 1) Train in‐memory
    starting_states = [-2]
    brick_layouts = [constants.INVERTED_PYRAMID_LAYOUT]
    for brick_layout in brick_layouts:

        for starting_state in starting_states:

            agent = MonteCarloAgent(epsilon=0.1)
            agent.run(
                num_episodes=constants.NUM_OF_EPISODES,
                layout=brick_layout,
                rows=constants.ROWS_RECTANGLE,
                cols=constants.BRICK_COLUMNS,
                print_every=100,
                ball_start_direction=starting_state
            )

            # # 2) Evaluate with actual rendering
            pygame.init()
            pygame.event.set_allowed(None)
            screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
            pygame.display.set_caption("MC Agent Evaluation")
            clock = pygame.time.Clock()

            game = Game(screen, ball_start_direction=starting_state)
            game.create_bricks_layout(
                brick_layout,
                num_rows=constants.ROWS_RECTANGLE,
                num_cols=constants.BRICK_COLUMNS
            )
            # reset ball & paddle positions
            game.ball.reset()
            game.paddle.rect.x = (constants.SCREEN_WIDTH - constants.PADDLE_WIDTH)//2

            ball_trail = []
            running = True
            while running:
                pygame.event.pump()  # allow window events (so it doesn’t “not responding”)
                # 1) pick action by policy
                # state = agent.get_state(game)
                # action = agent.policy.get(state, agent.choose_action(state))
                agent.epsilon = 0
                state = agent.get_state(game)
                action = agent.greedy_action(state)

                if action == -1:
                    game.paddle.move_left()
                elif action == 1:
                    game.paddle.move_right()
                # 2) step & draw
                game.update()
                ball_trail.append(game.ball.rect.center)
                game.draw()
                # pygame.time.delay(30)

                clock.tick(60)

                if game.game_over:
                    running = False

            game.draw(ball_trail, True)

            filename = f"imgs/trajectories/final_game_state_{brick_layout}_{starting_state}.png"

            try:
                # Get the entire display surface and save it
                pygame.image.save(screen, filename)
                print(f"Screenshot saved as {filename}")
            except pygame.error as e:
                print(f"Error saving screenshot: {e}")

    pygame.quit()
    print("Evaluation complete!")