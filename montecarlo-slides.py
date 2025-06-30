import random

import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

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
        paddle_bin = game.paddle.rect.x // (2*constants.GAME_UNIT)
        ball_bin_x = game.ball.rect.x // (2*constants.GAME_UNIT)
        ball_bin_y = game.ball.rect.y // (2*constants.GAME_UNIT)
        # Discretize ball direction
        ball_dx = int(game.ball.dx) # -1, 0, 1
        ball_dy = int(np.sign(game.ball.dy))

        paddle_velocity = int(game.paddle.vx)

        # min_y = min(b.rect.y for b in game.bricks)
        # lowest_bricks = [b for b in game.bricks if b.rect.y == min_y]
        # lowest_brick_x = (lowest_bricks[0].rect.x + lowest_bricks[-1].rect.right) // 2
        # lowest_brick_x_bin = lowest_brick_x // (2 * constants.GAME_UNIT)

        return paddle_bin, ball_bin_x, ball_bin_y, ball_dx, ball_dy, paddle_velocity

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
        # self.epsilon = max(0.995 * self.epsilon, 0.01)

        # Initialize game
        game = Game(screen, max_score=max_steps, ball_start_direction=ball_start_direction)
        game.create_bricks_layout(layout, num_rows=rows, num_cols=cols)

        episode = []
        total_reward = 0

        # Reset paddle and ball
        screen_width, _ = screen.get_size()
        game.paddle.rect.x = (screen_width - constants.PADDLE_WIDTH) // 2
        game.ball.spawn(ball_start_direction)

        for t in range(max_steps):
            state = self.get_state(game)
            action = self.choose_action(state)

            # Apply action to paddle
            game.paddle.vx = max(-game.paddle.max_speed, min(game.paddle.max_speed, game.paddle.vx + action))

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

    def run(self, screen, num_episodes=1000, layout="rectangle", rows=5, cols=10, print_every=100, ball_start_direction=0, max_steps=1000):
        """
        Run Monte Carlo control for a number of episodes.
        """
        # Initialize Pygame (headless mode)
        pygame.init()
        # Create a screen surface (not displayed)

        for ep in range(1, num_episodes + 1):
            episode, total_reward = self.generate_episode(screen, layout, rows, cols, ball_start_direction=ball_start_direction, max_steps=max_steps)
            self.rewards_history.append(total_reward)

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

            if ep % print_every == 0:
                print(f"Episode {ep}/{num_episodes}, last reward: {total_reward}")

        rewards = np.array(self.rewards_history)
        window = 100
        running = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.scatter(range(len(rewards)), rewards, s=8, alpha=0.2)
        plt.plot(range(window - 1, len(rewards)), running, color='orange', lw=2)
        plt.ylim(min(rewards), 0)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'MC Control learning ({layout})')
        plt.savefig(f"imgs/{brick_layout}-trajectory_{rows}x{cols}_{ball_start_direction}_learning.png")
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

def run_monte_carlo(epsilon, num_episodes, max_steps, layout, rows, cols, ball_start_direction, print_every):
    screen_width = cols * constants.BRICK_WIDTH + constants.GAME_UNIT * 3
    screen_height = constants.GAME_UNIT * 8 + rows * constants.BRICK_HEIGHT * 2
    screen = pygame.display.set_mode((screen_width, screen_height))


    agent = MonteCarloAgent(epsilon=epsilon)

    start_time = time.time()

    agent.run(
        screen=screen,
        num_episodes=num_episodes,
        layout=layout,
        rows=rows,
        cols=cols,
        print_every=print_every,
        ball_start_direction=ball_start_direction,
        max_steps=max_steps
    )

    elapsed_time = time.time() - start_time

    # # 2) Evaluate with actual rendering
    pygame.init()
    pygame.event.set_allowed(None)

    pygame.display.set_caption("MC Agent Evaluation")
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((screen_width, screen_height))

    game = Game(screen, max_score=max_steps, ball_start_direction=ball_start_direction)
    game.create_bricks_layout(
        layout,
        num_rows=rows,
        num_cols=cols
    )
    # reset ball & paddle positions
    game.ball.spawn(ball_start_direction)
    game.paddle.rect.x = (screen_width - constants.PADDLE_WIDTH) // 2

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

        clock.tick(60)

        if game.game_over:
            running = False

    game.draw(ball_trail, True)

    filename = f"imgs/{brick_layout}-trajectory_{rows}x{cols}_{ball_start_direction}_final_state.png"

    try:
        # Get the entire display surface and save it
        pygame.image.save(screen, filename)
        print(f"Screenshot saved as {filename}")
    except pygame.error as e:
        print(f"Error saving screenshot: {e}")

    return elapsed_time

# Example usage:
if __name__ == "__main__":
    # 1) Train in‐memory

    # starting_states = [-2, -1, 0, 1, 2]
    # brick_layouts = [constants.RECTANGLE_LAYOUT, constants.PYRAMID_LAYOUT, constants.INVERTED_PYRAMID_LAYOUT]
    # runtimes = {layout: [] for layout in brick_layouts}
    # for starting_state in starting_states:
    #     for brick_layout in brick_layouts:
    #
    #         elapsed_time = run_monte_carlo(
    #             0.1,
    #             2000,
    #             5000,
    #             brick_layout,
    #             3,
    #             3,
    #             starting_state,
    #             100)
    #
    #         runtimes[brick_layout].append(elapsed_time)
    #         print(f"Training runtime: {elapsed_time:.2f} seconds")
    #
    #
    #
    # # plot runtime for each layout for each state
    # n_states = len(starting_states)
    # n_layouts = len(brick_layouts)
    # x = np.arange(n_states)
    # total_width = 0.8
    # bar_width = total_width / n_layouts
    #
    # plt.figure(figsize=(10, 6))
    # cmap = plt.get_cmap('tab10')
    #
    # for i, layout in enumerate(brick_layouts):
    #     y = runtimes[layout]
    #     bars = plt.bar(
    #         x + i * bar_width,
    #         y,
    #         width=bar_width,
    #         label=layout,
    #         color=[cmap(i)] * n_states,
    #         edgecolor='black',
    #         linewidth=1
    #     )
    #     # Add data labels
    #     for bar in bars:
    #         h = bar.get_height()
    #         plt.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             h + 0.3,
    #             f'{h:.2f}',
    #             ha='center',
    #             va='bottom',
    #             fontsize=9
    #         )
    #
    # # Clean up spines
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # # Formatting
    # plt.xticks(x + total_width / 2 - bar_width / 2, starting_states, fontsize=10)
    # plt.xlabel("Starting State", fontsize=12)
    # plt.ylabel("Runtime (seconds)", fontsize=12)
    # plt.title("Training Runtime per Layout & Starting State", fontsize=14, fontweight='bold')
    # plt.legend(title="Layout")
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig('imgs/grouped_runtimes.png')
    # plt.show()

    starting_states = [0]
    brick_layouts = [constants.PYRAMID_LAYOUT, constants.INVERTED_PYRAMID_LAYOUT, constants.RECTANGLE_LAYOUT]
    runtimes = {layout: [] for layout in brick_layouts}
    for starting_state in starting_states:
        for brick_layout in brick_layouts:
            elapsed_time = run_monte_carlo(
                0.01,
                10000,
                20000,
                brick_layout,
                5,
                5,
                starting_state,
                100)

            runtimes[brick_layout].append(elapsed_time)
            print(f"Training runtime: {elapsed_time:.2f} seconds")


    # plot runtime for each layout for each state
    n_states = len(starting_states)
    n_layouts = len(brick_layouts)
    x = np.arange(n_states)
    total_width = 0.8
    bar_width = total_width / n_layouts

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, layout in enumerate(brick_layouts):
        y = runtimes[layout]
        bars = plt.bar(
            x + i * bar_width,
            y,
            width=bar_width,
            label=layout,
            color=[cmap(i)] * n_states,
            edgecolor='black',
            linewidth=1
        )
        # Add data labels
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f'{h:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    # Clean up spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Formatting
    plt.xticks(x + total_width / 2 - bar_width / 2, starting_states, fontsize=10)
    plt.xlabel("Starting State", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Training Runtime per Layout & Starting State", fontsize=14, fontweight='bold')
    plt.legend(title="Layout")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('imgs/grouped-large_runtimes.png')
    plt.show()

    pygame.quit()
    print("Evaluation complete!")