import sys

import pygame
import constants
from game import Game
from montecarlo import MonteCarloAgent
import pickle
import os


def play_with_policy(agent, layout="rectangle", rows=3, cols=3, max_steps=2000):
    pygame.init()
    screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
    pygame.display.set_caption("Breakout AI Agent Playback")
    game = Game(screen)
    game.create_bricks_layout(layout, num_rows=rows, num_cols=cols)
    running = True
    step = 0

    while running and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        state = agent.get_state(game)
        action = agent.policy.get(state, 0)  # Default to 'stay' if unseen state
        if action == -1:
            game.paddle.move_left()
        elif action == 1:
            game.paddle.move_right()
        game.update()
        game.draw()
        pygame.time.delay(30)  # slow down for visualization
        if game.game_over:
            print("Game over!")
            running = False
        step += 1

    pygame.quit()


if __name__ == "__main__":
    # Load a saved agent if you want, or retrain
    layout_type = "inverted_pyramid"
    if os.path.exists(f"pickle/mc_agent_{layout_type}.pkl"):
        with open(f"pickle/mc_agent_{layout_type}.pkl", "rb") as f:
            agent = pickle.load(f)
    else:
        print(f"No agent pickle file found. Please first run the train_{layout_type}.py file")
        sys.exit()

    play_with_policy(agent, layout=layout_type, rows=constants.ROWS_INVERTED_PYRAMID, cols=constants.BRICK_COLUMNS)

    layout_type = "pyramid"
    if os.path.exists(f"pickle/mc_agent_{layout_type}.pkl"):
        with open(f"pickle/mc_agent_{layout_type}.pkl", "rb") as f:
            agent = pickle.load(f)
    else:
        print(f"No agent pickle file found. Please first run the train_{layout_type}.py file")
        sys.exit()

    play_with_policy(agent, layout=layout_type, rows=constants.ROWS_INVERTED_PYRAMID, cols=constants.BRICK_COLUMNS)


    layout_type = "rectangle"
    if os.path.exists(f"pickle/mc_agent_{layout_type}.pkl"):
        with open(f"pickle/mc_agent_{layout_type}.pkl", "rb") as f:
            agent = pickle.load(f)
    else:
        print(f"No agent pickle file found. Please first run the train_{layout_type}.py file")
        sys.exit()

    play_with_policy(agent, layout=layout_type, rows=constants.ROWS_INVERTED_PYRAMID, cols=constants.BRICK_COLUMNS)
