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

    try:
        if os.path.exists("mc_agent.pkl"):
            with open("mc_agent.pkl", "rb") as f:
                agent = pickle.load(f)
        else:
            agent = MonteCarloAgent()
            agent.run(num_episodes=500, layout=constants.BRICK_LAYOUT, rows=3, cols=constants.BRICK_COLUMNS)
            with open("mc_agent.pkl", "wb") as f:
                pickle.dump(agent, f)
        play_with_policy(agent, layout=constants.BRICK_LAYOUT, rows=3, cols=constants.BRICK_COLUMNS)
    except Exception as e: print(e)
