import pygame
import constants
from constants import ROWS_RECTANGLE, ROWS_PYRAMID, ROWS_INVERTED_PYRAMID
from game import Game

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
pygame.display.set_caption("Breakout Game") # Changed title for basic game
clock = pygame.time.Clock()

# --- Main Game Loop ---

def main_game_loop():
    """Run the main game loop."""
    game = Game(screen)
    running = True

    # Define the initial layout for the simplified game
    num_bricks_config = {}
    if constants.BRICK_LAYOUT == "rectangle":
        num_bricks_config = {"rows": ROWS_RECTANGLE, "cols": constants.BRICK_COLUMNS}
    elif constants.BRICK_LAYOUT == "pyramid":
        num_bricks_config = {"rows": ROWS_PYRAMID, "cols": constants.BRICK_COLUMNS}
    elif constants.BRICK_LAYOUT == "inverted_pyramid":
        num_bricks_config = {"rows": ROWS_INVERTED_PYRAMID, "cols": constants.BRICK_COLUMNS}

    print(f"Loading layout: {constants.BRICK_LAYOUT} with {num_bricks_config['rows']} rows, {num_bricks_config['cols']} cols")
    game.create_bricks_layout(
        constants.BRICK_LAYOUT,
        num_rows=num_bricks_config['rows'],
        num_cols=num_bricks_config['cols']
    )

    # Manual Play Mode
    print("\n--- Starting Manual Play Mode ---")

    while running:
        pygame.event.get()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            game.paddle.move_left()
        if keys[pygame.K_RIGHT]:
            game.paddle.move_right()
        if keys[pygame.K_ESCAPE]:
            running = False

        if game.game_over:
            running = False
        game.update()
        game.draw()
        clock.tick(60) # Limit frame rate to 60 FPS

    pygame.quit()
    print("Pygame exited.")


if __name__ == "__main__":
    main_game_loop()

