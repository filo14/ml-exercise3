import pygame
import random
from brick import Brick
from ball import Ball
from paddle import Paddle
import constants

class Game:
    """Manages the overall game state and logic."""
    def __init__(self, screen):
        """Initialize the game and set screen."""
        self.screen = screen

        self.all_sprites = pygame.sprite.Group()
        self.bricks = pygame.sprite.Group()
        self.paddle = Paddle()
        self.ball = Ball()

        self.all_sprites.add(self.paddle)
        self.all_sprites.add(self.ball)

        self.score = 0
        self.game_over = False

        self.layout_type = None
        self.num_rows = None
        self.num_cols = None

        self.score_per_point = constants.SCORE_PER_POINT

    def create_bricks_layout(self, layout_type="rectangle", num_rows=5, num_cols=10):
        """
        Creates bricks based on a specified layout type.
        You can expand this for different layouts.
        """
        self.layout_type = layout_type
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.bricks.empty() # Clear existing bricks
        # Remove bricks from all_sprites before adding new ones
        for sprite in self.all_sprites:
            if isinstance(sprite, Brick):
                sprite.kill()

        start_x = 0
        start_y = 50

        if layout_type == "rectangle":
            for row in range(num_rows):
                for col in range(num_cols):
                    x = start_x + col * constants.BRICK_WIDTH
                    y = start_y + row * constants.BRICK_HEIGHT
                    print(x)
                    color = constants.BRICK_COLORS[row % len(constants.BRICK_COLORS)]
                    brick = Brick(x, y, color)
                    self.bricks.add(brick)
                    self.all_sprites.add(brick)
        elif layout_type == "pyramid":
            max_cols = num_cols
            for row in range(num_rows):
                current_cols = max_cols - row
                row_start_x = (constants.SCREEN_WIDTH - current_cols * constants.BRICK_WIDTH) // 2
                for col in range(current_cols):
                    x = row_start_x + col * constants.BRICK_WIDTH
                    y = start_y + row * constants.BRICK_HEIGHT
                    color = constants.BRICK_COLORS[row % len(constants.BRICK_COLORS)]
                    brick = Brick(x, y, color)
                    self.bricks.add(brick)
                    self.all_sprites.add(brick)
        elif layout_type == "scattered":
            num_bricks_to_place = constants.ROWS_SCATTERED * constants.BRICK_COLUMNS # Approximate number
            for _ in range(num_bricks_to_place):
                x = random.randint(0, constants.SCREEN_WIDTH - constants.BRICK_WIDTH)
                y = random.randint(start_y, start_y + num_rows * constants.BRICK_HEIGHT)
                color = random.choice(constants.BRICK_COLORS)
                brick = Brick(x, y, color)
                # Avoid overlapping too much if possible, for simplicity, we'll just place
                self.bricks.add(brick)
                self.all_sprites.add(brick)

        print(f"Bricks created for layout: {layout_type} with {len(self.bricks)} bricks.")


    def handle_collisions(self):
        """Handles ball collisions with paddle and bricks."""
        # Ball-paddle collision
        if pygame.sprite.collide_rect(self.ball, self.paddle):
            # Ensure ball doesn't get stuck in paddle
            if self.ball.dy > 0: # Only reflect if ball is moving down
                self.ball.dy *= -1
                # Adjust ball's y-position to prevent sticking
                self.ball.rect.bottom = self.paddle.rect.top

                # Calculate new horizontal velocity based on where it hit the paddle
                # Hitting the center results in vertical bounce, hitting edges causes more horizontal bounce
                ball_center_x = self.ball.rect.centerx
                paddle_center_x = self.paddle.rect.centerx
                # Normalize hit position to [-1, 1] relative to paddle width
                hit_position = (ball_center_x - paddle_center_x) / (constants.PADDLE_WIDTH / 2)
                # Adjust ball's dx based on hit_position
                self.ball.dx = hit_position * 2 # Max horizontal speed 5

        # Ball-brick collisions
        hit_bricks = pygame.sprite.spritecollide(self.ball, self.bricks, True) # True means remove brick
        for brick in hit_bricks:
            self.score += self.score_per_point

            # Collision from top
            if self.ball.old_rect.bottom <= brick.rect.top < self.ball.rect.bottom:
                self.ball.dy *= -1
                self.ball.rect.bottom = brick.rect.top  # Adjust ball to prevent sticking

            # Collision from bottom
            elif self.ball.old_rect.top >= brick.rect.bottom > self.ball.rect.top:
                self.ball.dy *= -1
                self.ball.rect.top = brick.rect.bottom  # Adjust ball to prevent sticking

            # Collision from left
            elif self.ball.old_rect.right <= brick.rect.left < self.ball.rect.right:
                self.ball.dx *= -1
                self.ball.rect.right = brick.rect.left  # Adjust ball to prevent sticking

            # Collision from right
            elif self.ball.old_rect.left >= brick.rect.right > self.ball.rect.left:
                self.ball.dx *= -1
                self.ball.rect.left = brick.rect.right  # Adjust

    def update(self): # Removed 'action' parameter
        """Updates all game elements and handles game logic."""
        if self.game_over:
            return

        self.paddle.update() # No action parameter here
        self.ball.update()
        self.handle_collisions()
        if self.score_per_point > 1:
            self.score_per_point -= 1

        # Check if ball went past the paddle
        if self.ball.rect.top > constants.SCREEN_HEIGHT:
            self.create_bricks_layout(self.layout_type, num_rows=self.num_rows, num_cols=self.num_cols)
            self.score = 0
            self.ball.reset()
            self.paddle.rect.x = (constants.SCREEN_WIDTH - constants.PADDLE_WIDTH) // 2 # Reset paddle position
            self.score_per_point = constants.SCORE_PER_POINT

        # Check if all bricks are cleared
        if not self.bricks:
            self.game_over = True

            print("You Win!")

    def draw(self):
        """Draws all game elements on the screen."""
        self.screen.fill(constants.BLACK) # Clear screen
        self.all_sprites.draw(self.screen)

        # Display score and lives
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, constants.WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip() # Update the display