import pygame

import constants
from ball import Ball
from brick import Brick
from paddle import Paddle


class Game:
    """Manages the overall game state and logic."""

    def __init__(self, screen, max_score=1000):
        """Initialize the game and set screen."""
        self.screen = screen

        self.screen_width, self.screen_height = screen.get_size()

        self.max_score = max_score
        self.score = max_score

        self.all_sprites = pygame.sprite.Group()
        self.bricks = pygame.sprite.Group()
        self.init_bricks = pygame.sprite.Group()
        self.paddle = Paddle(self.screen_width, self.screen_height)

        self.ball = Ball(self.screen_width, self.screen_height)

        self.all_sprites.add(self.paddle)
        self.all_sprites.add(self.ball)

        self.game_over = False

        self.layout_type = None
        self.num_rows = None
        self.num_cols = None

    def create_bricks_layout(self, layout_type="rectangle", num_rows=5, num_cols=10):
        """
        Creates bricks based on a specified layout type.
        You can expand this for different layouts.
        """
        self.layout_type = layout_type
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.bricks.empty()  # Clear existing bricks
        # Remove bricks from all_sprites before adding new ones
        for sprite in self.all_sprites:
            if isinstance(sprite, Brick):
                sprite.kill()

        start_x = (self.screen_width - num_cols * constants.BRICK_WIDTH) // 2
        start_y = 2 * constants.GAME_UNIT

        if layout_type == "rectangle":
            for row in range(num_rows):
                for col in range(num_cols):
                    x = start_x + col * constants.BRICK_WIDTH
                    y = start_y + row * constants.BRICK_HEIGHT
                    color = constants.BRICK_COLORS[row % len(constants.BRICK_COLORS)]
                    brick = Brick(x, y, color)
                    self.bricks.add(brick)
                    self.init_bricks.add(Brick(x, y, color))
                    self.all_sprites.add(brick)
        elif layout_type == "pyramid":
            max_cols = num_cols
            for row in range(num_rows):
                current_cols = max_cols - row
                row_start_x = (self.screen_width - current_cols * constants.BRICK_WIDTH) // 2
                for col in range(current_cols):
                    x = row_start_x + col * constants.BRICK_WIDTH
                    y = start_y + row * constants.BRICK_HEIGHT
                    color = constants.BRICK_COLORS[row % len(constants.BRICK_COLORS)]
                    brick = Brick(x, y, color)
                    self.bricks.add(brick)
                    self.init_bricks.add(Brick(x, y, color))
                    self.all_sprites.add(brick)
        elif layout_type == "inverted_pyramid":
            for row in range(num_rows):
                current_cols = row + 1
                if current_cols > num_cols:
                    current_cols = num_cols

                row_width = current_cols * constants.BRICK_WIDTH
                row_start_x = (self.screen_width - row_width) // 2

                for col in range(current_cols):
                    x = row_start_x + col * constants.BRICK_WIDTH
                    y = start_y + row * constants.BRICK_HEIGHT
                    color = constants.BRICK_COLORS[row % len(constants.BRICK_COLORS)]
                    brick = Brick(x, y, color)
                    self.bricks.add(brick)
                    self.init_bricks.add(Brick(x, y, color))
                    self.all_sprites.add(brick)

    def handle_collisions(self):
        """Handles ball collisions with paddle and bricks."""
        # Ball-paddle collision
        if pygame.sprite.collide_rect(self.ball, self.paddle):
            if self.ball.dy > 0:  # Only reflect if ball is moving down
                self.ball.dy *= -1
                self.ball.rect.bottom = self.paddle.rect.top

                # Calculate new horizontal velocity based on where it hit the paddle
                # Hitting the center results in vertical bounce, hitting edges causes more horizontal bounce
                ball_center_x = self.ball.rect.centerx
                paddle_center_x = self.paddle.rect.centerx
                # Normalize hit position to [-2, 2] relative to paddle width
                hit_position = (ball_center_x - paddle_center_x) / (constants.PADDLE_WIDTH / 4)
                # Adjust ball's dx based on hit_position
                self.ball.dx = hit_position * 2  # Max horizontal speed 2

        # Ball-brick collisions
        hit_bricks = pygame.sprite.spritecollide(self.ball, self.bricks, True)
        for brick in hit_bricks:
            # Collision from top
            if self.ball.old_rect.bottom <= brick.rect.top < self.ball.rect.bottom:
                self.ball.dy *= -1
                self.ball.rect.bottom = brick.rect.top

            # Collision from bottom
            elif self.ball.old_rect.top >= brick.rect.bottom > self.ball.rect.top:
                self.ball.dy *= -1
                self.ball.rect.top = brick.rect.bottom

            # Collision from left
            elif self.ball.old_rect.right <= brick.rect.left < self.ball.rect.right:
                self.ball.dx *= -1
                self.ball.rect.right = brick.rect.left

            # Collision from right
            elif self.ball.old_rect.left >= brick.rect.right > self.ball.rect.left:
                self.ball.dx *= -1
                self.ball.rect.left = brick.rect.right

    def update(self):
        """Updates all game elements and handles game logic."""
        if self.game_over:
            return

        self.paddle.update()
        self.ball.update()
        self.handle_collisions()
        if self.score >= 1:
            self.score -= 1

        # Check if ball went past the paddle
        if self.ball.rect.top > self.screen_height:
            self.create_bricks_layout(self.layout_type, num_rows=self.num_rows, num_cols=self.num_cols)
            self.score = self.max_score
            self.ball.reset()  # Reset ball randomly
            self.paddle.rect.x = (self.screen_width - constants.PADDLE_WIDTH) // 2  # Reset paddle position
            self.paddle.vx = 0

        # Check if all bricks are cleared
        if not self.bricks:
            self.game_over = True

    def draw(self, draw_trail=None, draw_hit_bricks=False):
        """Draws all game elements on the screen."""
        self.screen.fill(constants.BLACK)  # Clear screen

        if draw_hit_bricks:
            self.init_bricks.draw(self.screen)

        if draw_trail is not None:
            for point in draw_trail:
                ball_x, ball_y = point
                pygame.draw.circle(self.screen, constants.RED, (int(ball_x), int(ball_y)), 1, 0)

        self.all_sprites.draw(self.screen)

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, constants.WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()  # Update the display
