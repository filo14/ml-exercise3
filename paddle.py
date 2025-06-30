import pygame
import constants

class Paddle(pygame.sprite.Sprite):
    """Represents the player's paddle."""
    def __init__(self, screen_width, screen_height):
        """Initialize the paddle."""
        super().__init__()
        self.vx = 0
        self.max_speed = 2
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image = pygame.Surface([constants.PADDLE_WIDTH, constants.PADDLE_HEIGHT])
        self.image.fill(constants.WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = (self.screen_width - constants.PADDLE_WIDTH) // 2
        self.rect.y = self.screen_height - constants.PADDLE_HEIGHT - 30
        self.speed = constants.GAME_UNIT # Paddle movement speed

    def update(self):
        """Moves the paddle."""
        self.rect.x += self.vx
        if self.rect.x < 0:
            self.rect.x = 0
        if self.rect.x > self.screen_width - constants.PADDLE_WIDTH:
            self.rect.x = self.screen_width - constants.PADDLE_WIDTH

    def move_left(self):
        """Moves the paddle to the left."""
        if self.vx > -self.max_speed:
            self.vx -= 1

    def move_right(self):
        """Moves the paddle to the right."""
        if self.vx < self.max_speed:
            self.vx += 1
