import pygame
import constants

class Paddle(pygame.sprite.Sprite):
    """Represents the player's paddle."""
    def __init__(self):
        """Initialize the paddle."""
        super().__init__()
        self.image = pygame.Surface([constants.PADDLE_WIDTH, constants.PADDLE_HEIGHT])
        self.image.fill(constants.WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = (constants.SCREEN_WIDTH - constants.PADDLE_WIDTH) // 2
        self.rect.y = constants.SCREEN_HEIGHT - constants.PADDLE_HEIGHT - 30
        self.speed = constants.GAME_UNIT # Paddle movement speed

    def move_left(self):
        """Moves the paddle to the left."""
        self.rect.x -= self.speed
        if self.rect.x < 0:
            self.rect.x = 0

    def move_right(self):
        """Moves the paddle to the right."""
        self.rect.x += self.speed
        if self.rect.x > constants.SCREEN_WIDTH - constants.PADDLE_WIDTH:
            self.rect.x = constants.SCREEN_WIDTH - constants.PADDLE_WIDTH
