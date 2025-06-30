import pygame
import random
import constants

class Ball(pygame.sprite.Sprite):
    """Represents the ball."""
    def __init__(self, screen_width, screen_height):
        """Initialize the ball."""
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image = pygame.Surface([constants.BALL_RADIUS * 2, constants.BALL_RADIUS * 2], pygame.SRCALPHA)
        pygame.draw.circle(self.image, constants.WHITE, (constants.BALL_RADIUS, constants.BALL_RADIUS), constants.BALL_RADIUS)
        self.rect = self.image.get_rect()
        self.reset() # Initial position and velocity
        self.dx = random.randint(-2, 2)

    def spawn(self, dx=None):
        """Resets the ball to its starting position and velocity."""
        # if self.initial_dx is None:
        if dx is not None:
            self.dx = dx
        self.rect.x = self.screen_width // 2 - constants.BALL_RADIUS
        self.rect.y = self.screen_height // 2 + self.screen_height // 4 - constants.BALL_RADIUS
        self.dy = 1 # Initial vertical velocity (upwards)
        self.speed_multiplier = 1.0 # Can be increased for difficulty
        self.old_rect = self.rect.copy()

    def reset(self):
        """Resets the ball to its starting position and velocity."""
        # if self.initial_dx is None:
        self.dx = random.randint(-2, 2)
        self.rect.x = self.screen_width // 2 - constants.BALL_RADIUS
        self.rect.y = self.screen_height // 2 + self.screen_height // 4 - constants.BALL_RADIUS
        self.dy = 1 # Initial vertical velocity (upwards)
        self.old_rect = self.rect.copy()

    def update(self):
        """Updates the ball's position."""
        self.old_rect = self.rect.copy()

        self.rect.x += self.dx * self.speed_multiplier
        self.rect.y += self.dy * self.speed_multiplier

        # Wall collisions (left/right)
        if self.rect.left < 0 or self.rect.right > self.screen_width:
            self.dx *= -1 # Reverse horizontal direction
            self.rect.left = max(0, self.rect.left) # Keep within bounds
            self.rect.right = min(self.screen_width, self.rect.right)

        # Wall collision (top)
        if self.rect.top < 0:
            self.dy *= -1 # Reverse vertical direction
            self.rect.top = 0 # Keep within bounds
