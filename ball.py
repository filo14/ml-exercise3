import pygame
import random
import constants

class Ball(pygame.sprite.Sprite):
    """Represents the ball."""
    def __init__(self):
        """Initialize the ball."""
        super().__init__()
        self.image = pygame.Surface([constants.BALL_RADIUS * 2, constants.BALL_RADIUS * 2], pygame.SRCALPHA)
        pygame.draw.circle(self.image, constants.WHITE, (constants.BALL_RADIUS, constants.BALL_RADIUS), constants.BALL_RADIUS)
        self.rect = self.image.get_rect()
        self.reset() # Initial position and velocity

    def reset(self):
        """Resets the ball to its starting position and velocity."""
        self.rect.x = constants.SCREEN_WIDTH // 2
        self.rect.y = constants.SCREEN_HEIGHT // 2
        self.dx = random.randint(-2, 2) # Initial horizontal velocity
        self.dy = 3 # Initial vertical velocity (upwards)
        self.speed_multiplier = 1.0 # Can be increased for difficulty
        self.old_rect = self.rect.copy()

    def update(self):
        """Updates the ball's position."""
        self.old_rect = self.rect.copy()

        self.rect.x += self.dx * self.speed_multiplier
        self.rect.y += self.dy * self.speed_multiplier

        # Wall collisions (left/right)
        if self.rect.left < 0 or self.rect.right > constants.SCREEN_WIDTH:
            self.dx *= -1 # Reverse horizontal direction
            self.rect.left = max(0, self.rect.left) # Keep within bounds
            self.rect.right = min(constants.SCREEN_WIDTH, self.rect.right)

        # Wall collision (top)
        if self.rect.top < 0:
            self.dy *= -1 # Reverse vertical direction
            self.rect.top = 0 # Keep within bounds
