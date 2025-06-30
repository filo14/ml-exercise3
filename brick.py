import pygame

import constants


class Brick(pygame.sprite.Sprite):
    """Represents a single brick."""

    def __init__(self, x, y, color):
        """Initialize the brick."""
        super().__init__()
        self.image = pygame.Surface(
            [constants.BRICK_WIDTH - constants.BRICK_DRAW_OFFSET, constants.BRICK_HEIGHT - constants.BRICK_DRAW_OFFSET])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x + constants.BRICK_DRAW_OFFSET // 2
        self.rect.y = y + constants.BRICK_DRAW_OFFSET // 2
