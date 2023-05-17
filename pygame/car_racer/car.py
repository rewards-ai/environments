import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pygame
import math
import os
from .config import Config
from typing import Optional, Tuple
import random

class Car():
    def __init__(self, radar_nums: Optional[int] = 5, show_radar: Optional[bool] = True) -> None:
        """Initialize the Car(Agent) class

        Args:
            radar_nums ([int]): Number of equi-distant radars. Serves as an Input to learning algorithms.
            show_radar (bool): Flag to show radars on screen or not. Defaults to True.
        """
        self.radar_nums = radar_nums
        self.radar_angles = np.linspace(-90, 90, self.radar_nums)
        
        self.config = Config()
        self.show_radar = show_radar
        self.image_raw = pygame.image.load(Path(__file__).parent.joinpath(self.config.image_raw))
        self.image_raw = pygame.transform.scale(self.image_raw, (500, 500))
        self.move_factor = self.config.move_factor
        
        self.reset()
    
    def reset(self) -> None:
        """Reset the Car back to the starting position. Reset other car variables too.
        """
        self.angle = 0
        self.reward = 0
        self.image = pygame.transform.rotozoom(self.image_raw, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.config.center)
        
        self.velocity_vector = pygame.math.Vector2(self.config.velocity_vector)
        self.rotation_vel = self.config.rotation_velocity
        
        self.radar_locations = [self.rect.center] * self.radar_nums
        
        self.direction = 0
        self.alive = True
        self.reward = 0
        self.radars = [0] * self.radar_nums
    
    
    def _is_off_track(self) -> bool:
        """Checks if Car is off the track using first and last radars. Also Checks if Car is off screen

        Returns:
            bool: True if car is off-track else False
        """
        x = self.rect.center[0]
        y = self.rect.center[1]
        if (self.radars[0] < 20 and self.radars[-1] < 20) or ((x < 0 or x > 800) or (y < 0 or y > 700)):
            self.alive = False
        else:
            self.alive = True
            
    def _drive(self) -> None:
        """Drives Car's pygame.Rect every frame by a move_factor on velocity_vector
        """
        self.rect.center += self.velocity_vector * self.move_factor
        
    def _generate_radar(self, i: int, radar_angle: int, screen) -> None:
        """Generates a radar for the list of radars of the Car

        Args:
            i (int): Denotes the i'th radar
            radar_angle (int): Denotes angle offset of radar
            screen: Pygame main screen display variable
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while not screen.get_at((x, y)) == pygame.Color(173, 255, 133, 255) and length < 200:
                length += 1
                x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)
                
            self.radar_locations[i] = (x, y)
            dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
            self.radars[i] = dist
        except:
            self.alive = False
        
    def _rotate(self) -> None:
        """Rotates Car's pygame.Rect by an angle based on direction
        """
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.velocity_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.velocity_vector.rotate_ip(-self.rotation_vel)
            
        self.image = pygame.transform.rotozoom(self.image_raw, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

            
    def _draw(self, i: int, screen) -> None:
        """Draws Car to the screen

        Args:
            i (int): i'th car in the environement
            screen: Pygame main screen display variable
        """
        screen.blit(self.image, self.rect.topleft)
        
        font = pygame.font.Font(None, 24)
        text = "car " + str(i)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = self.rect.topleft
        screen.blit(text_surface, text_rect)
        
        if self.show_radar and self.alive:
            for radar in self.radar_locations:
                pygame.draw.line(screen, (255, 255, 255, 255), self.rect.center, radar, 1)
                pygame.draw.circle(screen, (0, 255, 0, 0), radar, 3)
        
    def step(self, action: list, screen) -> None:
        """Drives, rotates and check off-track of the car based on action list.

        Args:
            action (list): List of actions to display what move it=s taken.
            screen: Pygame main screen display variable
        """
        self._drive()
        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
            self._drive()
        elif action[3] == 1:
            self.velocity_vector.scale_to_length(0.8)
            self.rotation_vel = 15
        elif action[4] == 1:
            self.velocity_vector.scale_to_length(1.2)
            self.rotation_vel = 10
        elif action[5] == 1:
            self.velocity_vector.scale_to_length(1.6)
            self.rotation_vel = 7
        else:
            self.direction = 0
            
        self._rotate()
        
        for i, radar_angle in enumerate(self.radar_angles):
            self._generate_radar(i, radar_angle, screen)

        self._is_off_track()