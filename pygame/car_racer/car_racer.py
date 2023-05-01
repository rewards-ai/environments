import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pygame
import math
import os
from pydantic import BaseModel, Field
from typing import Optional, Tuple

class Config(BaseModel):
    image_raw: str = "assets/car.png"
    center: Tuple[float, float] = Field(default=(200, 50), description="The center of the image")
    velocity_vector: Tuple[float, float] = Field(default=(0.8, 0), description="The velocity vector of the object")
    rotation_velocity: float = Field(default=15, description="The rotation velocity of the object")
    show_radar: bool = Field(default=True, description="Allows radars of the render")

class Car():
    def __init__(self, radar_nums: Optional[int] = 5) -> None:
        self.radar_nums = radar_nums
        self.radar_angles = np.linspace(-90, 90, self.radar_nums)
        
        self.config = Config()
        self.show_radar = self.config.show_radar
        self.image_raw = pygame.image.load(Path(__file__).parent.joinpath(self.config.image_raw))
        self.image_raw = pygame.transform.scale(self.image_raw, (500, 500))
        
        self.reset()
    
    def reset(self) -> None:
        self.angle = 0
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
        if self.radars[0] < 20 and self.radars[-1] < 20:
            self.alive = False
        else:
            self.alive = True
            
    def drive(self) -> None:
        self.rect.center += self.velocity_vector * 12
        
    def generate_radar(self, i, radar_angle, screen) -> None:
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
        
    def rotate(self) -> None:
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.velocity_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.velocity_vector.rotate_ip(-self.rotation_vel)
            
        self.image = pygame.transform.rotozoom(self.image_raw, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

            
    def draw(self, screen) -> None:
        screen.blit(self.image, self.rect.topleft)
        if self.show_radar and self.alive:
            for radar in self.radar_locations:
                pygame.draw.line(screen, (255, 255, 255, 255), self.rect.center, radar, 1)
                pygame.draw.circle(screen, (0, 255, 0, 0), radar, 3)
        
    def step(self, action, screen) -> None:
        self.drive()
        
        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
            self.drive()
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
            
        self.rotate()
        
        for i, radar_angle in enumerate(self.radar_angles):
            self.generate_radar(i, radar_angle, screen)

        self._is_off_track()
            
        if not self.alive:
            self.reset()

class CarRacer:
    def __init__(self, screen, agent: Car = Car(), is_human: bool = False) -> None:
        self.track = pygame.image.load(Path(__file__).parent.joinpath("assets/training/track-2.png"))
        self.agent = agent
        self.screen = screen
        self.iterations = 0
        self.FPS = 20
        self.is_human = is_human
        self.clock = pygame.time.Clock()
        
        
    def _is_quitting(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
    def reset(self) -> None:
        self.iterations = 0
        self.agent.reset()
    
    def step(self, action: list = [0, 0, 0, 0, 0, 0]):
        if self.is_human:
            action = self._human_mode()
        self.agent.step(action, self.screen)
        self._is_quitting()
        
        self.clock.tick(self.FPS)
        
        return (self.agent.alive, self.agent.radars)
    
    def draw(self) -> None:
        self.screen.blit(self.track, (0, 0))
        self.agent.draw(self.screen)
        
    def render(self) -> None:
        self.iterations += 1
        self.draw()
        
    def _human_mode(self) -> list:
        action = [0] * 6
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_a]:
            action[0] = 1
        elif keys[pygame.K_d]:
            action[1] = 1
        elif keys[pygame.K_w]:
            action[2] = 1
        elif keys[pygame.K_1]:
            action[3] = 1
        elif keys[pygame.K_2]:
            action[4] = 1
        elif keys[pygame.K_3]:
            action[5] = 1
            
        return action