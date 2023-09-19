from typing import Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pygame
import random
import math

# TODO:
#   - Work on different speed controls
#     (Code is done, but first we need progress_completed variable from track as
#      without progress_completed or time, car learns to collect max rewrds by driving
#      slow.)

class Car():
    def __init__(
            self,
            id: str,
            angle,
            center,
            radar_nums: Optional[int] = 5, 
            show_radar: Optional[bool] = False,
            color: Optional[Tuple[int, int, int]] = None
        ) -> None:
        """Initialize the Car(Agent) class

        Args:
            id (str): unique id given to the agent. Used for agent and model recoginzation
            radar_nums (int, optional): number of radars for the car agent. Defaults to 5.
            show_radar (bool, optional): Bool value to toogle if radar to be displayed or not. Defaults to False.
            color (Tuple[int, int, int], optional): RGB color, given to the agent. Defaults to None which generates random color.
        """        
        self.id = id
        self.radar_nums = radar_nums
        self.show_radar = show_radar
        self.radar_angles = np.linspace(-90, 90, self.radar_nums)
        
        self.initial_angle = angle
        self.initial_center = center
        
        self.image_raw = "assets/car.png"
        self.initital_rotation_velocity = 15
        self.move_factor = 12
        
        self.image_raw = pygame.image.load(Path(__file__).parent.joinpath(self.image_raw))
        self.image_raw = pygame.transform.scale(self.image_raw, (500, 500))
        self._spray_paint(color)
        
        self.reset()
    
    def reset(self) -> None:
        """Reset the Car back to the starting position. Reset other car variables too.
        """
        self.angle = self.initial_angle
        randians = math.radians(self.angle)
        self.reward = 0
        
        self.velocity_vector = (math.cos(randians), - math.sin(randians))
        self.center = self.initial_center
        
        self.image = pygame.transform.rotozoom(self.image_raw, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.center)
        
        self.velocity_vector = pygame.math.Vector2(self.velocity_vector)
        self.rotation_velocity = self.initital_rotation_velocity
        
        self.radar_locations = [self.rect.center] * self.radar_nums
        
        self.reward = 0
        self.alive = True
        self.direction = 0
        self.radars = [0] * self.radar_nums
        self.isDisabled = False
    
        self.params = {}
        self._update_params()
        
    def _update_params(self) -> None:
        """Updates the "self.params" variable for reward_function
        """
        self.params = {
            "is_alive": self.alive,
            "direction": self.direction,
            "radars": self.radars,
            "radar_angles": self.radar_angles,
            "car_location": self.rect.center,
            "steering_angle": self.angle
        }
    def _spray_paint(self, color: Any) -> None:
        """Colors the template of the car image, using given RGB value.
           If "None" is passed, then randomly chooses a color 

        Args:
            color (Tuple[int ,int, int]): A tuple of RGB values each ranging 0-255
        """
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ) if color is None else color
            
        color += tuple([255]) # For alpha in rbga
        light_shade = tuple([c+10 if c+10 < 255 else 255 for c in color])
        
        color_map = {
            (175, 177, 190, 255): color,
            (228, 228, 228, 255): light_shade
        }
        
        width = self.image_raw.get_width()
        height = self.image_raw.get_height()
        
        for x in range(width):
            for y in range(height):
                pixel = self.image_raw.get_at((x, y))
                try: self.image_raw.set_at((x, y), color_map[tuple(pixel)])
                except: pass
        
    def _is_off_track(self) -> bool:
        """Checks if Car is off the track using first and last radars. Also Checks if Car is off screen

        Returns:
            bool: True if car is off-track else False
        """
        x = self.rect.center[0]
        y = self.rect.center[1]
        
        if (self.radars[0] < 20 and self.radars[-1] < 20) or \
            ((x < 0 or x > 800) or (y < 0 or y > 700)):
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
            self.angle -= self.rotation_velocity
            self.velocity_vector.rotate_ip(self.rotation_velocity)
        if self.direction == -1:
            self.angle += self.rotation_velocity
            self.velocity_vector.rotate_ip(-self.rotation_velocity)
            
        self.image = pygame.transform.rotozoom(self.image_raw, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

            
    def _draw(self, i: int, screen: Any) -> None:
        """Draws Car to the screen

        Args:
            i (int): i'th car in the environement
            screen: Pygame main screen display variable
        """
        screen.blit(self.image, self.rect.topleft)
        
        font = pygame.font.Font(None, 24)
        text_surface = font.render(self.id, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = self.rect.topleft
        screen.blit(text_surface, text_rect)
        
        if self.show_radar and self.alive:
            for radar in self.radar_locations:
                pygame.draw.line(screen, (255, 255, 255, 255), self.rect.center, radar, 1)
                pygame.draw.circle(screen, (0, 255, 0, 0), radar, 3)
                
    def disable(self) -> None:
        self.isDisabled = True
        
    def step(self, action: list, screen) -> None:
        """Drives, rotates and check off-track of the car based on action list.

        Args:
            action (list): List of actions to display what move it=s taken.
            screen: Pygame main screen display variable
        """
        # TODO: commented code controls car speed, need to work on that
        self._drive()
        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
        # elif action[3] == 1:
        #     self.velocity_vector.scale_to_length(0.8)
        #     self.rotation_velocity = 15
        # elif action[4] == 1:
        #     self.velocity_vector.scale_to_length(1.2)
        #     self.rotation_velocity = 10
        # elif action[5] == 1:
        #     self.velocity_vector.scale_to_length(1.6)
        #     self.rotation_velocity = 7
        else:
            self.direction = 0
            
        self._rotate()
        for i, radar_angle in enumerate(self.radar_angles):
            self._generate_radar(i, radar_angle, screen)

        self._is_off_track()
        self._update_params()