"""
Copyright (c) rewards.ai 

This program mainly serves as the pygame engine for the car environment. 
User can either use it to play the game or use it to the rewards environment. 
There are some sets of methods and attributes that are fixed here:

TODO:
-----
- Able to add more tracks 
- Add the functionality to play the game in the human mode. Current version has some bugs in 
`human` mode 
"""

import os
import math
import pygame
from pathlib import Path
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, SupportsFloat


class CarConfig(BaseModel):
    car_scale: int = Field(default=500)
    drive_factor: int = Field(default=12)
    car_fps: int = Field(default=15)
    car_angle: int = Field(default=0)

    car_rect_size: Tuple[int, int] = Field(default=(200, 50))
    car_velocity_vector: Tuple[float, float] = Field(default=(0.8, 0.0))
    car_rotational_velocity: Union[int, float] = Field(default=15)
    car_direction: Union[int, float] = Field(default=0)
    car_is_alive: bool = Field(default=True)
    car_reward: int = Field(default=(0,))
    car_radar: List[Union[int, float]] = Field(default=[0, 0, 0, 0, 0])

    render_mode: str = Field(default="human", title="Frame rendering mode")
    render_fps: int = Field(
        default=30, title="Frames per second for rendering")
    screen_size : Tuple[int, int] = Field(default=(800, 700))


class CarGame:
    def __init__(self, mode: str, track_num: int, reward_function: Optional[Callable] = None, config: Optional[CarConfig] = None):
        """TBD

        Args:
            mode (str): This indicates in which mode the car environment is in. 
            It is required for track selection
            config (Optional[CarConfig], optional): All the car configurations. Defaults to None.
            reward_function (Optional[Callable], optional): The reward function for car's evaluation
        """
        self.mode = mode
        self.parent_path = str(Path(__file__).parent)
        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'modes' : ['training', 'evaluation'], 
            'render_fps': 30,
            'training_car_tracks': {
                1: "track-1.png",
                2: "track-2.png",
                3: "track-3.png",
            },
            'evaluation_car_tracks': {
                1: "track-1.png"
            },
            'car_image' : 'car.png', 
            'assets_path': str(os.path.join(self.parent_path, "assets"))
        }

        self.config = CarConfig() if config is None else config
        self.reward_function = self._default_reward_function if reward_function is None else reward_function

        assert self.config.render_mode in self.metadata['render_modes']
        assert self.mode in self.metadata['modes']
        assert (self.mode == "training" and (track_num >= 1 or track_num <= 3)) or (self.mode == "evaluation" and track_num == 1)
        
        # Load car and track path
        self.track_options = self.metadata['training_car_tracks'] if mode == "training" else self.metadata['evaluation_car_tracks']
        self.track_image_path = os.path.join(
            self.metadata['assets_path'], mode, self.track_options[track_num]
        ).replace(os.sep, '/')
        self.car_image_path = os.path.join(self.metadata['assets_path'], self.metadata['car_image'])
        
        # Building Car and Track Paths 
        self.track_image = pygame.image.load(self.track_image_path)
        self.car_image = pygame.transform.scale(
            pygame.image.load(
                self.car_image_path), (self.config.car_scale, self.config.car_scale),
        )
        
        # Building PyGame Screen only if the choosen rendering mode is "human"
        self.screen = pygame.display.set_mode(
            self.config.screen_size) if self.config.render_mode == "human" else pygame.Surface(self.config.screen_size)
        self.screen.fill((0, 0, 0))
        
        # All the basic car configurations
        self.angle = self.config.car_angle
        self.original_image = self.car_image
       
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.config.car_rect_size)
        self.vel_vector = pygame.math.Vector2(self.config.car_velocity_vector)
        self.rotation_vel = self.config.car_rotational_velocity
        
        self.direction = self.config.car_direction # current action 
        self.drive_factor = self.config.drive_factor
        self.alive = self.config.car_is_alive # current terminal 
        self.radars = self.config.car_radar   # current state space 
        self.reward = 0 # current reward 
        
         # Additional configuration
        self.clock = pygame.time.Clock()
        self.track = self.track_image
        self.iterations = 0
        self.FPS = self.config.car_fps
        
        # current observation space 
        self.params = {
            "is_alive": None,
            "observation": None,
            "direction": None,
            "rotational_velocity": None,
        }
        
        self.info =  {
            'direction' : self.direction,
            'rotational_velocity' : self.rotation_vel
        }
        print("=> Game initialization finished")
        
    def _default_reward_function(self, props: Dict[str, Any]) -> Union[int, float]:
        if props["is_alive"]:
            return 1
        return 0
    
    def initialize(self) -> Union[List[float], Dict[str, Any]]:
        """
        Initializes the car environment with all the default properties
        
        Returns:
            info : List[List[float], Dict[str, Any]]. A List that returns the current environment state and the current environment information. 
        """
        self.angle = self.config.car_angle
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.config.car_rect_size)
        self.vel_vector = pygame.math.Vector2(self.config.car_velocity_vector)
        self.rotation_vel = self.config.car_rotational_velocity
        self.direction = self.config.car_direction
        self.drive_factor = self.config.drive_factor
        self.alive = self.config.car_is_alive
        self.radars = self.config.car_radar
        self.reward = 0

        self.info['direction'] = self.direction
        self.info['rotational_velocity'] = self.rotation_vel
        return self.radars, self.info 
        
        
    def _did_quit(self):
        """
        Quits the game when user presses the 'quit'/'close' key.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
    
    def _did_collide(self):
        """
        Checks the status whether the car collied or not. If the car collides, 
        then `isAlive` is False and game terminates.

        TODO: 
        -----
        - This function needs to be checked
        """

        length = 20  # parameter to be know n
        collision_point_right = [
            int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle + 18)) * length
            ),
            int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle + 18)) * length
            ),
        ]

        collision_point_left = [
            int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle - 18)) * length
            ),
            int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle - 18)) * length
            ),
        ]

        try:
            if self.screen.get_at(collision_point_right) == pygame.Color(
                173, 255, 133, 255
            ) or self.screen.get_at(collision_point_left) == pygame.Color(
                173, 255, 133, 255
            ):
                self.alive = False

            pygame.draw.circle(
                self.screen, (0, 255, 255, 0), collision_point_right, 4
            )
            pygame.draw.circle(
                self.screen, (0, 255, 255, 0), collision_point_left, 4
            )

        except:
            self.alive = False
            
    
    def _did_rotate(self):
        """
        Checks whether the car rotates off the track and took wrong direction or not

        TODO: 
        -----
        The function implementation needs to be checked
        """

        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 0.1
        )
        self.rect = self.image.get_rect(center=self.rect.center)
    
    
    def _update_radar(self, i: int, radar_angle: Union[int, float]) -> None:
        """
        The Car is made up of 6 radars. At every step this functions updates the radar to get 
        the current direction and also updates the overall current status of the car.

        Args:
            i (int): The current index number
            radar_angle (Union[int, float]): The current angles in the radar.
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while (
                not self.screen.get_at(
                    (x, y)) == pygame.Color(173, 255, 133, 255)
                and length < 200
            ):
                length += 1
                x = int(
                    self.rect.center[0]
                    + math.cos(math.radians(self.angle + radar_angle)) * length
                )
                y = int(
                    self.rect.center[1]
                    - math.sin(math.radians(self.angle + radar_angle)) * length
                )

            pygame.draw.line(
                self.screen, (255, 255, 255, 255), self.rect.center, (x, y), 1
            )
            pygame.draw.circle(self.screen, (0, 255, 0, 0), (x, y), 3)

            dist = int(
                math.sqrt(
                    math.pow(self.rect.center[0] - x, 2)
                    + math.pow(self.rect.center[1] - y, 2)
                )
            )
            self.radars[i] = dist
        except:
            self.alive = False
            
    def _drive(self):
        """
        Drives the car's center vector to the next state
        TODO
        ----
        - Need to check why the value is 12 and nothing else
        """
        self.rect.center += self.vel_vector * self.drive_factor

    def draw(self) -> None:
        """
        Draws the car on the screen
        """
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)
    
    
    def timeTicking(self):
        self.clock.tick(self.FPS)

    def get_current_state(self, action: List[int]) -> Dict[str, Any]:
        """
        Returns the current state of the car. This states are determined the parameters 
        of the Car mentioned below:
            - `isAlive` : This parameter determine whether the game is finished or not
            - `obs` : This represents the car's current observation. 
            - `dir` : This represents the car's current direction 
            - `rotationVel` : This represents the current rotational velocity of the car.
        Where each of the parameters are the keys of the dictionary returned by this function.

        Args:
            action (List[int]): The current action of the agent

        Returns:
            Dict[str, Any]: The current state of the car
        """
        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
            self._drive()
        elif action[3] == 1:
            self.vel_vector.scale_to_length(0.8)
            self.rotation_vel = 15
        elif action[4] == 1:
            self.vel_vector.scale_to_length(1.2)
            self.rotation_vel = 10
        elif action[5] == 1:
            self.vel_vector.scale_to_length(1.6)
            self.rotation_vel = 7
        else:
            self.direction = 0

        self._did_rotate()
        self._did_collide()

        for i, radar_angle in enumerate((-60, -30, 0, 30, 60)):
            self._update_radar(i, radar_angle)  # to be implemented

        if (
            self.radars[0] < 15
            and self.radars[4] < 15
            and self.radars[1] < 25
            and self.radars[2] < 25
            and self.radars[3] < 25
        ):
            self.alive = False
        else:
            self.alive = True

        self.params = {
            "is_alive": self.alive,
            "observation": self.radars,
            "direction": self.direction,
            "rotational_velocity": self.rotation_vel,
        }

        return self.params

    
    def step(self, action: List[int]) -> List[Any]:
        """
        Plays a single step of the game. This function is called by the agent during each step.
        Current version of rewards-envs does not provides `info` of that given step. However,it 
        will be supported in the future versions. 

        Args:
            action (List[int]): The current action of the agent

        Returns:
            List[Any]: [observation, reward, terminated, info]
        """
        self.iterations += 1
        if self.config.render_mode == "human":
            self._did_quit()
        self.draw()
        self.radars = [0, 0, 0, 0, 0]
        self._drive()
        
        current_state_params = self.get_current_state(action=action)
        current_reward = self.reward_function(current_state_params)
        self.reward += current_reward
        
        observation = current_state_params['observation']
        terminated = current_state_params['is_alive']
        return observation, self.reward, terminated, self.info 

    
    # TODO: Needs to deprecated soon in the next major release
    def play_step(self, action: List[int]) -> List[Any]:
        """
        Plays a single step of the game. This function is called by the agent during each step.

        Args:
            action (List[int]): The current action of the agent

        Returns:
            List[Any]: [current_reward, is_alive, overall_reward]
        """
        self.iterations += 1
        
        if self.config.render_mode == "human":
            self._did_quit()
            
        self.draw()
        self.radars = [0, 0, 0, 0, 0]
        self._drive()

        current_state_params = self.get_current_state(action=action)
        current_reward = self.reward_function(current_state_params)
        self.reward += current_reward

        if self.config.render_mode == "rgb_array":
            pixel_data = pygame.surfarray.array3d(self.screen)
            return current_reward, not self.alive, self.reward, pixel_data
        else:
            return current_reward, not self.alive, self.reward
    

if __name__ == '__main__':
    car_game = CarGame(mode="training", track_num=1)
