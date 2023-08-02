from typing import List, Optional, Tuple, Callable
import numpy as np
from pathlib import Path
import pygame
from .config import CarRacerConfig

# TODO:
#   - Work on auto-generated tracks from noise
#   - Track progress/how much track completed

class CarRacer:
    def __init__(self, config: CarRacerConfig) -> None:
        """Initializes the CarRacer class

        Args:
            config (CarRacerConfig): defines configuration for CarRacer
        """
        self._extract_config(config)
        
        self.track = self._get_track()
        self.screen = self._get_screen()
        
        self.iterations = 0
        self.num_actions = 3
        self.clock = pygame.time.Clock()
        
    def _extract_config(self, config: CarRacerConfig) -> None:
        """extracts all the varibales from config variable

        Args:
            config (CarRacerConfig): Class CarRacerConfig instance that contains configuration details of CarRacer
        """
        self.reward_function = config.reward_function
        self.require_pixel = config.require_pixel
        self.display_mode = config.display_mode
        self.window_size = config.window_size
        self.track_num = config.track_num
        self.is_human = config.is_human
        self.agents = config.agents
        self.FPS = config.FPS
            
    def _is_quitting(self) -> None:
        """Checks if pygame window is closed
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
    def reset(self) -> None:
        """Resets all the agents
        """
        self.iterations = 0
        for agent in self.agents: agent.reset()
    
    def step(self, i: int, action: list = [0, 0, 0, 0, 0, 0]):
        """Performs calculations for next step of i'th car agent

        Args:
            i (int): Denotes i'th car agent
            action (list): List of action variables. Default set to [0, 0, 0, 0, 0, 0].

        Returns:
            reward (int): Total generated reward based on a reward function
            is_done (bool): negation of alive variable of i'th car agent
            pixel_data (list): Pixel data of the main pygame suface/window
        """
        
        if self.is_human: action = self._human_mode()
        self.agents[i].step(action, self.screen)
        self.clock.tick(self.FPS)
                
        reward = self.reward_function(self.agents[i].params)
        pixel_data = pygame.surfarray.array3d(self.screen) if self.require_pixel else None
        
        return reward, not self.agents[i].alive, pixel_data
    
    def _draw(self) -> None:
        """Draws the track and all the car agents onto the pygame screen
        """
        self.screen.blit(self.track, (0, 0))
        for i, agent in enumerate(self.agents): 
            if not agent.isDisabled:
                agent._draw(i, self.screen)
        
    def render(self) -> None:
        """Manages iterations and draw function
        """
        self.iterations += 1
        self._draw()
        if self.display_mode == "window": pygame.display.update()
        
    def _human_mode(self) -> list:
        """Manages "human" mode of car agent control using keyboard.

        Returns:
            action (list): List generated action list based on keyboard input
        """
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
    
    def get_state(self, i: int):
        """Returns the state of the i'th agent

        Args:
            i (int): denotes index of the agent in the list 'agents'

        Returns:
            state (numpy.array): state of the agent as a numpy array
        """
        state = self.agents[i].radars
        return np.array(state, dtype=int)

    def _get_track(self):
        """Loads path of the track image based on 'track_num' and returns image as pygame image

        Returns:
            track: pygame image of track
        """
        parent_path = Path(__file__).parent
        track_path = parent_path.joinpath(f"assets/training/track-{self.track_num}.png")
        track = pygame.image.load(track_path)

        return track

    def _get_screen(self):
        """Returns the screen to blit/render the environment and agents, based on 'display_mode'

        Returns:
            Union[pygame.display, pygame.Surface]: Either a pygame.display object or a pygame.Surface object.
        """
        if self.display_mode == "window":
            return pygame.display.set_mode(self.window_size)
        return pygame.Surface(self.window_size)