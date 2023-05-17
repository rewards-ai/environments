import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pygame
from typing import Union, Optional
from .car import Car

class CarRacer:
    def __init__(self, screen, agents: Optional[list] = [Car()], is_human: Optional[bool] = False) -> None:
        """Initializes CarRacer class. This class runs and managed all the car agents.

        Args:
            screen: Pygame display window variable
            agents (Union[list, Car], optional): List of . Defaults to Car().
            is_human (bool, optional): Flag to check if environment is to be controlled using "human" or "algorithm". Defaults to False.
        """
        self.track = pygame.image.load(Path(__file__).parent.joinpath("assets/training/track-3.png"))
        self.agents = agents if type(agents) == list else [agents]
        self.screen = screen
        self.iterations = 0
        self.FPS = 100
        self.is_human = is_human
        self.clock = pygame.time.Clock()
        
        assert not self.is_human or len(agents) <= 1, "\"human\" mode does not work with multi-agents"
        
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
        """Increases step of i'th car agent

        Args:
            i (int): Denotes i'th car agent
            action (list, optional): List of action variables. Defaults to [0, 0, 0, 0, 0, 0].

        Returns:
            not alive (bool): negation of alive variable of i'th car agent
            reward (int): Total generated reward based on a reward function
            pixel_data (list): Pixel data of the main pygame suface
        """
        # TODO: generate pixel_data
        pixel_data = []
        
        if self.is_human:
            action = self._human_mode()
        self.agents[i].step(action, self.screen)
        self._is_quitting()
        self.clock.tick(self.FPS)
        
        reward = 1 if self.agents[i].alive else 0
            
        return reward, not self.agents[i].alive, pixel_data
    
    def _draw(self) -> None:
        """Draws the track and all the car agents onto the pygame screen
        """
        self.screen.blit(self.track, (0, 0))
        for i, agent in enumerate(self.agents): agent._draw(i, self.screen)
        
    def render(self) -> None:
        """Manages iterations and draw function
        """
        self.iterations += 1
        self._draw()
        self.clock.tick(self.FPS)
        
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