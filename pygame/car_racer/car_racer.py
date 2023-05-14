import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pygame
from typing import Union
from .car import Car

class CarRacer:
    def __init__(self, screen, agents: Union[list, Car] = Car(), is_human: bool = False) -> None:
        self.track = pygame.image.load(Path(__file__).parent.joinpath("assets/training/track-2.png"))
        self.agents = agents if type(agents) == list else [agents]
        self.screen = screen
        self.iterations = 0
        self.FPS = 20
        self.is_human = is_human
        self.clock = pygame.time.Clock()
        
        assert not self.is_human or len(agents) <= 1, "\"human\" mode does not work with multi-agents"
        
    def _is_quitting(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
    def reset(self) -> None:
        self.iterations = 0
        for agent in self.agents: agent.reset()
    
    def step(self, action: list = [0, 0, 0, 0, 0, 0]):
        if self.is_human:
            action = self._human_mode()
        for agent in self.agents: agent.step(action, self.screen)
        self._is_quitting()
        self.clock.tick(self.FPS)
        
        return ([agent.alive for agent in self.agents], [agent.radars for agent in self.agents])
    
    def _draw(self) -> None:
        self.screen.blit(self.track, (0, 0))
        for agent in self.agents: agent._draw(self.screen)
        
    def render(self) -> None:
        self.iterations += 1
        self._draw()
        
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