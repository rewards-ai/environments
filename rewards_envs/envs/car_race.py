import cv2 
import pygame 
import rewards_envs as rv 
from rewards_envs import CarConfig, CarGame
from typing import Callable, Optional, Dict, List, Any, Union, SupportsFloat


class CarRaceEnv(rv.RewardsEnv):
    def __init__(self, mode : str, track_num : int, reward_function : Callable, car_config : Optional[CarConfig] = None) -> None:
        
        _default_car_config = CarConfig(
            render_mode = "rgb_array"
        )
        
        self.game = CarGame(
            mode = mode, track_num=track_num, 
            reward_function=reward_function, 
            config=car_config if car_config else _default_car_config
        ) 
        self.reset() 
    
    def reset(self) -> Union[List[float], Dict[str, Any]]:
        return self.game.initialize() 
        
    @property 
    def current_observation(self) -> List[float]:
        """
        Returns the current observation of a particular timestamp
        """
        return self.game.radars

    @property
    def current_action(self):
        """
        Returns the current action that has been taken by the agent.
        """
        return self.game.direction
    
    def step(self, action : rv.ActType) -> tuple[rv.ObsType, SupportsFloat, bool, dict[str, Any]]:
        return self.game.step(action=action)
    
    def render(self):
        image = pygame.surfarray.array3d(self.game.screen)
        return image 
    
    def close(self):
        pygame.quit() 
        if self.game.config.render_mode == "rgb_array": cv2.destroyAllWindows() 
        