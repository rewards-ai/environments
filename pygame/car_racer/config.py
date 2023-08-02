from pydantic import BaseModel, Field, validator
from typing import Tuple, List, Callable, Optional
from .car import Car
from pathlib import Path

class CarRacerConfig(BaseModel):
    agents: List
    reward_function: Callable[[dict], int]
    track_num: Optional[int] = 1
    is_human: Optional[bool] = False
    display_mode: Optional[str] = "window"
    window_size: Optional[Tuple[int, int]] = (800, 700)
    FPS: Optional[int] = 100
    require_pixel: Optional[bool] = False
    
    @validator('agents')
    def validate_agents(cls, value):
        if type(value) == list:
            if len(value) == 0:
                raise ValueError("\nEmpty list of agents was passed into 'agents'")
            elif type(value[0]) != Car:
                raise ValueError("\nThe 'agents' attribute should be a list of 'Car' objects.")
        elif type(value) == Car:
            return [value]
        return value
    
    @validator('is_human')
    def validate_is_human(cls, value):
        if value and len(cls.agents) == 1:
            raise ValueError("\n\"human\" mode does not work with multi-agents")
        return value
    
    @validator('display_mode')
    def validate_display_mode(cls, value):
        if value not in ["window", "surface"]:
            raise ValueError("\ndisplay_mode can be 'window' or 'surface'. For more info read docs.")
        return value
        
    @validator('track_num')
    def validate_track_num(cls, value) -> int:
        tracks_path = Path(__file__).parent.joinpath("assets/training")
        tracks_num = len([file_path for file_path in tracks_path.iterdir()])
        
        if value < 1:
            raise ValueError("\ntrack variable uses '1-based indexing'")
        elif value > tracks_num:
            raise ValueError(f"\nOnly {tracks_num} tracks available.")
        return value
    
    def __init__(self, **args):
        """Overridden __init__ function for checking unexpected parameters

        Raises:
            ValueError: When unexpeceted parameters are found
        """
        unexpected_variables = set(args.keys()) - set(self.__fields__)
        if unexpected_variables:
            raise ValueError(f"\nUnexpected variables found: {unexpected_variables}")
        super().__init__(**args)