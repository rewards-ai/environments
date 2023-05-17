from pydantic import BaseModel, Field
from typing import Tuple

class Config(BaseModel):
    """Car config class. Defines configuration parameters for the car agent.
    """
    image_raw: str = "assets/car.png"
    center: Tuple[float, float] = Field(default=(200, 50), description="The center of the image")
    velocity_vector: Tuple[float, float] = Field(default=(0.8, 0), description="The velocity vector of the object")
    rotation_velocity: float = Field(default=15, description="The rotation velocity of the object")
    move_factor: int = Field(default=12, description="Defines by how much factor, car should move every frame")