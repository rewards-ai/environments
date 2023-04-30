import rewards_envs
from rewards_envs.core import RewardsEnv, ActType, ObsType, RenderFrame

# importing all the game engines

from rewards_envs.engines import pygame
from rewards_envs.engines.pygame.car_race.car_race import CarConfig, CarGame

# importing all the environment

from rewards_envs import envs
from rewards_envs.envs.car_race import CarRaceEnv
