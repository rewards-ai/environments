# rewards_envs

rewards-envs is the official repository for environment store, custom environment developement and integration. When it comes to reinforcement learning, we face a hard time creating custom environment and wraping under `gymnasium`'s format. Hence introducing `rewards_envs`. With just few lines of code, Now you can create or convert your games (pygame) to cutsom environment for your agents.

### Installation and getting started.

To install rewards-envs you just have to write the command

```bash
pip install rewards_envs 
```

This installs our latest version. Current version of rewards_envs provides a single default custom environment, called `car_race`. More custom environment and integration with `gymnasium` environments will be supported in the coming versions.

In order to build our custom environment, you have to simply write these few lines of code.

```python
from rewards.environments import CarConfig, CarGame

# First create the configurations to create the environment 
# configurations helps to maintain multiple experiments 

env_config = CarConfig(
    render_mode = "rgb_array", 
    car_fps=30, screen_size = (1000, 700)
)

# create the game 
env = CarGame(
    mode = "evaluation", 
    track_num = 1, 
    reward_function = reward_function, 
    config = env_config
)
```

If you see when you are initializing the environment, there is a parameter called `reward_function` : `Callable`. This is a function that you have to define based on the given car's properties. Below is a sample reward function that works best for this environment.

```python
def reward_function(props) -> int:
    reward = 0
    if props["is_alive"]:
        reward = 1
    obs = props["observation"]
    if obs[0] < obs[-1] and props["direction"] == -1:
        reward += 1
        if props["rotational_velocity"] == 7 or props["rotational_velocity"] == 10:
            reward += 1
    elif obs[0] > obs[-1] and props["direction"] == 1:
        reward += 1
        if props["rotational_velocity"] == 7 or props["rotational_velocity"] == 10:
            reward += 1
    else:
        reward += 0
        if props["rotational_velocity"] == 15:
            reward += 1
    return reward
```

The agent (here the car) has some following properties named under the dictionary `props`. Here the name and the explaination of all the properties.

- `is_alive` : This states, whether the car is alive or not
- `observation` : Observation is a array of 5 float values, which are the radars of the car.
- `direction` : Direction provides the current action taken by the car.
- `rotational_velocity` : The rotational velocity of the car.

The properties of the car are determined during the process of creation of the game. If you want to create a custom environment, then you can define your agent's properties there. The propeties must be those, which determines whether or how much an agent is gonna win/loose the game.

### Roadmap

We want to make rewards as a general repository for RL research and RL education. Most of the RL research are heavily dependent on the environment. After environment creation, practicioners either face lot of issues wraping that environment around `gymnasium` 's custom environment wrapper or create everything of their own. We at rewards want to solve this issue. Through `rewards-sdk` and `rewards_envs` user must be able to create custom environment made using Pygame, Unity or any other engine and integrate it and start/organize RL research in no time.

### Contributing

Both `rewards` and `rewards_envs` are undergoing through some heavy developement. Being a open source projects we are open for contributions. Write now due to lack of docs, we are unable to come down with some guidelines. We will be doing that very soon. Till then please star this project and play with our sdk. If you find some bugs or need a new feature, then please create a new issue under the issues tab.
