from foragax.registry import make
from gymnax.wrappers.gym import GymnaxToGymWrapper
from gym.wrappers.flatten_observation import FlattenObservation
import numpy as np


class ConvertWrapper(GymnaxToGymWrapper):
    """Wraps a Gymnax environment to convert JAX arrays to NumPy arrays."""

    def __init__(self, env, params=None, seed=None):
        super().__init__(env, params, seed)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return np.asarray(obs), info

    def step(self, action):
        obs, reward, done, done, info = super().step(action)
        return np.asarray(obs), float(reward), bool(done), bool(done), info


def make_foragax_gym_env(env_name, **kwargs):
    """Creates a Foragax environment and wraps it for Gymnasium compatibility."""
    env = make(env_name, **kwargs)
    env = ConvertWrapper(env)
    env = FlattenObservation(env)
    return env
