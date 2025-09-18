from foragax.registry import make
from gymnax.wrappers.gym import GymnaxToGymWrapper


def make_foragax_gym_env(env_name, **kwargs):
    """Creates a Foragax environment and wraps it for Gymnasium compatibility."""
    env = make(env_name, **kwargs)
    return GymnaxToGymWrapper(env)
