from enum import Enum

from .base import BaseTrajectory
from .chained_polynomial import ChainedPolynomial
from .circle import Circle
from .line import Line
from .pointed_star import NPointedStar
from .polynomial import Polynomial
from .square import Square
from .zigzag import RandomZigzag
from .traj_file import TrajFile


def get_trajectory(traj_type: str, num_trajs: int, **kwargs) -> BaseTrajectory:
    return {
        'line': Line,
        'chained_poly': ChainedPolynomial,
        'circle': Circle,
        'star': NPointedStar,
        'poly': Polynomial,
        'zigzag': RandomZigzag,
        'square': Square,
        'traj_file': TrajFile
    }[traj_type](num_trajs, **kwargs)