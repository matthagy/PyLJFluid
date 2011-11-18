
from __future__ import division

import numpy as np

from base_components import (Parameters, NeighborsTable, ForceField, LJForceFeild,
                             BaseConfig, System)


class Config(BaseConfig):

    @classmethod
    def create(cls, N, rho, sigma=1.0, T=1.0, mass=1.0):
        V = N * sigma**3 / rho
        box_size = V**(1/3)
        positions = np.random.uniform(0.0, box_size, (N, 3))
        velocities = np.random.normal(scale=np.sqrt(T / mass), size=(N, 3))
        v_rms =  (velocities**2).sum(axis=1).mean() ** 0.5
        return cls(positions, positions - velocities, box_size)

    def calculate_velocities(self):
        return self.positions - self.last_positions

    def calculate_rms_velocity(self):
        v = self.calculate_velocities()
        return (v**2).sum(axis=1).mean()**0.5

    def calculate_temperature(self, mass=1.0):
        v_rms = self.calculate_rms_velocity()
        return v_rms**2 * mass / 3
