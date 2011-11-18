
from __future__ import division

import numpy as np
try:
    from scipy.optimize import fmin_cg
except ImportError:
    def fmin_cg(*args, **kwds):
        raise RuntimeError('fmin_cg not available; install scipy to use this functionality')

from base_components import (Parameters, NeighborsTable, ForceField, LJForceFeild,
                             BaseConfig, System)
from util import periodic_distances


def create_velocities(N, T=1.0, mass=1.0):
    return np.random.normal(scale=np.sqrt(T / mass), size=(N, 3))

class Config(BaseConfig):

    @classmethod
    def create(cls, N, rho, dt, sigma=1.0, T=1.0, mass=1.0):
        V = N * sigma**3 / rho
        box_size = V**(1/3)
        positions = np.random.uniform(0.0, box_size, (N, 3))
        inst = cls(positions, positions, box_size, dt)
        inst.randomize_velocities(T=T, mass=mass)
        return inst

    def with_new_positions(self, new_positions):
        cp = self.__class__(new_positions, None, self.box_size, self.dt)
        cp.set_velocities(self.calculate_velocities())
        return cp

    def randomize_velocities(self, **kwds):
        self.set_velocities(create_velocities(self.N_particles, **kwds))

    def set_velocities(self, velocities):
        self.last_positions = self.positions - self.dt * velocities

    def calculate_velocities(self):
        return (self.positions - self.last_positions) / self.dt

    def calculate_rms_velocity(self):
        v = self.calculate_velocities()
        return (v**2).sum(axis=1).mean()**0.5

    def calculate_kinetic_energy(self, mass=1.0):
        v_rms = self.calculate_rms_velocity()
        return 0.5 * mass * v_rms**2

    def calculate_temperature(self, mass=1.0):
        KE = self.calculate_kinetic_energy(mass)
        return 2 * KE / 3

    @property
    def N_particles(self):
        return len(self.positions)

    def propagate(self, forces, mass=1.0):
        positions = -self.last_positions + 2 * self.positions + (self.dt**2 / mass) * forces
        self.last_positions = self.positions
        self.positions = positions


class NeighborsTableTracker(object):

    def __init__(self, neighbors_table, box_size):
        self.neighbors_table = neighbors_table
        self.box_size = box_size
        self.last_positions = None

    def maybe_rebuild_neighbor(self, current_positions):
        if self.last_positions is None:
            self.rebuild_neighbors(current_positions)
        else:
            delta = periodic_distances(current_positions, self.last_positions, self.box_size)
            self.acc_delta += delta

            d_r2 = (self.acc_delta**2).sum(axis=1)
            if d_r2.max() > 0.4:
                self.rebuild_neighbors(current_positions)

        self.last_positions = current_positions

    def rebuild_neighbors(self, current_positions):
        self.neighbors_table.rebuild_neighbors(current_positions, self.box_size)
        self.acc_delta = np.zeros_like(current_positions)




class EnergyMinimzer(object):

    def __init__(self, config_init, forcefield, maxiter=100, neighbors_table=None, neighbors_table_skin=1.0):
        self.config_init = config_init
        self.forcefield = forcefield
        self.maxiter = maxiter

        if neighbors_table is None:
            neighbors_table = NeighborsTable(
                r_forcefield_cutoff=forcefield.r_cutoff,
                r_skin=neighbors_table_skin)
        self.neighbors_table = neighbors_table

        self.forces = np.empty_like(self.config_init.positions)
        self.neighbors_table_tracker = NeighborsTableTracker(self.neighbors_table, self.config_init.box_size)

    def minimize(self):
        [x_final, self.U_min, self.n_func_calls, self.n_grad_calls, self.warnfalgs
         ] = fmin_cg(self.evaluate_potential,
                     self.config_init.positions,
                     fprime=self.evaluate_gradient,
                     maxiter=self.maxiter,
                     callback=self.callback,
                     full_output=True, disp=False)
        self.config_final = self.config_init.with_new_positions(self.create_positions(x_final))
        return self.config_final

    def evaluate_potential(self, x):
        return self.forcefield.evaluate_potential(self.create_positions(x),
                                                  self.config_init.box_size,
                                                  self.neighbors_table)

    def evaluate_gradient(self, x):
        self.forcefield.evaluate_forces(self.forces,
                                        self.create_positions(x),
                                        self.config_init.box_size,
                                        self.neighbors_table)
        return -self.forces.reshape(3 * self.config_init.N_particles)

    def callback(self, x):
        self.neighbors_table_tracker.maybe_rebuild_neighbor(self.create_positions(x))

    def create_positions(self, x):
        return x.reshape((self.config_init.N_particles, 3)) % self.config_init.box_size






