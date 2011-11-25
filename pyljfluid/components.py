
from __future__ import division

import numpy as np
try:
    from scipy.optimize import fmin_cg
except ImportError:
    def fmin_cg(*args, **kwds):
        raise RuntimeError('fmin_cg not available; install scipy to use this functionality')

from base_components import (NeighborsTable, ForceField, LJForceField,
                             BaseConfig, BasePairCorrelationFunctionCalculator)
from util import periodic_distances


__all__ = ['NeighborsTable', 'ForceField', 'LJForceField',
           'Config', 'NeighborsTableTracker', 'EnergyMinimzer',
           'MDSimulator',
           'PairCorrelationData', 'PairCorrelationFunctionCalculator',
           'PairCorrelationIntegrator', 'LJPairCorrelationIntegrator']


def create_velocities(N, T=1.0, mass=1.0):
    return np.random.normal(scale=np.sqrt(T / mass), size=(N, 3))

def calculate_box_size(N, sigma, rho):
    V = N * sigma**3 / rho
    return V**(1/3)


class Config(BaseConfig):

    @classmethod
    def create(cls, N, rho, dt, sigma=1.0, T=1.0, mass=1.0):
        box_size = calculate_box_size(N, sigma, rho)
        positions = np.random.uniform(0.0, box_size, (N, 3))
        inst = cls(positions, None, box_size, dt)
        inst.randomize_velocities(T=T, mass=mass)
        inst.sigma = sigma
        return inst

    def copy(self):
        cp = self.__class__(self.positions.copy(),
                            self.last_positions.copy(),
                            self.box_size,
                            self.dt)
        cp.sigma = self.sigma
        return cp

    def with_new_positions(self, new_positions):
        cp = self.__class__(new_positions, None, self.box_size, self.dt)
        cp.set_velocities(self.calculate_velocities())
        cp.sigma = self.sigma
        return cp

    def randomize_velocities(self, **kwds):
        self.set_velocities(create_velocities(self.N, **kwds))

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
    def N(self):
        return len(self.positions)

    @property
    def V(self):
        return self.box_size**3

    @property
    def rho(self):
        return self.N / self.V

    def rescale_boxsize(self, new_boxsize):
        velocities = self.calculate_velocities()
        self.positions %= self.box_size
        self.positions *= new_boxsize / self.box_size
        self.box_size = new_boxsize
        self.set_velocities(velocities)

    def rescale_boxsize_rho(self, new_rho):
        self.rescale_boxsize(calculate_box_size(self.N, self.sigma, new_rho))

    def normalize_positions(self):
        velocities = self.calculate_velocities()
        self.positions %= self.box_size
        self.set_velocities(velocities)

    def change_dt(self, dt):
        velocities = self.calculate_velocities()
        self.dt = dt
        self.set_velocities(velocities)

    def propagate(self, forces, mass=1.0):
        positions = -self.last_positions + 2 * self.positions + (self.dt**2 / mass) * forces
        self.last_positions = self.positions
        self.positions = positions

    def __reduce__(self):
        return (Config, (self.positions, self.last_positions, self.box_size, self.dt))


class NeighborsTableTracker(object):

    def __init__(self, neighbors_table, box_size):
        self.neighbors_table = neighbors_table
        self.box_size = box_size
        self.last_positions = None

    def reset(self, current_positions):
        self.neighbors_table.rebuild_neighbors(current_positions, self.box_size)
        self.last_positions = current_positions
        if self.acc_delta is None:
            self.acc_delta = np.empty_like(current_positions)
        self.acc_delta.fill(0.0)

    def moved(self, current_positions, check_were_valid=True):
        delta = periodic_distances(current_positions, self.last_positions, self.box_size)
        self.last_positions = current_positions
        self.acc_delta += delta
        dr_max = (self.acc_delta**2).sum(axis=1).max()**0.5
        r_acceptable = 0.5 * self.neighbors_table.r_skin
        if dr_max > r_acceptable:
            return self.rebuild_neighbors(current_positions, check_were_valid=check_were_valid)
        return True

    acc_delta = None

    def rebuild_neighbors(self, current_positions, check_were_valid=True):
        if check_were_valid:
            old_vital_neighbors = self.find_vital_neighbors(current_positions)
        self.neighbors_table.rebuild_neighbors(current_positions, self.box_size)
        if check_were_valid:
            new_vital_neighbors = self.find_vital_neighbors(current_positions)
            were_valid = old_vital_neighbors >= new_vital_neighbors
        self.acc_delta.fill(0.0)
        if check_were_valid:
            return were_valid

    def find_vital_neighbors(self, positions):
        return self.neighbors_table.find_set_of_neighbors_within_distance(self.neighbors_table.r_forcefield_cutoff,
                                                                          positions, self.box_size)

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
        self.neighbors_table_tracker.reset(self.config_init.positions)
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
        return -self.forces.reshape(3 * self.config_init.N)

    def callback(self, x):
        self.neighbors_table_tracker.moved(self.create_positions(x))

    def create_positions(self, x):
        return x.reshape((self.config_init.N, 3)) % self.config_init.box_size



class PairCorrelationFunctionCalculator(BasePairCorrelationFunctionCalculator):

    def accumulate_config(self, config):
        self.accumulate_positions(config.positions, config.box_size)

    def calculate_rs(self):
        return self.r_min + self.r_prec * np.arange(self.N_bins)

    def calculate_gs(self):
        N = self.bins.sum()
        if not N:
            return None

        r_max = self.N_bins * self.r_prec
        V = 4.0 / 3.0 * np.pi * r_max**3
        rho = N / V

        rs = self.calculate_rs()
        vs = 4.0 / 3.0 * np.pi * ((rs + self.r_prec)**3 - rs**3)
        rhos =  self.bins / vs

        return rhos / rho

    def calculate_hs(self):
        return self.calculate_gs() - 1.0



class MDSimulator(object):

    def __init__(self, config, forcefield, mass=1.0, r_skin=1.0):
        self.config = config
        self.forcefield = forcefield
        self.forces = np.empty_like(config.positions)
        self.mass = mass
        self.neighbors_table = NeighborsTable(r_forcefield_cutoff=self.forcefield.r_cutoff,
                                              r_skin=r_skin)
        self.neighbors_table_tracker = NeighborsTableTracker(self.neighbors_table, self.config.box_size)
        self.neighbors_table_tracker.reset(self.config.positions)

        self.backup_positions = np.empty_like(self.config.positions)
        self.backup_last_positions = np.empty_like(self.config.last_positions)

    normalize_positions_rate = 20

    def cycle(self, n=1):
        for i in xrange(n):
            if not i%self.normalize_positions_rate:
                self.normalize_positions()

            self.backup_positions[...] = self.config.positions
            self.backup_last_positions[...] = self.config.last_positions
            were_neighbors_valid = self.propagate_attempt()
            if not were_neighbors_valid:
                self.config.positions[...] = self.backup_positions
                self.config.last_positions[...] = self.backup_last_positions
                were_neighbors_valid = self.propagate_attempt()
                if not were_neighbors_valid:
                    raise RuntimeError("couldn't create valid neighbor list, increase r_skin or decrease dt")

    def propagate_attempt(self):
        self.forcefield.evaluate_forces(self.forces,
                                        self.config.positions % self.config.box_size,
                                        self.config.box_size, self.neighbors_table)
        self.config.propagate(self.forces, self.mass)
        return self.neighbors_table_tracker.moved(self.config.positions, check_were_valid=True)

    def normalize_positions(self):
        self.config.normalize_positions()
        self.neighbors_table_tracker.reset(self.config.positions)

    def rescale_boxsize_rho(self, rho):
        self.config.rescale_boxsize_rho(rho)
        self.neighbors_table_tracker.reset(self.config.positions)

    def evaluate_potential(self):
        return self.forcefield.evaluate_potential(self.config.positions % self.config.box_size,
                                                  self.config.box_size, self.neighbors_table)

    def minimize(self, maxiter=10):
        minimizer = EnergyMinimzer(self.config, self.forcefield, maxiter=8,
                                   neighbors_table=self.neighbors_table)
        self.config = minimizer.minimize()
        return minimizer.U_min

    def minimize_until(self, cutoff, verbose=True):
        while True:
            U_min = self.minimize(100)
            if verbose:
                print '%.4e' % U_min
            if U_min < cutoff:
                return U_min

    def get_config(self):
        config = self.config.copy()
        config.normalize_positions()
        return config


