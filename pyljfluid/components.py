
from __future__ import division

from functools import wraps

import numpy as np
try:
    from scipy.optimize import fmin_cg
except ImportError:
    def fmin_cg(*args, **kwds):
        raise RuntimeError('fmin_cg not available; install scipy to use this functionality')

from base_components import (NeighborsTable, ForceField, LJForceField, BaseConfig,
                             BasePairCorrelationFunctionCalculator,
                             BaseMeanSquareDisplacementCalculator,
                             BaseVelocityAutocorrelationCalculator)
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


def cached_property(name_or_func, cache_name=None):
    if isinstance(name_or_func, basestring):
        return lambda func: cached_property(func, cache_name=name_or_func)

    func = name_or_func
    assert callable(func)

    if cache_name is None:
        cache_name = '_' + func.func_name
    assert isinstance(cache_name, basestring)

    @property
    @wraps(func)
    def wrapper(self):
        try:
            return getattr(self, cache_name)
        except AttributeError:
            value = func(self)
            setattr(self, cache_name, value)
            return value
    return wrapper


class PairCorrelationData(object):

    def __init__(self, bins, r_min, r_prec):
        self.bins = bins
        self.r_min = r_min
        self.r_prec = r_prec

    @cached_property
    def r(self):
        return self.r_min + self.r_prec * np.arange(self.bins.size)

    @cached_property
    def g(self):
        N = self.bins.sum()
        if not N:
            return None

        r_max = self.bins.size * self.r_prec
        V = 4.0 / 3.0 * np.pi * r_max**3
        rho = N / V
        v = 4.0 / 3.0 * np.pi * ((self.r + self.r_prec)**3 - self.r**3)
        rhos =  self.bins / v
        return rhos / rho

    @cached_property
    def h(self):
        return self.g - 1.0


class PairCorrelationFunctionCalculator(BasePairCorrelationFunctionCalculator):

    def accumulate_config(self, config):
        self.accumulate_positions(config.positions, config.box_size)

    def get_data(self):
        return PairCorrelationData(self.bins.copy(), self.r_min, self.r_prec)


class PairCorrelationIntegrator(object):

    def __init__(self, pc_data, forcefield, rho=None, beta=None):
        self.pc_data = pc_data
        self.forcefield = forcefield
        self.rho = rho
        self.beta = beta

    def integrate_g_product_over_space_ex(self, func, where):
        r = self.pc_data.r[where]
        g = self.pc_data.g[where]
        return np.trapz(4*np.pi*r**2 * g * func(r), r)

    @cached_property
    def where_sparse(self):
        return self.pc_data.g != 0.0

    def integrate_g_product_over_space(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_sparse)

    @cached_property
    def where_sparse_and_in_cutoff(self):
        return (self.pc_data.g != 0.0) & (self.pc_data.r <= self.forcefield.r_cutoff)

    def integrate_g_product_over_space_in_cutoff(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_sparse_and_in_cutoff)

    @cached_property
    def excess_internal_energy(self):
        return self.integrate_g_product_over_space_in_cutoff(self.forcefield.evaluate_potential_function)

    @cached_property
    def excess_virial_sampled(self):
        return -self.integrate_g_product_over_space_in_cutoff(
            lambda r: r*self.forcefield.evaluate_scalar_force_function(r))


class LJPairCorrelationIntegrator(PairCorrelationIntegrator):

    def virial_correction(self, r_v):
        '''Contribution to virial in the range [r_v:oo] assuming
           g(r) == 1 in this range
        '''
        return -self.forcefield.epsilon**2*(
            208*np.pi*r_v**6*self.forcefield.sigma**7 -
            224*np.pi*self.forcefield.sigma**13/3)/(91*r_v**9)

    @cached_property
    def excess_virial_correction(self):
        return self.virial_correction(self.forcefield.r_cutoff)

    @cached_property
    def virial_reduction_factor(self):
        return self.beta * self.rho / 6.0

    @cached_property
    def reduced_virial(self):
        return 1.0 + self.virial_reduction_factor * (
            self.excess_virial_sampled + self.excess_virial_correction)

class WindowAnalyzeBase(object):

    def __init__(self, window_size, N_particles, analyze_rate=1, *args, **kwds):
        super(WindowAnalyzeBase, self).__init__(window_size, N_particles, *args, **kwds)
        self.analyze_rate = analyze_rate

    def compute_time(self):
        return self.analyze_rate * np.arange(self.window_size)

class MeanSquareDisplacementCalculator(WindowAnalyzeBase, BaseMeanSquareDisplacementCalculator):

    def analyze_config(self, config):
        self.analyze_positions(config.positions, config.box_size)

    def compute_msd(self):
        n_acc = self.calculate_n_accumulates()
        if not n_acc:
            return None

        return self.acc_msd_data / float(n_acc * self.N_particles)

    def __reduce__(self):
        return (create_msdc,
                (self.window_size, self.N_particles,
                 self.analyze_rate,
                 self.n_positions_seen,
                 self.displacement_window, self.last_positions,
                 self.acc_msd_data))

def create_msdc(window_size, N_particles, analyze_rate, n_positions_seen,
                displacement_window, last_positions, acc_msd_data):
    return MeanSquareDisplacementCalculator(window_size, N_particles,
                                            analyze_rate=analyze_rate,
                                            n_positions_seen=n_positions_seen,
                                            displacement_window=displacement_window,
                                            last_positions=last_positions,
                                            acc_msd_data=acc_msd_data)


class VelocityAutocorrelationCalculator(WindowAnalyzeBase, BaseVelocityAutocorrelationCalculator):

    def analyze_config(self, config):
        self.analyze_velocities(config.calculate_velocities())

    def compute_vacf(self):
        n_acc = self.calculate_n_accumulates()
        if not n_acc:
            return None

        return self.acc_correlations / float(n_acc * self.N_particles)

    def __reduce__(self):
        return (create_vacfc,
                (self.window_size, self.N_particles,
                 self.analyze_rate,
                 self.n_velocities_seen,
                 self.velocities_windows, self.acc_correlations))

def create_vacfc(window_size, N_particles, analyze_rate, n_velocities_seen,
                 velocities_windows, acc_correlations):
    return VelocityAutocorrelationCalculator(window_size, N_particles,
                                             analyze_rate=analyze_rate,
                                             n_velocities_seen=n_velocities_seen,
                                             velocities_windows=velocities_windows,
                                             acc_correlations=acc_correlations)



