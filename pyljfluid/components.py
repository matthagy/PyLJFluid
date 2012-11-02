#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

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

# TODO
# Separate simulation and analysis code into separate modules.
# Further separate analysis code into static vs. dynamic analysis.

__all__ = ['NeighborsTable', 'ForceField', 'LJForceField',
           'Config', 'NeighborsTableTracker', 'EnergyMinimzer',
           'MDSimulator',
           'StaticPairCorrelation',
           'StaticPairCorrelationCalculator',
           'StaticPairCorrelationIntegrator']

def create_random_state(op=None):
    if isinstance(op, np.random.RandomState):
        return op
    return np.random.RandomState(op)

def create_velocities(N, T=1.0, mass=1.0, random_state=None):
    return create_random_state(random_state).normal(scale=np.sqrt(T / mass), size=(N, 3))

def calculate_box_size(N, sigma, rho):
    '''Calculate the lateral size for a cubic box of N
       particles, each of diameter sigma, such that the particle
       density is rho = N*sigma**3/V
    '''
    V = N * sigma**3 / rho
    return V**(1/3)

def create_hcp_positions(l):
    '''Create a hexagonal close-packed (hcp) lattice of sphere positions that are
       contained within a cubic box of lateral size l (each sphere has unit diameter).
    '''
    r = 0.5
    row_height_shift = r * np.sqrt(3)
    plane_height_shift = np.sqrt(6) * r * 2.0 / 3.0
    n_row = int(np.floor(l))

    row0 = np.array([np.arange(n_row), np.zeros(n_row), np.zeros(n_row)]).T
    plane0 = np.array([row0 + np.array([r if i%2==1 else 0, i*row_height_shift, 0])
                       for i in xrange(int(np.floor(l / row_height_shift)))])
    planes = np.array([plane0 + np.array([r if i%2==1 else 0,
                                         np.sqrt(3)/3*r if i%2==1 else 0,
                                          i * plane_height_shift])
                       for i in xrange(int(np.floor(l / plane_height_shift)))])

    sites = planes.reshape((np.prod(planes.shape[:3:]), 3))
    return sites

class Config(BaseConfig):
    '''Represents the current static and dynamic state of isotropic
       particle system within a cubic periodic box.
    '''

    @classmethod
    def create(cls, N, rho, dt, sigma=1.0, T=1.0, mass=1.0, random_state=None):
        '''Create a new random config with N particle at density rho.
        '''
        random_state = create_random_state(random_state)
        box_size = calculate_box_size(N, sigma, rho)
        hcp_positions = sigma * create_hcp_positions(box_size / sigma)
        if len(hcp_positions) < N:
            raise ValueError("config cannot be initialized from hcp lattice")
        hcp_indices = np.arange(len(hcp_positions))
        random_state.shuffle(hcp_indices)
        positions = hcp_positions[hcp_indices[:N:]]
        inst = cls(positions, last_positions=None, box_size=box_size, dt=dt, sigma=sigma)
        inst.randomize_velocities(T=T, mass=mass, random_state=random_state)
        return inst

    def copy(self):
        return self.__class__(positions=self.positions.copy(),
                              last_positions=self.last_positions.copy(),
                              box_size=self.box_size,
                              dt=self.dt,
                              sigma=self.sigma)

    def with_new_positions(self, new_positions):
        cp = self.__class__(positions=new_positions, last_positions=None,
                            box_size=self.box_size, dt=self.dt, sigma=self.sigma)
        cp.set_velocities(self.calculate_velocities())
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

    def calculate_kinetic_energy(self, mass):
        v_rms = self.calculate_rms_velocity()
        return 0.5 * mass * v_rms**2

    def calculate_temperature(self, mass):
        KE = self.calculate_kinetic_energy(mass)
        return 2.0 / 3.0 * KE

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
        return (Config, (self.positions, self.last_positions, self.box_size, self.dt, self.sigma))


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

    kB = 1.0

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

    def compute_potential_energy(self):
        '''Compute the internal potential energy due to the force field
           interactions. Result are inversely scaled by the number of particles.
        '''
        return self.forcefield.evaluate_potential(self.config.positions % self.config.box_size,
                                                  self.config.box_size, self.neighbors_table)

    def compute_kinetic_energy(self):
        '''Compute the kinetic energy due to translational veloctiy of particles.
           Result are inversely scaled by the number of particles.
        '''
        return self.config.calculate_kinetic_energy(self.mass)

    def compute_energy(self):
        '''Compute the total energy (sum kinetic and potential) of the system.
           Result are inversely scaled by the number of particles.
        '''
        return self.compute_potential_energy() + self.compute_kinetic_energy()

    def compute_temperature(self):
        return self.config.calculate_temperature(self.mass)

    def compute_excess_pressure(self, correct_long_range):
        '''Compute the pressure due to particle pair interactions.
           Result are inversely scaled by the number of particles.
        '''
        v = 1.0 / 3.0 * self.forcefield.evaluate_virial_sum(
            self.config.positions % self.config.box_size,
            self.config.box_size, self.neighbors_table)
        if correct_long_range:
            v += self.forcefield.long_range_virial_correction()
        return (self.kB / self.config.V) * v

    def compute_ideal_pressure(self):
        '''Compute pressure due to kinetic energy alone.
           Result are inversely scaled by the number of particles.
        '''
        return self.kB * self.compute_temperature() / self.config.V

    def compute_total_pressure(self, correct_long_range=True):
        '''Compute the total (sum kinetic and potential) pressure.
           Result are inversely scaled by the number of particles.
        '''
        return self.compute_excess_pressure(correct_long_range) + self.compute_ideal_pressure()

    def compute_virial(self, correct_long_range=True):
        P = self.compute_total_pressure(correct_long_range)
        return P * self.config.V / (self.kB * self.compute_temperature())

    def compute_excess_virial(self, correct_long_range):
        v = ((3 * self.kB * self.compute_temperature())**-1 *
             self.forcefield.evaluate_virial_sum(self.config.positions % self.config.box_size,
                                                 self.config.box_size, self.neighbors_table))
        if correct_long_range:
            v += self.forcefield.long_range_virial_correction()
        return v

    # Old (deprecated) methods
    def evaluate_potential(self):
        return self.compute_potential_energy()

    def evaluate_hamiltonian(self):
        return self.compute_energy()

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
    '''Wrapper to create a cached readonly property for a class.
    '''
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


class StaticPairCorrelation(object):
    '''Describes the static correlation of isotropic particle pairs.
          e.g. The numerical analogs of the standard g(r) and h(r) functions.

       The underlying data is a histogram of pair separation distances at
       fixed spacing dr (i.e. the i-th bin contains the total number of
       pair observed at distances of i*dr <= r < (i+1)*dr.

       Additionally, the separation distances can be shifted by an r_offset.
       This is useful in avoiding the avoid the leading zeros that correspond
       to unphysically close pair distances.
    '''

    def __init__(self, pair_distance_histogram, dr, r_offset=0.0):
        pair_distance_histogram = np.asarray(pair_distance_histogram)
        assert pair_distance_histogram.ndim == 1
        assert pair_distance_histogram.size > 0
        self.pair_distance_histogram = pair_distance_histogram
        self.dr = dr
        self.r_offset = r_offset

    @cached_property
    def r_lower(self):
        '''Radial distance corresponding to correlation distances at
           the lower bound of each bin.
        '''
        return self.r_offset + self.dr * np.arange(self.pair_distance_histogram.size)

    @cached_property
    def r_mid(self):
        '''Radial distance corresponding to correlation distances at
           center of each bin.
        '''
        return self.r_lower + 0.5 * self.dr

    @property
    def r(self):
        return self.r_mid

    @cached_property
    def g(self):
        '''Reducued density pair correlations
        '''
        N = self.pair_distance_histogram.sum()
        if not N:
            return None

        r_max = self.pair_distance_histogram.size * self.dr
        V = 4.0 / 3.0 * np.pi * r_max**3
        rho = N / V
        v = 4.0 / 3.0 * np.pi * ((self.r_lower + self.dr)**3 - self.r_lower**3)
        rhos =  self.pair_distance_histogram / v
        return rhos / rho

    @cached_property
    def h(self):
        '''Shifted reducued density pair correlations
        '''
        if self.g is None:
            return None
        return self.g - 1.0


class StaticPairCorrelationCalculator(BasePairCorrelationFunctionCalculator):
    '''Calculate the PairCorrelationData from configurations (Config) of
       particles. The calculation is performed by accumulating a histogram
       of pair separation distances one configuration at a time. The
       intermediate state of calculation can be saved by pickeling the
       calculator object.
    '''

    def accumulate_config(self, config):
        self.accumulate_positions(config.positions, config.box_size)

    def get_accumulated(self):
        return StaticPairCorrelation(self.bins.copy(), self.dr, self.r_min)


class StaticPairCorrelationIntegrator(object):
    '''Calculate thermodynamic properties of an isotropic particle system
       by integrating over the sampled pair correlation (StaticPairCorrelation object)
       for the system. Uses the potential and gradient of the force field associated
       with the system.
    '''

    def __init__(self, pair_correlation, forcefield, rho, beta):
        self.pair_correlation = pair_correlation
        self.forcefield = forcefield
        self.rho = rho
        self.beta = beta

    def integrate_g_product_over_space_ex(self, func, mask=Ellipsis):
        '''Numerically integrate
            \int g(r) * r**2 * func(r)

           The mask argument allows the specification of which
           elements of data arrays to include (i.e. allows the
           exclusion of zero elements)
        '''
        r = self.pair_correlation.r[mask]
        g = self.pair_correlation.g[mask]
        return np.trapz(r**2 * g * func(r), r)

    @cached_property
    def where_g_nonzero(self):
        return self.pair_correlation.g != 0.0

    @cached_property
    def where_g_nonzero_and_in_cutoff(self):
        return (self.pair_correlation.g != 0.0) & (self.pair_correlation.r <= self.forcefield.r_cutoff)

    def integrate_g_product_over_space(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_g_nonzero)

    def integrate_g_product_over_space_in_cutoff(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_g_nonzero_and_in_cutoff)

    def calculate_excess_internal_energy(self):
        return (2.0 * np.pi * self.rho *
                self.integrate_g_product_over_space_in_cutoff(self.forcefield.evaluate_potential_function))

    def calculate_virial(self, correct_long_range=True):
        v = 1.0 - 2.0 / 3.0 * np.pi * self.beta * self.rho * (
            self.integrate_g_product_over_space_in_cutoff(
            lambda r: r * -self.forcefield.evaluate_scalar_force_function(r)))
        if correct_long_range:
            v += self.forcefield.long_range_virial_correction(self.pair_correlation.r.max())
        return v


class BaseTimeCorelationCalculator(object):
    '''Base class for Time Correlation Function (TCF) computations
    '''

    def __init__(self, window_size, N_particles, *args, **kwds):
        assert 'analyze_rate' in kwds
        self.analyze_rate = kwds.pop('analyze_rate')
        super(BaseTimeCorelationCalculator, self).__init__(window_size, N_particles, *args, **kwds)

    @classmethod
    def create(cls, window_size, N_particles, analyze_rate=1):
        # ensure analyze_rate is passed as a keyword
        return cls(window_size, N_particles, analyze_rate=analyze_rate)

    def compute_time(self):
        return self.analyze_rate * np.arange(self.window_size)


class MeanSquareDisplacementCalculator(BaseTimeCorelationCalculator, BaseMeanSquareDisplacementCalculator):
    '''Compute the mean square displacment TCF; i.e. th self-positional TCF
    '''

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


class VelocityAutocorrelationCalculator(BaseTimeCorelationCalculator, BaseVelocityAutocorrelationCalculator):
    '''Compute the velocity autocorrelation TCF (VACF); i.e. the self-velocity TCF
    '''

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



