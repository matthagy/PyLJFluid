from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

from pyljfluid.components import (Parameters,
                                  NeighborsTable, NeighborsTableTracker,
                                  LJForceField, Config,
                                  System, EnergyMinimzer,
                                  PairCorrelationFunctionCalculator)



parameters = Parameters(
    mass=1.0,
    delta_t=1e-2)

forcefield = LJForceField(
    sigma=1.0,
    epsilon=1.0,
    r_cutoff=3.3)

neighbors_table = NeighborsTable(
    r_forcefield_cutoff=forcefield.r_cutoff,
    r_skin=1.0)

N_particles = 1000
rho = 0.4
T = 1.35

def main():
    config = Config.create(N=N_particles, rho=rho, dt=parameters.delta_t,
                           sigma=forcefield.sigma, T=T, mass=parameters.mass)

    minizer = EnergyMinimzer(config, forcefield, maxiter=40, neighbors_table=neighbors_table)
    config = minizer.minimize()
    print 'minimized U=%.2f' % (minizer.U_min / N_particles,)

    neighbors_table_tracker = NeighborsTableTracker(neighbors_table, config.box_size)

    i_scale = 30

    pcc = PairCorrelationFunctionCalculator(0.01, config.box_size / 2)

    forces = np.empty_like(config.positions)
    for j in xrange(500):

        if j < i_scale:
            config.randomize_velocities(T=T, mass=parameters.mass)
        else:
            pcc.accumulate_config(config)

        for i in xrange(25):
            forcefield.evaluate_forces(forces, config.positions % config.box_size,
                                       config.box_size, neighbors_table)
            neighbors_table_tracker.maybe_rebuild_neighbor(config.positions)
            config.propagate(forces, parameters.mass)

        KE = config.calculate_kinetic_energy()
        Tx = config.calculate_temperature(mass=parameters.mass)
        U = forcefield.evaluate_potential(config.positions % config.box_size,
                                          config.box_size, neighbors_table) / N_particles
        print '%d T=%.2f K=%.2f U=%.2f H=%.2f' % ( j, Tx, KE, U, KE + U)

    plt.clf()
    plt.plot(pcc.calculate_rs(), pcc.calculate_gs())
    plt.show()



def prof_main():
  import cProfile, pstats, sys
  prof = cProfile.Profile()
  prof = prof.runctx('main()', globals(), locals())
  stats = pstats.Stats(prof, stream=sys.stderr)
  stats.sort_stats('time')
  stats.print_stats(10)

__name__ == '__main__' and prof_main()


