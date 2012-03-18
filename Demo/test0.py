from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

from pyljfluid.components import (Parameters,
                                  NeighborsTable, NeighborsTableTracker,
                                  LJForceFeild, Config,
                                  System, EnergyMinimzer)



parameters = Parameters(
    mass=1.0,
    delta_t=1e-2)

forcefield = LJForceFeild(
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

    Tc = config.calculate_temperature(mass=parameters.mass)

    neighbors_table.rebuild_neighbors(config.positions, config.box_size)
    U = forcefield.evaluate_potential(config.positions, config.box_size, neighbors_table)

    print 'init T=%.2f U=%.2e' % (Tc, U / N_particles)

    # neighbors_table.rebuild_neighbors(config.positions, config.box_size)
    # print 'neighbors', neighbors_table.size
    # forces = np.zeros_like(config.positions)
    # forcefield.evaluate_forces(forces, config.positions, config.box_size, neighbors_table)
    # U = forcefield.evaluate_potential(config.positions, config.box_size, neighbors_table)

    minizer = EnergyMinimzer(config, forcefield, maxiter=40, neighbors_table=neighbors_table)
    config = minizer.minimize()
    print 'minimized U=%.2f' % (minizer.U_min / N_particles,)

    neighbors_table.rebuild_neighbors(config.positions, config.box_size)
    U = forcefield.evaluate_potential(config.positions, config.box_size, neighbors_table)
    print 'run T=%.2f U=%.2e' % (Tc, U / N_particles)

    neighbors_table_tracker = NeighborsTableTracker(neighbors_table, config.box_size)

    i_scale = 30

    acc = []

    forces = np.empty_like(config.positions)
    for j in xrange(100):

        if j < i_scale:
            config.randomize_velocities(T=T, mass=parameters.mass)

        for i in xrange(50):
            forcefield.evaluate_forces(forces, config.positions % config.box_size,
                                       config.box_size, neighbors_table)
            neighbors_table_tracker.maybe_rebuild_neighbor(config.positions)
            #forces.fill(0.0)
            config.propagate(forces, parameters.mass)

        KE = config.calculate_kinetic_energy()
        Tx = config.calculate_temperature(mass=parameters.mass)
        U = forcefield.evaluate_potential(config.positions % config.box_size,
                                          config.box_size, neighbors_table) / N_particles
        print '%d T=%.2f K=%.2f U=%.2f H=%.2f' % (
            j, Tx, KE, U, KE + U)
        acc.append([Tx, KE, U])

    Ts, Ks, Us = np.array(acc).T
    plt.clf()
    plt.axvline(i_scale, color='k')
    plt.plot(Ks, 'b-')
    plt.plot(Us, 'r-')
    plt.plot(Us + Ks, 'm-')
    plt.show()

def prof_main():
  import cProfile, pstats, sys
  prof = cProfile.Profile()
  prof = prof.runctx('main()', globals(), locals())
  stats = pstats.Stats(prof, stream=sys.stderr)
  stats.sort_stats('time')
  stats.print_stats(10)

__name__ == '__main__' and prof_main()

