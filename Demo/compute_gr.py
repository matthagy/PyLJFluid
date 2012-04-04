'''Simulate a Lennard-Jones fluid and plot the computed g(r)
'''

from pyljfluid.components import (LJForceField, Config, MDSimulator,
                                  StaticPairCorrelationCalculator)
import matplotlib.pyplot as plt

# Parameters
N = 1000
rho = 0.2
T = 1.5
forcefield = LJForceField(sigma=1.0, epsilon=1.0, r_cutoff=2.5)
r_neighbor_skin = 1.0 * forcefield.sigma
mass = 48 * forcefield.epsilon * forcefield.sigma ** -2
dt = 0.032

# Initialize system
config0 = Config.create(N=N, rho=rho, dt=dt, sigma=forcefield.sigma, T=T, mass=mass)
sim = MDSimulator(config0, forcefield, mass=mass, r_skin=r_neighbor_skin)

# Equilibrate
for i in xrange(200):
    sim.config.randomize_velocities(T=T, mass=mass)
    sim.cycle(50)
    print 'equilibrate cycle i=%03d U=%.3f' % (i, sim.evaluate_potential())

# Compute g(r)
gr_calc = StaticPairCorrelationCalculator(dr=0.01 * forcefield.sigma,
                                          r_max=config0.box_size / 2)
for i in xrange(1000):
    print 'compute cycle i=%04d H=%.4f ' % (i, sim.evaluate_hamiltonian())
    sim.cycle(25)
    gr_calc.accumulate_config(sim.config)
gr = gr_calc.get_accumulated()

# Plot g(r)
plt.clf()
plt.plot(gr.r, gr.g)
plt.xlabel('Pair Separation, $r$ ($\sigma$)')
plt.ylabel('Reduced Pair Density, $g(r)$')
plt.show()
