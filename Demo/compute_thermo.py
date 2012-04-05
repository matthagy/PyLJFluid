'''Compute the thermodynamic properties - excess internal energy and
   the virial - of a Lennard-Jones fluid.
   This is accomplished by computing the static pair correlation
   function of particles with in the system. All of the thermodynamic
   properties can then be computed from g(r) due to the spherical
   symmetry of the isotropic potential.
'''

from pyljfluid.components import (LJForceField, Config, MDSimulator,
                                  StaticPairCorrelationCalculator,
                                  StaticPairCorrelationIntegrator)

N = 1000
rho = 0.75
T = 2.849
forcefield = LJForceField(sigma=1.0, epsilon=1.0, r_cutoff=2.5)
untruncated_forcefield = LJForceField(sigma=1.0, epsilon=1.0, r_cutoff=1e10)
r_neighbor_skin = 1.0 * forcefield.sigma
mass = 48 * forcefield.epsilon * forcefield.sigma ** -2
dt = 0.032
gr_dr = 0.01 * forcefield.sigma

def main():
    virial, U_interact = compute_thermodynamics(rho, T, verbose=True)
    print 'average: virial=%.3f U_i=%.3f' % (virial, U_interact)

def compute_thermodynamics(rho, T, verbose=True):
    beta = 1.0 / T
    config0 = Config.create(N=N, rho=rho, dt=dt, sigma=forcefield.sigma, T=T, mass=mass)
    sim = MDSimulator(config0, forcefield, mass=mass, r_skin=r_neighbor_skin)
    equilibrate_system(sim)
    gr = compute_gr(sim, rho, beta, verbose=True)
    virial, U_interact = calculate_thermodynamic_integrals(gr, rho, beta)
    return virial, U_interact

def equilibrate_system(sim, verbose=True):
    for i in xrange(50):
        sim.config.randomize_velocities(T=T, mass=mass)
        sim.cycle(50)
        if verbose:
            print 'equilibrate cycle i=%03d U=%.3f virial=%.3f' % (
                i, sim.compute_potential_energy(), sim.compute_virial())

def compute_gr(sim, rho, beta, verbose=True):
    gr_calc = StaticPairCorrelationCalculator(dr=gr_dr,
                                              r_max=sim.config.box_size / 2)
    for i in xrange(200):
        sim.cycle(20)
        gr_calc.accumulate_config(sim.config)
        if verbose:
            virial, U_interact = calculate_thermodynamic_integrals(
                                  gr_calc.get_accumulated(), rho, beta)
            print 'compute cycle i=%04d H=%.4f <virial>=%.3f <U_i>=%.3f' % (
                                i, sim.evaluate_hamiltonian(),
                                virial, U_interact)
    return gr_calc.get_accumulated()

def calculate_thermodynamic_integrals(gr, rho, beta, ):
    spci = StaticPairCorrelationIntegrator(gr, untruncated_forcefield, rho=rho, beta=beta)
    virial = spci.calculate_virial()
    U_interact = spci.calculate_excess_internal_energy()
    return virial, U_interact

__name__ == '__main__' and main()
