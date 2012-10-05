/* Optimized code for evaluating Lennard-Jones forces
 * for a configuration of particles within a cubic periodic
 * box.
 * Currently this code accounts for roughly 70% of simulation
 * runtime so optimization is worthwhile.
 */

#include <stdlib.h>
#include <stdint.h>
#include "opt.h"

typedef unsigned int uint;

#define dok3 for (int k=0; k<3; k++)

// Calculates net forces as due to LJ interactions on each particle within the
// N x 3 force array.
// Doesn't zero the force array, such that the calculated forces can be
// accumulated on top of the forces already calculated by another force field.
// Uses the neighbors table stored in the N_neighbor x 2 array of neighbor
// indices.
void
PyLJFluid_evaluate_LJ_forces(double *OPT_RESTRICT forces,
                             size_t N_neighbors,
                             uint *OPT_RESTRICT neighbors,
                             double *OPT_RESTRICT positions,
                             double box_size,
                             double sigma, double epsilon, double r_cutoff)
{
  double half_size = 0.5 * box_size;
  double r_cutoff_sqr = r_cutoff * r_cutoff;
  double sigma2 = sigma * sigma;
  double factor = -4 * 6 * epsilon;

  // loop over pairs of particles in neighbors tables
  for (uint neighbor_i=0; neighbor_i<N_neighbors; neighbor_i++) {
    uint inx_i3 = 3*neighbors[neighbor_i*2];
    uint inx_j3 = 3*neighbors[neighbor_i*2 + 1];

    // calculate separation vector with periodic boundary condition
    double r_ij[3];
    { double *p_i_base = positions + inx_i3;
      double *p_j_base = positions + inx_j3;
      dok3 {
        double l = p_j_base[k] - p_i_base[k];
        if (unlikely(l > half_size)) {
          l -= box_size;
        } else if (unlikely(l < -half_size)) {
          l += box_size;
        }
        r_ij[k] = l;
      }
    }
    double r_sqr = 0.0;
    dok3 { r_sqr += r_ij[k] * r_ij[k]; }
    if (r_sqr > r_cutoff_sqr) continue;

    // calculate the scalar force
    // scale it by 1/r such that it includes the normalization term to
    // make the separation vector a unit vector as needed to calculate
    // components of the force vector
    double scaled_f; {
      double inv_r_sqr = 1.0 / r_sqr;
      double x2 = sigma2 * inv_r_sqr;
      double x6 = x2*x2*x2;
      scaled_f = factor * inv_r_sqr * (2*x6*x6 - x6);
    }

    // calculate components of the force vector and add to net force
    // for each particle
    { double *f_i_base = forces + inx_i3;
      dok3 { f_i_base[k] += r_ij[k] * scaled_f; }
    }
    { double *f_j_base = forces + inx_j3;
      dok3 { f_j_base[k] -= r_ij[k] * scaled_f; }
    }
  }
}


