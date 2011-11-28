

#include <stdlib.h>
#include <stdint.h>
#include "opt.h"

typedef unsigned int uint;

#define dok3 for (int k=0; k<3; k++)

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

  for (uint i=0; i<N_neighbors; i++) {
    uint inx_i3 = 3*neighbors[i*2];
    uint inx_j3 = 3*neighbors[i*2 + 1];

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

    double scaled_f; {
      double inv_r_sqr = 1.0 / r_sqr;
      double x2 = sigma2 * inv_r_sqr;
      double x6 = x2*x2*x2;
      scaled_f = factor * inv_r_sqr * (2*x6*x6 - x6);
    }
    { double *f_i_base = forces + inx_i3;
      dok3 { f_i_base[k] += r_ij[k] * scaled_f; }
    }
    { double *f_j_base = forces + inx_j3;
      dok3 { f_j_base[k] -= r_ij[k] * scaled_f; }
    }
  }
}


