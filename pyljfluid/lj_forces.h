
#ifndef _LJ_FORCES_H_
#define _LJ_FORCES_H_

void
PyLJFluid_evaluate_LJ_forces(double *forces,
                             size_t N_neighbors,
                             unsigned int* neighbors,
                             double *positions,
                             double box_size,
                             double sigma, double epsilon, double r_cutoff);

#endif /* _LJ_FORCES_H_ */
