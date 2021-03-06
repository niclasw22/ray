#ifndef RAY_OUT
#define RAY_OUT

#include <hdf5.h>
#include "par.h"
#include "varr.h"

void output_print_init(double *args, int n, char *filename);
void output_print_step(double t, double *x, int n, char *filename);
void output_h5_init(char *filename);
void output_par_h5(struct parList *pars, char *filename);
void output_map_h5(double *map, int *nhits, int Nrays, int buf_width,
                    char *filename);
void output_track_h5(struct varr *track, int id, char *filename);

void createFile(char *fname);
void createGroup(char *fname, char *gname);
void createDataset(char *fname, char *gname, char *dname, int dim,
                    hsize_t *fdims, hid_t type);
void writeSimple(char *file, char *group, char *dset, void *data, hid_t type);
void writePatch(char *file, char *group, char *dset, void *data, hid_t type,
                int dim, hsize_t *loc_start, hsize_t *glo_start, 
                hsize_t *read_size, hsize_t *loc_size, hsize_t *glo_size);

#endif
