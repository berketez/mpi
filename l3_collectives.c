#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define MASTER 0
#define N 1000000

/*
    Monte Carlo estimation of pi using MPI collective operations.
    Each process generates random points, counts hits inside the unit circle,
    then MPI_Reduce aggregates all counts at the master.
*/
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Broadcast the number of samples per process */
    int samples_per_proc = N / size;
    MPI_Bcast(&samples_per_proc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* Each process generates random points */
    unsigned int seed = rank * 12345 + 67890;
    int local_hits = 0;

    for (int i = 0; i < samples_per_proc; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0)
            local_hits++;
    }

    /* Reduce all local counts to master */
    int total_hits = 0;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        double pi_estimate = 4.0 * total_hits / (samples_per_proc * size);
        double error = fabs(pi_estimate - M_PI);
        printf("Processes: %d, Samples: %d\n", size, samples_per_proc * size);
        printf("Pi estimate: %.8f (error: %.2e)\n", pi_estimate, error);
    }

    /* Demonstrate Allreduce — every process gets the result */
    int global_hits = 0;
    MPI_Allreduce(&local_hits, &global_hits, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double pi_all = 4.0 * global_hits / (samples_per_proc * size);
    printf("  [rank %d] Pi via Allreduce: %.6f\n", rank, pi_all);

    MPI_Finalize();
    return 0;
}
