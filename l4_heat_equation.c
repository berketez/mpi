#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N_GLOBAL  1000      /* total grid points                      */
#define N_STEPS   5000      /* number of time steps                   */
#define ALPHA     0.01      /* thermal diffusivity                    */
#define DX        0.001     /* spatial step                           */
#define DT        0.00004   /* time step (must satisfy CFL: dt < dx^2/(2*alpha)) */
#define T_LEFT    100.0     /* left boundary temperature              */
#define T_RIGHT   0.0       /* right boundary temperature             */
#define MASTER    0
#define TAG       99

/*
    Parallel 1D heat equation solver using finite differences.

    The domain is decomposed across MPI processes, each solving its
    local portion. Ghost cells are exchanged at subdomain boundaries
    using MPI_Sendrecv for efficient neighbor communication.

    PDE:  dT/dt = alpha * d²T/dx²
    Scheme: Forward Euler (explicit)
    CFL:   dt * alpha / dx² = 0.4 < 0.5 ✓
*/
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Domain decomposition */
    int local_n = N_GLOBAL / size;
    int start = rank * local_n;

    /* Allocate with ghost cells: [ghost_left | local_0 ... local_n-1 | ghost_right] */
    double *T     = calloc(local_n + 2, sizeof(double));
    double *T_new = calloc(local_n + 2, sizeof(double));

    /* Initial condition: T = 0 everywhere, boundaries fixed */
    for (int i = 0; i <= local_n + 1; i++)
        T[i] = 0.0;

    double r = ALPHA * DT / (DX * DX);   /* CFL number */

    if (rank == MASTER)
        printf("CFL number: %.4f (must be < 0.5)\n", r);

    int left  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Status status;

    /* Time integration */
    for (int step = 0; step < N_STEPS; step++) {

        /* Apply boundary conditions */
        if (rank == 0)
            T[0] = T_LEFT;         /* left ghost = Dirichlet BC */
        if (rank == size - 1)
            T[local_n + 1] = T_RIGHT;  /* right ghost = Dirichlet BC */

        /* Exchange ghost cells with neighbors using Sendrecv */
        MPI_Sendrecv(&T[1],       1, MPI_DOUBLE, left,  TAG,
                     &T[local_n + 1], 1, MPI_DOUBLE, right, TAG,
                     MPI_COMM_WORLD, &status);

        MPI_Sendrecv(&T[local_n], 1, MPI_DOUBLE, right, TAG,
                     &T[0],       1, MPI_DOUBLE, left,  TAG,
                     MPI_COMM_WORLD, &status);

        /* Update interior points: Forward Euler */
        for (int i = 1; i <= local_n; i++)
            T_new[i] = T[i] + r * (T[i-1] - 2.0*T[i] + T[i+1]);

        /* Swap pointers */
        double *tmp = T;
        T = T_new;
        T_new = tmp;
    }

    /* Gather results at master */
    double *T_global = NULL;
    if (rank == MASTER)
        T_global = malloc(N_GLOBAL * sizeof(double));

    MPI_Gather(&T[1], local_n, MPI_DOUBLE,
               T_global, local_n, MPI_DOUBLE,
               MASTER, MPI_COMM_WORLD);

    /* Print result */
    if (rank == MASTER) {
        printf("\nFinal temperature distribution (sampled every %d points):\n", N_GLOBAL / 20);
        printf("%-10s %-12s\n", "x", "T(x)");
        printf("──────────────────────\n");
        for (int i = 0; i < N_GLOBAL; i += N_GLOBAL / 20) {
            double x = i * DX;
            printf("%-10.4f %-12.4f\n", x, T_global[i]);
        }
        printf("\nExpected: linear profile T(x) = %.1f - %.1f*x/L (steady state)\n",
               T_LEFT, T_LEFT - T_RIGHT);
        free(T_global);
    }

    free(T);
    free(T_new);
    MPI_Finalize();
    return 0;
}
