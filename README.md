# MPI Training — ITU Supercomputer Club

Training materials for the **OpenMPI workshop** organized by the [ITU Supercomputer Club](https://github.com/ITU-Supercomputer-Club), Turkey's first university-level supercomputing club.

These examples were developed as hands-on exercises for club members learning parallel programming fundamentals on ITU's HPC infrastructure.

## Lessons

### Lesson 1 — Basics (`l1_basics.c`)

MPI initialization, rank/size queries, and conditional execution by rank.

```
$ mpirun -np 4 ./l1_basics
Hello HPC! from process 0 out of 4
Hello HPC! from process 2 out of 4
```

**Concepts:** `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Finalize`, SPMD model

### Lesson 2a — Ring Communication (`l2_ring.c`)

Processes form a ring topology. Each process receives a cumulative sum from its left neighbor, adds its own rank, and forwards the result to the right.

```
Process 0 ──→ Process 1 ──→ Process 2 ──→ Process 3
    ↑                                         │
    └─────────────────────────────────────────┘
```

```
$ mpirun -np 4 ./l2_ring
Received sum: 0, Current sum: 1, my rank: 1, rank of left: 0, rank of right: 2
Received sum: 1, Current sum: 3, my rank: 2, rank of left: 1, rank of right: 3
Received sum: 3, Current sum: 6, my rank: 3, rank of left: 2, rank of right: 0
Total rank sum is: 6
```

**Concepts:** `MPI_Send`, `MPI_Recv`, deadlock avoidance (master sends first), ring topology

### Lesson 2b — Work Partitioning (`l2_sharing_partition.c`)

Parallel summation of numbers 1–100 across 4 processes. Each process computes a local sum over its partition, then sends results to the master for aggregation.

```
Process 0:  1–25  → local_sum = 325
Process 1: 26–50  → local_sum = 950
Process 2: 51–75  → local_sum = 1575
Process 3: 76–100 → local_sum = 2200
─────────────────────────────────────
Master total:                   5050
```

**Concepts:** Domain decomposition, master-worker pattern, `MPI_Send`/`MPI_Recv` gather

### Lesson 3 — Collective Operations (`l3_collectives.c`)

Monte Carlo estimation of Pi using `MPI_Bcast`, `MPI_Reduce`, and `MPI_Allreduce`. Each process generates random samples independently, then collective operations aggregate the results.

```
$ mpirun -np 8 ./l3_collectives
Processes: 8, Samples: 1000000
Pi estimate: 3.14182400 (error: 2.32e-04)
  [rank 0] Pi via Allreduce: 3.141824
  [rank 1] Pi via Allreduce: 3.141824
  ...
```

**Concepts:** `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`, `MPI_SUM`, Monte Carlo methods, reproducible seeding

### Lesson 4 — Parallel Heat Equation Solver (`l4_heat_equation.c`)

1D heat equation (dT/dt = alpha * d²T/dx²) solved with explicit finite differences across distributed processes. Each process owns a subdomain and exchanges boundary data via ghost cells.

```
Domain decomposition (4 processes):

|  Proc 0  |  Proc 1  |  Proc 2  |  Proc 3  |
|←─ghost─→||←─ghost─→||←─ghost─→||←─ghost─→|
T=100                                      T=0
(Dirichlet)                          (Dirichlet)
```

```
$ mpirun -np 4 ./l4_heat
CFL number: 0.4000 (must be < 0.5)

Final temperature distribution:
x          T(x)
──────────────────────
0.0000     100.0000
0.0500     95.0000
0.1000     90.0000
...
0.9500     5.0000
```

**Concepts:** Domain decomposition, ghost cell exchange, `MPI_Sendrecv`, `MPI_Gather`, `MPI_PROC_NULL`, CFL stability condition, Forward Euler scheme

## Build & Run

```bash
mpicc -o l1_basics l1_basics.c
mpicc -o l2_ring l2_ring.c
mpicc -o l2_sharing l2_sharing_partition.c
mpicc -o l3_collectives l3_collectives.c -lm
mpicc -o l4_heat l4_heat_equation.c -lm

mpirun -np 4 ./l1_basics
mpirun -np 4 ./l2_ring
mpirun -np 4 ./l2_sharing
mpirun -np 8 ./l3_collectives
mpirun -np 4 ./l4_heat
```

## Requirements

- OpenMPI or MPICH
- GCC with MPI headers (`mpi.h`)
- `-lm` for math library (lessons 3–4)

## Context

These materials were created for the OpenMPI training series at **ITU Supercomputer Club** (2022–2023). The club was founded to provide hands-on HPC education to university students, covering parallel computing, Linux systems, and cluster administration.
