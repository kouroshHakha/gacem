from mpi4py import MPI


def gather(local_x, global_x, root=0):
    MPI.COMM_WORLD.Gather(local_x, global_x, root)