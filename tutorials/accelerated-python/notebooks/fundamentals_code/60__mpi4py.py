MPIRUN="mpirun --oversubscribe"

import os

if os.getenv("COLAB_RELEASE_TAG"): # If running in Google Colab
    !pip install mpi4py

from mpi4py import MPI

print("Hello World!")
