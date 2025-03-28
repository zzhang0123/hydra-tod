
## Installation

```bash
virtualenv /envs/TOD

source /envs/TOD/bin/activate

pip install numpy scipy emcee jax matplotlib tqdm seaborn astropy astroquery healpy joblib mpmath mpi4py psutil git+https://github.com/telegraphic/pygdsm
```


Enable Hybrid Parallelism - This allows safe combination of:

- MPI for coarse-grained process parallelism
- Joblib for fine-grained thread-based parallelism within processes

Note that Joblib is only used for single-threaded code. 
Don't use it for multithreaded code to prevent hidden thread explosions in math libraries.
Also to avoid a conflict between MPI and Python's multiprocessing, try adding this environment variable:
```bash
export JOBLIB_START_METHOD="forkserver"
export OMP_NUM_THREADS= # Number of threads in each process

# Then run MPI
mpiexec -n 4 python your_script.py
```

The automatic initialization is disabled (via mpi4py.rc.initialize = False ) to Avoid Conflicts with Parallel Libraries. To enable it, only manually initialize it explicitly once in your main script (the one executed with mpirun ):
```python
# All module imports first (below is an example)
import mpiutil
import flicker_model

# Then initialize MPI explicitly
from mpi4py import MPI
MPI.Init()

# ... rest of your code ...

MPI.Finalize()
```
