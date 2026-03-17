import numpy
import scipy
import numba
import sys
import os

print(f"Python: {sys.version}")
print(f"Numpy: {numpy.__version__} @ {os.path.dirname(numpy.__file__)}")
print(f"Scipy: {scipy.__version__} @ {os.path.dirname(scipy.__file__)}")
print(f"Numba: {numba.__version__} @ {os.path.dirname(numba.__file__)}")
