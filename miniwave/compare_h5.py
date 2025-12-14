import sys, h5py, numpy as np
if len(sys.argv) < 3:
    print("Uso: python compare_h5.py ref.h5 other.h5")
    sys.exit(1)
a = h5py.File(sys.argv[1],'r')["vector"][:].ravel()
b = h5py.File(sys.argv[2],'r')["vector"][:].ravel()
diff = a - b
print("Linf:", np.max(np.abs(diff)))
print("L2:", np.linalg.norm(diff))
