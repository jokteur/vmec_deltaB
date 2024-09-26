import os

os.chdir(os.path.dirname(__file__))
import h5py
import numpy as np

file = h5py.File("data/satire_kink.h5", "r")

xm = file["xm"][:]
xn = file["xn"][:]

bsubsmns = file["bsubsmns"][:].T
bsubumnc = file["bsubumnc"][:].T
bsubvmnc = file["bsubvmnc"][:].T

out = h5py.File("build/evaluate.h5", "w")
out["xm"] = xm
out["xn"] = xn
out["sin_coefficient"] = bsubsmns

theta = np.linspace(0, 2 * np.pi, 500)
phi = np.linspace(0, 2 * np.pi, 500)

out["u"] = theta
out["v"] = phi
out.attrs["has_sin"] = True
out.attrs["has_cos"] = False
