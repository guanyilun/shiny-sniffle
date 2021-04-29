"""This script attempts to search for pulsar in the compiled
lightcurve given a grid of frequency and phase to search from.

To begin with, we will not consider the boxcar function, and
simply do a brute-force search since we don't have a lot of
data anyway.

"""

import argparse, os, os.path as op
import numpy as np, glob
from scipy.stats import iqr
# we assumed a von Mises pulse profile
def profile(phi, D=1/40):
    kappa = np.log(2)/(2*np.sin(np.pi*D/2)**2)
    return np.exp(kappa*(np.cos(phi)-1))

def phase(t, omega_c, phi_c):
    """phase of the pulsar, omega_c is the angular frequency, phi_c is 
    the initial phase, so the phase is given by
      phi(t) = phi_c + omega_c t
    here we will assume a constant phase model.
    """
    return phi_c + omega_c * t

# utility
def parse_colon(expr):
    i, j, n = expr.split(':')
    return np.linspace(float(i),float(j),int(n))

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--odir', default='out')
parser.add_argument('--infiles', nargs='+')
parser.add_argument("--fwhm", type=float, default=0.01)
parser.add_argument("--freqs", help="freqs linspace in start:end:npoints in Hz", default="1:20:100")
parser.add_argument("--phis", help="phis linspace in start:end:npoints in deg", default="0:360:100")
parser.add_argument('-v', action='store_true', default=False)
parser.add_argument('--oname', default='stats.npy')

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
phis = parse_colon(args.phis) / 180 * np.pi
omegas = 2*np.pi*parse_colon(args.freqs)
# load compiled time series file
data = np.hstack([np.load(f) for f in args.infiles])
# sort by ctime
idx = np.argsort(data[0])
data = data[:,idx]
# find median sampling time
dt = np.median(np.diff(data[0]))
# find the variance
nlev = 0.741 * iqr(data[1]) * dt**0.5
# var = np.var(data[1])
var = nlev ** 2
print(f"nlev = {nlev}")
# build search space since I know the answer, I will start with a
# smaller space close to the right answer to debug
# start searching
t = data[0]-data[0][0]
from tqdm import tqdm
stats = []
for omega in tqdm(omegas):
    for phi in phis:
        # get expected time series
        freq = omega/(2*np.pi)
        I_t = profile(phase(t, omega, phi), D=args.fwhm*freq)
        I_t -= I_t.mean()
        # normalize it such that ivar*dt* sum_k(A I_t)**2 = 1
        # => A^2 = 1/(ivar*dt)/sum((I_t)**2)
        A = (np.sum(I_t**2)*dt/var)**-0.5
        # search statistics
        chisq = dt/var*np.sum(data[1]*A*I_t)
        stats.append([omega, phi, chisq])
ofile = op.join(args.odir, args.oname)
print("Writing:", ofile)
np.save(ofile, stats)
