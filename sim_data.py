"""This script generates a simulated data to test algorithm.
It will work the following way: 
1. loads a tod template
2. loads a random detector
3. adds a period signal in the time stream
4. chop up the time stream into a few different pieces
5. merge a random collection of pieces
6. save the pieces of time stream in the same format as a data file

Another way that it will work: load the real signal, inject a
simulation into it. This sounds easier to start, so I will start with
this approach.

"""

import argparse, os, os.path as op
import numpy as np
from scipy import stats 
import cutslib as cl
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--odir', default='out')
parser.add_argument('-v', action='store_true', default=False)
parser.add_argument('--data', nargs='+')
parser.add_argument('--fwhm', help='fwhm of gaussian pulse in s', type=float, default=0.01)
parser.add_argument('--freq', help='freq of signal in Hz', type=float, default=12)
parser.add_argument('--phi', help='initial phase in deg', type=float, default=0)
parser.add_argument('--ampfrac', help='amplitude of signal as a fraction of noise level',
                    type=float, default=1)
parser.add_argument('--amp', help='amplitude of signal in mK', type=float, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--simonly', action='store_true')
parser.add_argument('--oname', default='sim.npy')
args = parser.parse_args()

# get real data
data = np.hstack([np.load(f) for f in args.data])
idx = np.argsort(data[0])
data = data[:,idx]
if not op.exists(args.odir): os.makedirs(args.odir)

# beam profile
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

t = data[0]-data[0][0]
phis = phase(t, 2*np.pi*args.freq, args.phi/180*np.pi)
# D = fwhm / T = fwhm * freq
I_t = profile(phis, D=args.fwhm*args.freq)
# signal amplitude = fraction of nlev * nlev
if args.amp is None:
    nlev = 0.741*stats.iqr(data[1])
    amp = args.ampfrac * nlev
else: amp = args.amp
# debug
if args.debug:
    _t = np.linspace(0,1,400)
    _I_t = profile(phase(_t, 2*np.pi*args.freq, args.phi/180*np.pi), D=args.fwhm*args.freq)
    plt.plot(_t, amp*_I_t)
    plt.xlabel('t [s]')
    plt.ylabel('signal [uK]')    
    plt.savefig('debug_1.png')
    plt.close()
# add signal
sim_data = data.copy()
if args.simonly:
    sim_data[1] = amp * I_t
else:
    sim_data[1] += amp * I_t
if args.debug:
    plt.plot(sim_data[1], 'k.', alpha=0.1, label='amp*I_t+data', markersize=1)
    plt.plot(amp * I_t, 'r.', alpha=0.1, label='amp*I_t', markersize=1)
    plt.plot(data[1], 'g.', alpha=0.1, label='data', markersize=1)
    # plt.xlabel('t [s]')
    plt.ylabel('signal [uK]')
    plt.legend()
    plt.savefig('debug_2.png')
# save
ofile = op.join(args.odir, args.oname)
print("Writing:", ofile)
np.save(ofile, sim_data)
