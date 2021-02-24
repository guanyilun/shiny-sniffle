"""This script loads the compiled per-tod reading and produce a plot
of all readings"""

import argparse, os, os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob

def sort_arr(idata):
    idx = np.argsort(idata[0])
    return idata[:,idx]

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--odir", default="plots")
parser.add_argument("--idir")
parser.add_argument("--oname", default="lightcurve.pdf")
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

# first load output files
# pa4 = glob.glob(op.join(args.idir, "*ar4*.npy"))
pa5 = glob.glob(op.join(args.idir, "*ar5*.npy"))
pa6 = glob.glob(op.join(args.idir, "*ar6*.npy"))


# load pa4
# data_pa4 = np.hstack([np.load(f) for f in pa4])
data_pa5 = sort_arr(np.hstack([np.load(f) for f in pa5]))
data_pa6 = sort_arr(np.hstack([np.load(f) for f in pa6]))

fig, axes = plt.subplots(2,1, sharex=True)
axes[0].plot(data_pa5[1], 'k.', markersize=1, alpha=0.1, label='PA5')
axes[1].plot(data_pa6[1], 'r.', markersize=1, alpha=0.1, label='PA6')
for ax in axes: ax.set_ylabel(r'$\mu$K')
plt.tight_layout()
ofile = op.join(args.odir, args.oname)
print("Writing:", ofile)
plt.savefig(ofile, bbox_inches='tight')
