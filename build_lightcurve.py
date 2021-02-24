"""In this script I will load all tods and try to build a times series
of point source"""

import argparse, os, os.path as op, sys
import numpy as np
import cutslib as cl
import moby2
from scipy.signal import savgol_filter
from scipy import stats

# for benchmarking
from enlib import bench

glitchp = {'nSig': 15, 'tGlitch' : 0.002, 'minSeparation': 30,
           'maxGlitch': 50000, 'highPassFc': 10.0, 'buffer': 5}

parser = argparse.ArgumentParser()
parser.add_argument("-o","--odir", default='out')
parser.add_argument("--tod", help="name of tod of interests")
parser.add_argument("--oname", default="ts.npy")
parser.add_argument("--srcmask-depot", default="/scratch/gpfs/yilung/gc_data/tmasks")
parser.add_argument("--srcmask-tag", default="s19_f090_gal_srcmask_mouse")
parser.add_argument("--window", type=int, default=800)
parser.add_argument("--poly", type=int, default=3)
parser.add_argument("-v","--verbose", action='store_true', default=False)
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

# first load metadata only
tod = cl.load_tod(args.tod, rd=False, autoloads=[], verbose=False)
# load srcmask
depot = cl.Depot(args.srcmask_depot)
srcmask = depot.read_object(cl.TODCuts, tod=tod, tag=args.srcmask_tag)
if np.sum([len(cv) for cv in srcmask.cuts])==0: sys.exit(0)

# now actually load the tod
with bench.show("load tod"):
    tod = cl.load_tod(args.tod, release='20201005', fs=False,
                      autoloads=['cuts','cal','abscal'], verbose=args.verbose)

# get detector masks, time masks
dets = tod.cuts.get_uncut()
cutmask = np.vstack([c.get_mask() for c in srcmask.cuts])
# get the time ranges of relevance
tmask = np.logical_or.reduce(cutmask[dets])
cv_tmask = cl.CutsVector.from_mask(tmask)
if np.sum(tmask) == 0: sys.exit(0)

# pre-process tod
with bench.show("preprocess tod"):
    cl.quick_transform(tod, steps=['ff_mce','ff_glitch','cal','abscal'],
                       glitchp=glitchp, verbose=args.verbose)
    
# to be more efficient, we will focus only on these ranges, but we need
# to expand to the window length first in order for the filtering to work
cv_tmask_expand = cv_tmask.get_buffered(args.window)
data = tod.data[np.ix_(dets,cv_tmask_expand.get_mask())] 
# filter data with polynomials in a given window length
if args.window % 2 == 0: window = args.window + 1
else: window = args.window
with bench.show("savgol filter"):
    filt_data = savgol_filter(data, window_length=window, polyorder=args.poly)
residue = data - filt_data
del data, filt_data
# we have overdone the job due to the buffer added, now reduce to the original
# set of samples
ind = np.zeros(len(tod.ctime)).astype(int)
ind[cv_tmask_expand.get_mask()] = np.arange(residue.shape[-1]).astype(int)
ind_orig = ind[cv_tmask.get_mask()]
residue = residue[:,ind_orig]
del ind, ind_orig
# now we are ready to extract the readings associated with the input
# srcmask, since we may have multiple readings at the same time, we perform
# a inverse-variance weighted average of all detector reading at each tsamp
rms = 0.741*stats.iqr(residue, axis=-1)
w = rms ** -2
# store readings in vals and corresponding tsamp in tsamps
tsamps = tod.ctime[tmask]
vals = np.zeros_like(tsamps)
cutmask_sel = cutmask[np.ix_(dets, cv_tmask.get_mask())]
for i in range(residue.shape[-1]):
    dmask = cutmask_sel[:,i]
    vals[i] = np.sum(w[dmask]*residue[dmask,i])/np.sum(w[dmask])
# store output
ofile = op.join(args.odir, f"{tod.info.name}.npy")
print("Writing:", ofile)
np.save(ofile, [tsamps, vals])
