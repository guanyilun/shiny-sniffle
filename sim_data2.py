# -*- tab-width: 4 -*-

"""This aims to insert pulsar to tods"""

import numpy as np, os, os.path as op
from enlib import errors, utils as u, config
from enact import filedb, actscan
import lib
from matplotlib import pyplot as plt

config.set("downsample", 1, "Amount to downsample tod by")
config.set("gapfill", "linear", "Gapfiller to use. Can be 'linear' or 'joneig'")
config.default("pmat_interpol_pad", 10.0, "Number of arcminutes to pad the interpolation coordinate system by")

parser = config.ArgumentParser()
parser.add_argument("catalog")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-s", "--srcs",      type=str,   default=None)
parser.add_argument("-v", "--verbose",   action="count", default=0)
parser.add_argument("-q", "--quiet",     action="count", default=0)
parser.add_argument(      "--minamp",    type=float, default=None)
parser.add_argument(      "--sys",       type=str,   default="cel")
args = parser.parse_args()

def read_srcs(fname):
    ra, dec, amp, T, phi = np.loadtxt(fname, unpack=True)
    # convert period[ms] to omega_c: T = 2*pi/omega_c
    omg = 2*np.pi/(T*1e-3)
    return np.array([ra*u.degree, dec*u.degree, amp, omg, phi])

filedb.init()
db       = filedb.scans.select(args.sel)
ids      = db.ids
sys      = args.sys
dtype    = np.float32
verbose  = args.verbose - args.quiet
down     = config.get("downsample")
poly_pad = 3*u.degree
bounds   = db.data["bounds"]
u.mkdir(args.odir)

# load source information
srcdata  = read_srcs(args.catalog)
srcpos, amps, omg, phi = srcdata[:2], srcdata[2], srcdata[3], srcdata[4]
if len(srcpos.shape) == 1: srcpos = srcpos[:,None]
# Which sources pass our requirements?
base_sids  = set(range(amps.size))
if args.minamp is not None:
    base_sids &= set(np.where(amps > args.minamp)[0])
if args.srcs is not None:
    selected = [int(w) for w in args.srcs.split(",")]
    base_sids &= set(selected)
base_sids = list(base_sids)

for ind in range(len(ids)):
    id = ids[ind]
    ofile = op.join(args.odir, id.replace(":","_")+".pdf")
    sids = lib.get_sids_in_tod(id, srcpos[:,base_sids], bounds[...,ind], base_sids, src_sys=sys, pad=poly_pad)
    if len(sids) == 0:
        print(f"{id} has no sources: skipping")
        continue
    else:
        print(f"found {len(sids)} sources")
    # insert source into tod
    entry = filedb.data[id]
    try:
        scan = actscan.ACTScan(entry, verbose=verbose>=2)
        if scan.ndet < 2 or scan.nsamp < 1: raise errors.DataMissing("no data in tod")
    except errors.DataMissing as e:
        print("%s skipped: %s" % (id, e))
        continue
    scan = scan[:,::down]
    scan.tod = scan.get_samples()

    # build pointing matrix
    P = lib.PmatTotVar(scan, srcpos[:,sids], perdet=False, sys=sys)

    # project pulsar into the given tod
    # prepare source parameter: [T,Q,U,omega_c,phi_c]
    srcs = np.array([amps, amps*0, amps*0, omg, phi])
    tod_new = P.forward(scan.tod*0, srcs)

    # plot tod
    plt.figure()
    plt.plot(tod_new[:,70500:70530].T, 'k-', alpha=0.5)
    plt.xlabel('samps')
    plt.ylabel('uK')
    plt.title(id)
    print("Writing:", ofile)
    plt.savefig(ofile)
    plt.close(0)
