# -*- tab-width: 4 -*-

"""similar to sim_data, but this does it more seriously by making my
own pointing matrix.

"""

import numpy as np, os, os.path as op
from enlib import errors, utils as u, config, bench
from enact import filedb, actscan, actdata
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
    ra, dec, amp, T, phi, D = np.loadtxt(fname, unpack=True)
    # convert period[ms] to omega_c: T = 2*pi/omega_c
    omg = 2*np.pi/(T*1e-3)
    srcdata = np.array([ra*u.degree, dec*u.degree, amp, omg, phi, D])
    # if only one source is specified, make sure it has a shape
    # compatible with multiple sources
    if len(srcdata.shape) == 1: srcdata = srcdata[:, None]
    return srcdata

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
srcpos, amps, omg, phi, D = srcdata[:2], srcdata[2], srcdata[3], srcdata[4], srcdata[5]
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
    ofile = op.join(args.odir, id.replace(":","_")+".png")
    # find sources
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
    # build source lists
    # ra, dec, T, Q, U, omg, phi
    t0 = u.mjd2ctime(scan.mjd0)
    phi0 = np.mod(phi+t0*omg, 2*np.pi)
    srcs = np.array([srcpos[0], srcpos[1], amps, amps*0, amps*0, omg, phi0, D])
    # srcs = np.array([srcpos[0], srcpos[1], amps, amps*0, amps*0])
    # build pointing matrix
    P = lib.PmatTotVar(scan, srcs, perdet=False, sys=sys)
    # P = lib.PmatTot(scan, srcpos[:,sids], perdet=False, sys=sys)
    # project pulsar into the given tod
    # prepare source parameter: [T,Q,U,omega_c,phi_c]
    tod  = P.forward(scan.tod*0, amps)
    # scan.tod  = P.forward(scan.tod*0, amps)
    # plot tod for debugging
    if 1:
        sel = slice(70350,70850)
        plt.figure()
        plt.plot(tod[:,sel].T, 'k-', alpha=0.5)
        plt.xlabel('samps')
        plt.ylabel('uK')
        plt.title(id)
        # also plot the profile by itself
        # optionally overlay the actual profile
        ctimes = scan.boresight[:,0] + t0
        # get pulses from all sources if there are more than one
        # pulses = np.sum([lib.profile(lib.phase(ctimes, omg[i], phi[i])) for i in range(len(omg))], axis=0)
        pulses = amps*lib.profile(lib.phase(ctimes, omg, phi), D=1/10)
        plt.plot(pulses[sel], 'r-', alpha=0.2)
        # plt.yscale('log')
        # plt.ylim(bottom=1e-30)
        print("Writing:", ofile)
        plt.savefig(ofile)
        plt.close()

    # build search space as multiple sources
    # assuming that the search space only contain one source
    omgs = np.linspace(60,80,32)
    # omgs = np.linspace(60,80,100)  # update: 220209
    phis = np.linspace(0,2*np.pi,32)
    # nsrc = len(omgs) * len(phis)
    # srcs = np.zeros(7, nsrc, dtype=np.float64)
    # srcs[0,:] = srcpos[0,None]
    # srcs[1,:] = srcpos[1,None]
    # srcs[2,:] = 1
    # srcs[5:7,:] = 1
    # Note that it is important for amp to be 1 to be able to extract the response as its flux. 
    # srcs= np.array([[srcpos[0], srcpos[1], 1, 0, 0, o, p, D[0]] for o in omgs for p in phis]).T
    # srcs= np.array([[srcpos[0], srcpos[1], 1, 0, 0, omg, phi0, D]]).T  # use original pointing
    srcs= np.array([[srcpos[0], srcpos[1], 1, 0, 0, 70.6, 6.13, D]]).T  # use close-enough value
    # srcs= np.array([[srcpos[0], srcpos[1]]])
    with bench.show("create pointing matrix"):
        P = lib.PmatTotVar(scan, srcs, perdet=False, sys=sys)
        # P = lib.PmatTot(scan, srcs[:,0], perdet=False, sys=sys)

    # I should be able to use the same pointing matrix to do a search
    # of pulsars by treating different period and phases as different
    # sources, and estimating their amplitudes together. Let me give
    # that a try below

    # a factor to convert uK to mJy
    beam_area = lib.get_beam_area(scan.beam)
    _, uids   = actdata.split_detname(scan.dets) # Argh, stupid detnames
    freq      = scan.array_info.info.nom_freq[uids[0]]
    fluxconv  = u.flux_factor(beam_area, freq*1e9)/1e3
    print("fluxconv = ", fluxconv)

    # prepare pointing matrix and noise model for searching
    # for signal-only test I won't apply any noise model
    # N = NmatTot(scan, model="uncorr", window=2.0, filter=highpass)
    # srcs

    # rhs (= P'N"d)
    # N.apply(tod)
    with bench.show("project TOD to src space"):
        rhs = P.backward(tod, ncomp=1)

    # div (= P'N"P)
    tod[:] = 0
    P.forward(tod, rhs*0+1)
    # N.apply(scan.tod)
    div = P.backward(tod, ncomp=1)
    fluxconv = 1
    # get to mJy unit
    div /= fluxconv**2
    rhs /= fluxconv
    flux = rhs / div
    print("flux =",flux)
    import pdb;pdb.set_trace()
    dflux = div**-0.5
    # res = np.vstack([srcs[-3:-1], flux.reshape(1,-1)])
    res = flux.reshape(1,-1)
    # np.savetxt('ohf_orig.txt', res.T, header='amp')
    # np.savetxt('ohf.txt', res.T, header='omg phi amp')
    # input omg and phi0
    print(f"omg={omg}, phi={phi0}")
