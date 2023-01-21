# -*- tab-width: 4 -*-
"""search for crab pulsar

"""
import numpy as np
import os.path as op, h5py
from collections import OrderedDict
from enlib import errors, config, mpi, sampcut, pmat, coordinates
from pixell import utils
from pulsar import *
from enact import filedb, actscan, actdata

import lib

class Pulsar(PulsarTiming):
    def __init__(self, src, fname, profile=None, ephem="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp"):
        # Consider reading in the dispersion delay too. It's totally irrelevant in the mm though
        # profile is a function that takes a phase and returns a weight
        self.profile = profile if profile is not None else lambda x: np.ones_like(x)
        assert callable(profile)
        data = np.loadtxt(fname, usecols=(3,4,6,8), ndmin=2).T
        self.src   = np.array(src)  # (ra,dec,I), with ra,dec in radians 
        self.tref  = utils.mjd2ctime(data[0])
        self.t0    = data[1]
        self.freq  = data[2]
        self.dfreq = data[3]*1e-15
        self.P     = 1/self.freq
        self.dP    = -1/self.freq**2 * self.dfreq
        self.ephem = ephem
    def obstime2phase(self, ctime, delay=0, site=None, interp=True, step=10):
        if site is None: site = coordinates.default_site
        tdb  = tdt2tdb(tai2tdt(utc2tai(ctime)))
        ra, dec = self.src[1], self.src[0]
        tdb -= calc_obs_delay(ctime, np.array([ra, dec]), site, ephem=self.ephem, interp=interp, step=step)
        tdb -= delay
        ind, x = self.calc_ind_x(tdb)
        phase  = x % 1
        return phase
    def obstime2profile(self, ctime, delay=0, site=None, interp=True, step=10):
        phase = self.obstime2phase(ctime, delay, site, interp, step)
        return self.profile(phase)

class PmatPtsrcTransient(pmat.PmatPtsrc):
    """Re-use PmatPtsrc but replace the core function to a modified
    version which takes a different format of src, with nparam being
    dec, ra, T, Q, U, ibx, iby, ibxy. profiles are (nsrc, nsamps) array with 
    max=1.
    """
    def __init__(self, profiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiles = profiles
    def apply(self, dir, tod, srcs, tmul=None, pmul=None):
        if tmul is None: tmul = self.tmul
        if pmul is None: pmul = self.pmul
        while srcs.ndim < 4: srcs = srcs[:,None]
        # Handle angle wrapping without modifying the original srcs array
        wsrcs = srcs.copy()
        wsrcs[...,:2] = utils.rewind(srcs[...,:2], self.ref) # + self.dpos
        core = lib.get_core(tod.dtype)
        core.pmat_ptsrct(dir, tmul, pmul, tod.T, wsrcs.T, self.profiles.T,
                        self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T,
                        self.rbox.T, self.nbox.T, self.yvals.T,
                        self.beam[1], self.beam[0,-1], self.rmax,
                        self.cells.T, self.ncell.T, self.cbox.T)
        # Copy out any amplitudes that may have changed
        srcs[...,2:5] = wsrcs[...,2:5]  # 2:5 -> T,Q,U in nparams

class PmatTotTransient:
    def __init__(self, scan, srcs, ndir=1, perdet=False, sys="cel"):
        # Build source parameter struct for PmatPtsrc
        self.params = np.zeros([len(srcs),ndir,scan.ndet if perdet else 1,8],float)
        srcpos = np.array([p.src[:2] for p in srcs]).T  # [{ra,dec}, nsrc]
        self.params[:,:,:,:2] = srcpos[::-1,None,None,:].T  # ::-1 to convert to [dec,ra]
        # circular beam
        self.params[:,:,:,5:7] = 1
        ctime = utils.mjd2ctime(scan.mjd0) + scan.boresight[:,0]
        profiles = np.array([p.obstime2profile(ctime) for p in srcs])
        self.psrc = PmatPtsrcTransient(profiles, scan, self.params, sys=sys)
        self.pcut = pmat.PmatCut(scan)
        # Extract basic offset. Warning: referring to scan.d is fragile, since
        # scan.d is not updated when scan is sliced
        self.off0 = scan.d.point_correction
        self.off  = self.off0*0
        self.el   = np.mean(scan.boresight[::100,2])
        self.point_template = scan.d.point_template
        self.cut = scan.cut
    def set_offset(self, off):
        self.off = off*1
        self.psrc.scan.offsets[:,1:] = actdata.offset_to_dazel(self.point_template + off + self.off0, [0,self.el])
    def forward(self, tod, amps, pmul=1):
        # Amps should be [nsrc,ndir,ndet|1,ncomp] # where ncomp is [T,Q,U] or [T]
        params = self.params.copy()
        params[...,2:2+amps.shape[-1]] = amps 
        self.psrc.forward(tod, params, pmul=pmul)
        sampcut.gapfill_linear(self.cut, tod, inplace=True)
        return tod
    def backward(self, tod, amps=None, pmul=1, ncomp=3):
        params = self.params.copy()
        tod = sampcut.gapfill_linear(self.cut, tod, inplace=False, transpose=True)
        self.psrc.backward(tod, params, pmul=pmul)
        if amps is None: amps = params[...,2:2+ncomp]
        else: amps[:] = params[...,2:2+amps.shape[-1]]
        return amps

if __name__ == '__main__':
    config.set("downsample", 1, "Amount to downsample tod by")
    config.set("gapfill", "linear", "Gapfiller to use. Can be 'linear' or 'joneig'")
    config.default("pmat_interpol_pad", 10.0, "Number of arcminutes to pad the interpolation coordinate system by")
    
    parser = config.ArgumentParser()
    parser.add_argument("sel")
    parser.add_argument("--odir", default="out")
    parser.add_argument("--pulsar-file")
    parser.add_argument("--nbin", type=int, default=10)
    parser.add_argument("-v", "--verbose",   action="count", default=0)
    parser.add_argument("-q", "--quiet",     action="count", default=0)
    parser.add_argument("--sys", type=str, default="cel")
    # parser.add_argument("--oname", type=str, default="flux_crab.txt")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    srcs = OrderedDict() 
    srcs['tau_a'] = np.array([83.63308333*utils.degree, 22.0145*utils.degree, 1.0, 0.0, 0.0])  # ra dec T Q U
    # generate 10 random sources
    np.random.seed(args.seed)
    for i in range(10):
        srcs["rand%d" % i] = np.array([srcs["tau_a"][0] + (np.random.rand()-0.5)*10*utils.arcmin,
                                       srcs["tau_a"][1] + (np.random.rand()-0.5)*10*utils.arcmin,
                                       1.0, 0.0, 0.0])

    filedb.init()
    db       = filedb.scans.select(args.sel)
    ids      = db.ids
    sys      = args.sys
    comm     = mpi.COMM_WORLD
    dtype    = np.float32
    verbose  = args.verbose - args.quiet
    down     = config.get("downsample")
    poly_pad = 3*utils.degree
    bounds   = db.data["bounds"]
    highpass = [3, 10]
    utils.mkdir(args.odir)

    # Initialize pulsar template to search. We focus on the Crab pulsar
    # for now. In this case, I will assume a boxcar template that split
    # the phases into 10 bins. This should be equivalent to making a map
    # in 10 phase bins. 
    def boxcar(i, n):
        # i is the index of the bin, n is the total number of bins
        def profile_(phase):
            # phase -> (0, 1)
            idx = np.floor(phase*n)
            prof = np.zeros_like(phase)
            prof[idx == i] = 1
            prof = np.roll(prof, -len(prof)//(2*args.nbin))  # shift by half a bin to align 1st bin center to zero
            return prof
        return profile_
    pulsars = [Pulsar(srcs["tau_a"], args.pulsar_file, profile=boxcar(i,args.nbin)) for i in range(args.nbin)]
    # # also add some background sources
    # for _, src in srcs.items():
    #     pulsars += [Pulsar(src, args.pulsar_file, profile=boxcar(i,args.nbin)) for i in range(args.nbin)]
    srcpos = np.array([pulsar.src[:2] for pulsar in pulsars]).T  # ({ra,dec},nsrc) = (2, nbin)
    base_sids = list(range(len(pulsars)))

    # rhss = []
    # divs = []
    for ind in range(comm.rank, len(ids), comm.size):
        id = ids[ind]
        print(f"{ind:>3d}/{len(ids)}: {id}")
        # find sources
        sids = lib.get_sids_in_tod(id, srcpos[:,base_sids], bounds[...,ind], base_sids, src_sys=sys, pad=poly_pad)
        if len(sids) == 0:
            print(f"{id} has no sources: skipping")
            continue
        entry = filedb.data[id]
        try:
            # calibration happened here
            scan = actscan.ACTScan(entry, verbose=verbose>=2)
            if scan.ndet < 2 or scan.nsamp < 1: raise errors.DataMissing("no data in tod")
        except errors.DataMissing as e:
            print("%s skipped: %s" % (id, e))
            continue
        scan = scan[:,::down]
        scan.tod = scan.get_samples()
        utils.deslope(scan.tod, w=5, inplace=True)
        scan.tod = scan.tod.astype(dtype)

        # build pointing matrix and noise model
        P = PmatTotTransient(scan, pulsars, perdet=False, sys=sys)
        N = lib.NmatTot(scan, model="uncorr", window=2.0, filter=highpass)

        # rhs
        N.apply(scan.tod)
        rhs = P.backward(scan.tod, ncomp=1)

        # div
        scan.tod[:] = 0
        P.forward(scan.tod, rhs*0+1)  # assume ncomp=1
        N.apply(scan.tod)
        div = P.backward(scan.tod, ncomp=1)

        # a factor to convert uK to mJy
        beam_area = lib.get_beam_area(scan.beam)
        _, uids   = actdata.split_detname(scan.dets)
        freq      = scan.array_info.info.nom_freq[uids[0]]
        fluxconv  = utils.flux_factor(beam_area, freq*1e9)/1e3

        # get to mjy unit
        div /= fluxconv**2
        rhs /= fluxconv

        # divs.append(div)
        # rhss.append(rhs)

        # instead of accumulating, just save the results per TOD
        ofile = op.join(args.odir, f"{ids[ind]}_flux.h5")
        with h5py.File(ofile, "w") as hfile:
            hfile["id"]    = id.encode()
            hfile["sids"]  = sids
            hfile["dets"]  = np.char.encode(scan.dets)
            hfile["div"]   = div
            hfile["rhs"]   = rhs

    # div = np.sum(divs, axis=0)
    # rhs = np.sum(rhss, axis=0)

    # rhs = utils.allreduce(rhs, comm)
    # div = utils.allreduce(div, comm)

    # if comm.rank == 0:
    #     flux = np.ravel(rhs / div)
    #     dflux = np.ravel(div**-0.5)
    #     ofile = op.join(args.odir, args.oname)
    #     np.savetxt(ofile, np.array([flux, dflux]).T, fmt="%10.3f %10.3f", header="flux dflux")
