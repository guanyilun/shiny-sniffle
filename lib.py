import numpy as np
from enlib import pmat, sampcut, nmat, config, fft, utils
from scipy import integrate
from enact import nmat_measure

################
# beam profile #
################

def profile(phi, D=1/40):
    """pulsar profile: we assumed a von Mises pulse profile"""
    kappa = np.log(2)/(2*np.sin(np.pi*D/2)**2)
    return np.exp(kappa*(np.cos(phi)-1))

def phase(t, omega_c, phi_c):
    """phase of the pulsar, omega_c is the angular frequency, phi_c is
    the initial phase, so the phase is given by
      phi(t) = phi_c + omega_c t
    here we will assume a constant phase model.
    """
    return phi_c + omega_c * t

###################
# pointing matrix #
###################

class PmatPtsrcVar(pmat.PmatPtsrc):
    """Re-use PmatPtsrc but replace the core function to a modified
    version which takes a different format of src, with nparam being
    dec, ra, T, Q, U, omg_c, phi_c, ibx, iby, ibxy."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def apply(self, dir, tod, srcs, tmul=None, pmul=None):
        if tmul is None: tmul = self.tmul
        if pmul is None: pmul = self.pmul
        while srcs.ndim < 4: srcs = srcs[:,None]
        # Handle angle wrapping without modifying the original srcs array
        wsrcs = srcs.copy()
        wsrcs[...,:2] = utils.rewind(srcs[...,:2], self.ref) # + self.dpos
        core = get_core(tod.dtype)
        # boresight: (t, az, el)
        core.pmat_ptsrc3(dir, tmul, pmul, tod.T, wsrcs.T,
                        self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T,
                        self.rbox.T, self.nbox.T, self.yvals.T,
                        self.beam[1], self.beam[0,-1], self.rmax,
                        self.cells.T, self.ncell.T, self.cbox.T)
        # Copy out any amplitudes that may have changed
        srcs[...,2:5] = wsrcs[...,2:5]  # 2:5 -> T,Q,U in nparams

class PmatTotVar:
    def __init__(self, scan, srcs, ndir=1, perdet=False, sys="cel"):
        # Build source parameter struct for PmatPtsrc
        # srcs: {dec, ra, amp, omg_c, phi_c} => [nsrc, 5]
        # nparams: dec, ra, T, Q, U, omg_c, phi_c, ibx, iby, ibxy
        self.params = np.zeros([srcs.shape[-1],ndir,scan.ndet if perdet else 1,10],np.float)
        # dec, ra
        self.params[:,:,:,:2]  = srcs[::-1,None,None,:2].T
        # omg_c, phi_c
        # self.params[:,:,:,5:7] = srcpos[::-1,None,None,3:].T  # actually not needed in init
        # T, Q, U: assume unpolarized: amp -> T
        # default to cmb unit, better to use flux unit later
        # self.params[:,:,:,2] = srcpos[::-1,None,None,2]   # actually not needed in init
        # ibx, iby = 1: circular beam
        self.params[:,:,:,-3:-1] = 1
        self.psrc = PmatPtsrcVar(scan, self.params, sys=sys)
        self.pcut = pmat.PmatCut(scan)
        # Extract basic offset. Warning: referring to scan.d is fragile, since
        # scan.d is not updated when scan is sliced
        self.off0 = scan.d.point_correction
        self.off  = self.off0*0
        self.el   = np.mean(scan.boresight[::100,2])
        self.point_template = scan.d.point_template
        self.cut = scan.cut
    def forward(self, tod, amps, pmul=1):
        # Amps should be [nsrc,ndir,ndet|1,npol]
        params = self.params.copy()
        # this assumes that the input amps params start from tqu onwards (no ra,dec)
        params[...,2:2+amps.shape[-1]]   = amps
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

###############
# noise model #
###############

class NmatTot:
    def __init__(self, scan, model=None, window=None, filter=None):
        model  = config.get("noise_model", model)
        window = config.get("tod_window", window)*scan.srate
        nmat.apply_window(scan.tod, window)
        self.nmat = nmat_measure.NmatBuildDelayed(model, cut=scan.cut_noiseest, spikes=scan.spikes)
        self.nmat = self.nmat.update(scan.tod, scan.srate)
        nmat.apply_window(scan.tod, window, inverse=True)
        self.model, self.window = model, window
        self.ivar = self.nmat.ivar
        self.cut  = scan.cut
        # Optional extra filter
        if filter:
            freq = fft.rfftfreq(scan.nsamp, 1/scan.srate)
            fknee, alpha = filter
            with utils.nowarn():
                self.filter = (1 + (freq/fknee)**-alpha)**-1
        else: self.filter = None
    def apply(self, tod):
        nmat.apply_window(tod, self.window)
        ft = fft.rfft(tod)
        self.nmat.apply_ft(ft, tod.shape[-1], tod.dtype)
        if self.filter is not None: ft *= self.filter
        fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
        nmat.apply_window(tod, self.window)
        return tod
    def white(self, tod):
        nmat.apply_window(tod, self.window)
        self.nmat.white(tod)
        nmat.apply_window(tod, self.window)

#############
# utilities #
#############

def dets2id(dets):
    return np.array([int(bytes.decode(d).split('_')[1]) for d in dets]).astype(int)

def get_beam_area(beam):
    r, b = beam
    return integrate.simps(2*np.pi*r*b,r)

def get_core(dtype):
    from enlib.pmat import pmat_core_32, pmat_core_64
    if dtype == np.float32:
        return pmat_core_32.pmat_core
    else:
        return pmat_core_64.pmat_core

def rhand_polygon(poly):
    """Returns True if the polygon is ordered in the right-handed convention,
    where the sum of the turn angles is positive"""
    poly = np.concatenate([poly,poly[:1]],0)
    vecs = poly[1:]-poly[:-1]
    vecs /= np.sum(vecs**2,1)[:,None]**0.5
    vecs = np.concatenate([vecs,vecs[:1]],0)
    cosa, sina = vecs[:-1].T
    cosb, sinb = vecs[1:].T
    sins = sinb*cosa - cosb*sina
    coss = sinb*sina + cosb*cosa
    angs = np.arctan2(sins,coss)
    tot_ang = np.sum(angs)
    return tot_ang > 0

def pad_polygon(poly, pad):
    """Given poly[nvertex,2], return a new polygon where each vertex has been moved
    pad outwards."""
    sign  = -1 if rhand_polygon(poly) else 1
    pwrap = np.concatenate([poly[-1:],poly,poly[:1]],0)
    vecs  = pwrap[2:]-pwrap[:-2]
    vecs /= np.sum(vecs**2,1)[:,None]**0.5
    vort  = np.array([-vecs[:,1],vecs[:,0]]).T
    return poly + vort * sign * pad

def get_sids_in_tod(id, src_pos, bounds, isids=None, src_sys="cel", pad=0):
    if isids is None: isids = list(range(src_pos.shape[-1]))
    if bounds is not None:
        poly    = bounds*utils.degree
        poly[0] = utils.rewind(poly[0],poly[0,0])
        # bounds are defined in celestial coordinates. Must convert srcpos for comparison
        mjd     = utils.ctime2mjd(float(id.split(".")[0]))
        if src_sys != "cel":
            srccel = coordinates.transform(src_sys, "cel", src_pos, time=mjd)
        else:
            srccel = src_pos
        srccel[0] = utils.rewind(srccel[0], poly[0,0])
        if pad: poly = pad_polygon(poly.T, pad).T
        accepted = np.where(utils.point_in_polygon(srccel.T, poly.T))[0]
        sids     = [isids[i] for i in accepted]
    else:
        sids     = isids
    return sids
