import matplotlib as mpl

mpl.rcParams['font.size']  = 12
mpl.rcParams['figure.dpi'] = 180

from matplotlib import pyplot as plt
from cutslib.pathologies_tools import get_pwv
import glob, h5py, numpy as np
import argparse, os, os.path as op

def get_timerange(f):
    todname = f.split('flux_')[1].split('.ar')[0]
    t_beg = int(todname.split('.')[0])
    t_end = int(todname.split('.')[1])
    return t_beg, t_end

def has_overlap(beg_1, beg_2, end_1=None, end_2=None, threshold=400):
    if abs(beg_2 - beg_1) < threshold: return True
    else: return False
    
def match_files(files1, files2):
    data = []
    files = sorted(files1+files2, key=lambda x: get_timerange(x)[0])
    for f in files:
        t_beg, t_end = get_timerange(f)
        if len(data) == 0:
            data.append([f])
        else:
            t_beg_old, t_end_old = get_timerange(data[-1][0])
            if has_overlap(t_beg, t_beg_old):
                data[-1] = data[-1] + [f]
            else:
                data.append([f])
    return data

def get_opts(f):
    opts = {
        'markersize': 2,
        'capsize': 2,
        'alpha': 0.5
    }
    if 'ar5' in f:
        opts.update({'fmt': 'r.'})
        # opts.update({'c': 'r'})
        # opts.update({'style': 'r.'})
    elif 'ar6' in f:
        # opts.update({'c': 'b'})
        opts.update({'fmt': 'b.'})
    else: raise ValueError()
    return opts

def match_pwv(files):
    ctimes = [get_timerange(f)[0] for f in files]
    pwv = get_pwv(ctimes)
    lookup = {}
    for i in range(len(ctimes)):
        lookup[ctimes[i]] = pwv[i]
    return lookup 

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--odir', default='out')
parser.add_argument('-v', action='store_true', default=False)
parser.add_argument("--oname", default="flux.pdf")
parser.add_argument("--sid", help="source id in hdf file", type=int, default=0)
parser.add_argument("--pwv-y", type=float, default=35)
parser.add_argument("--dt-y", type=float, default=38)
parser.add_argument("--annotate", action="store_true")
parser.add_argument("--xlim", default=None)
parser.add_argument("--ylim", default=None)
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

files_ar5 = glob.glob("out_photometry/*.ar5_f090.hdf")
files_ar6 = glob.glob("out_photometry/*.ar6_f090.hdf")

# match pa5 and pa6
files = match_files(files_ar5, files_ar6)
# get a dictionary of pwv to look up
pwv = match_pwv(files_ar5+files_ar6)

# start plotting
cmap = plt.cm.Greens
fig, ax = plt.subplots(1, 1, figsize=(16,4))
t_shift = 0
t_ref_old = 0
duration = 0
for fs in files:
    # use the beginning time as a common reference
    t_ref = get_timerange(fs[0])[0]
    t_end = []
    for f in fs:
        with h5py.File(f, "r") as fp:
            if args.sid not in fp['sids'][:]: continue
            sid = np.where(fp['sids'][:]==args.sid)[0]
            t     = fp['t'][:][sid] - t_ref
            flux  = fp['flux'][:][sid]
            dflux = fp['dflux'][:][sid]
            # select valid flux measurements only
            sel   = flux != 0
            sel  *= dflux < 0.8*1e3  # error < 800 mJy
            if np.sum(sel) < 10: continue
            t    += t_shift
            opts  = get_opts(f)
            ax.errorbar(t[sel], flux[sel]*1e-3, dflux[sel]*1e-3, **opts)
            # ax.plot(t[sel], flux[sel]*1e-3, '.', **opts)
            # find out where things stop
            t_end.append(np.max(t))
    if len(t_end) == 0: continue
    ax.axvspan(t_shift, max(t_end), color=cmap(pwv[t_ref]), alpha=0.5)
    if args.annotate: ax.text((2*t_shift+max(t_end))/3, args.pwv_y, f"{pwv[t_ref]:.2f}", fontsize=8)
    ax.axvline(t_shift, linestyle='--', color='k')
    if t_ref_old != 0:  # this only run in the second loop so it's okay to use variables defined later
        t_jump = t_ref - (t_ref_old + duration)
        if args.annotate: ax.text(t_shift, args.dt_y, f"{t_jump/60:.0f}", fontsize=8, ha="center")
    duration = max(t_end) - t_shift
    t_shift = max(t_end)
    t_ref_old = t_ref

plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Flux [Jy]', fontsize=14)
plt.title("PA5 in red, PA6 in blue, PWV in shade, $\Delta t$ (in min) in dashed line", pad=20)
if args.xlim is not None: plt.xlim(eval(args.xlim))
if args.ylim is not None: plt.ylim(eval(args.ylim))
ofile = op.join(args.odir, args.oname)
print("Writing:", ofile)
plt.savefig(ofile, bbox_inches='tight')
