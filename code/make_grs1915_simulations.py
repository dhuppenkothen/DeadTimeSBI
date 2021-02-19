import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=8)
pal = sns.color_palette()

import glob

import numpy as np
import pandas as pd
from tqdm import tnrange, tqdm_notebook

import scipy.stats
import scipy.special
import scipy.fftpack
from tqdm import tqdm_notebook, tnrange 
import numba
from numba import jit, njit

from stingray import Lightcurve, Crossspectrum, Powerspectrum
from stingray import AveragedPowerspectrum, AveragedCrossspectrum
from stingray.simulator.simulator import Simulator
from stingray.events import EventList
from stingray.io import load_events_and_gtis
from stingray.gti import cross_gtis


import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn 
import torch.nn.functional as F 

import sbi.utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

def read_data(file_list):
    
    # load the data using Stingray
    data_b = load_events_and_gtis(file_list[0])
    data_a = load_events_and_gtis(file_list[1])

    # filter energies
    a_mask = ((data_a.energy_list >= 3.0) & (data_a.energy_list <= 80.0))
    b_mask = ((data_b.energy_list >= 3.0) & (data_b.energy_list <= 80.0))

    data_a.ev_list = data_a.ev_list[a_mask]
    data_b.ev_list = data_b.ev_list[b_mask]

    return data_a, data_b

def generate_lightcurves(file_list, dt=0.5/10.0, dt_plot = 1.0):
    """
    Generate light curves out of NuSTAR data
    
    Parameters
    ----------
    file_list : iterable
        A list of file names and paths to the FPMA and FPMB 
        fits files of an observation
        
    Returns
    -------
    lca, lcb : stingray.Lightcurve objects
        Light curves for FPMA and FPMB 
        
    """

    data_a, data_b = read_data(file_list)
    
    # set tstart and tseg
    tstart = data_a.t_start
    tseg = data_a.t_stop - data_a.t_start

    # get GTIs
    gti_list = cross_gtis([data_a.gti_list, data_b.gti_list])


    # make light curves
    lca = Lightcurve.make_lightcurve(data_a.ev_list, dt, gti=gti_list, 
                                      tstart=tstart, tseg=tseg)
    lcb = Lightcurve.make_lightcurve(data_b.ev_list, dt, gti=gti_list, 
                                      tstart=tstart, tseg=tseg)

    # coarser light curves for plotting
    lca_plot = lca.rebin(dt_plot)
    lcb_plot = lcb.rebin(dt_plot)
    
    # make a diagnostic plot
    #fig, ax = plt.subplots(1, 1, figsize=(8,4))

    #pal = sns.color_palette()
    #lw = 2
    #ds = "steps-mid"
    #a = 0.6

    #obs_name = file_list[0].split("/")[-1][2:13]
    
    #ax.plot(lca_plot.time, lca_plot.countrate, lw=lw, ds=ds, alpha=a, c=pal[1])
    #ax.plot(lcb_plot.time, lcb_plot.countrate, lw=lw, ds=ds, alpha=a, c=pal[3])
    #ax.set_xlabel("Time [NuSTAR MET]")
    #ax.set_ylabel("Count rate [cts/s]")
    #ax.set_title("ObsID: " + str(obs_name))
    
    return lca, lcb


@jit
def lorentzian(x, amp, x0, fwhm):
    fac1 = amp * (fwhm/2)**2.
    fac2 = (fwhm/2)**2. + (x - x0)**2.
    return fac1/fac2


def two_lorentzian(x, amp1, fwhm1, amp2, x2, fwhm2):
    l1 = lorentzian(x, amp1, 0.0, fwhm1)
    l2 = lorentzian(x, amp2, x2, fwhm2)
    
    return l1 + l2

def two_lorentzian_harmonic(x, amp1, fwhm1, amp2, x2, fwhm2, amp3):
    l1 = lorentzian(x, amp1, 0.0, fwhm1)
    l2 = lorentzian(x, amp2, x2, fwhm2)
    l3 = lorentzian(x, amp3, 2*x2, fwhm2)

    return l1 + l2 + l3


@jit(nopython=True)
def extract_and_scale(long_lc, red_noise, npoints, mean_counts, rms):
    """
    i) Make a random cut and extract a light curve of required
    length.

    ii) Rescale light curve i) with zero mean and unit standard
    deviation, and ii) user provided mean and rms (fractional
    rms * mean)

    Parameters
    ----------
    long_lc : numpy.ndarray
        Simulated lightcurve of length 'npoints' times 'red_noise'
    
    red_noise : float
        A multiplication factor for the length of the light curve, 
        to deal with red noise leakage
    
    npoints : int
        The total number of data points in the light curve
        
    mean_counts : float
        The mean counts per bin of the light curve to be 
        simulated
        
    rms : float [0, 1]
        The fractional rms amplitude of the variability in the 
        light curve.

    Returns
    -------
    lc : numpy.ndarray
        Normalized and extracted lightcurve of lengtha 'N'
    """
    if red_noise == 1:
        lc = long_lc
    else:
        # Make random cut and extract light curve of length 'N'
        extract = \
            np.random.randint(npoints-1,
                                      red_noise*npoints - npoints+1)
        lc = np.take(long_lc, np.arange(extract, extract + npoints))

    avg = np.mean(lc)
    std = np.std(lc)

    return (lc-avg)/std * mean_counts * rms + mean_counts


@jit(nopython=False)
def simulate_lc(mspec, dt, npoints, mean_counts, rms, tstart = 0.0, red_noise=1.0):
    """
    
    
    
    """

    time = dt*np.arange(npoints) + tstart

    a1 = np.random.normal(size=(2,len(mspec))) * np.sqrt(mspec)

    f = a1[0] + 1j * a1[1]

    f[0] = mean_counts

    # Obtain real valued time series
    f_conj = np.conjugate(np.array(f))

    cts = np.fft.irfft(f_conj, n=npoints)

    lc = Lightcurve(time, extract_and_scale(cts, red_noise, npoints, mean_counts, rms),
                err=np.zeros_like(time) + np.sqrt(mean_counts),
                err_dist='gauss', dt=dt, skip_checks=True)

    return lc

@jit(nopython=True)
def generate_events(time, counts1, counts2):

    cs1 = np.cumsum(counts1)
    cs2 = np.cumsum(counts2)

    times1 = np.zeros(cs1[-1])
    times2 = np.zeros(cs2[-1])

    ncounts = len(counts1)

    for i in range(ncounts):
        times1[cs1[i]:cs1[i+1]] = time[i]
        times2[cs2[i]:cs2[i+1]] = time[i]
        
    return times1, times2

@jit(nopython=False)
def simulate_deadtime(param, deadtime, freq=None, tseg=10.0, dt_hires=1e-5, dt=0.005, deadtime_type="const"):
    """
    Simulate a light curve with a QPO and with dead time. 
    
    Parameters
    ----------
    param : iterable
        The list of QPO parameters:
            * fractional rms amplitude of the entire light curve
            * fwhm of the zero-point Lorentzian
            * relative amplitude of the second Lorentzian
            * centroid frequency of the second Lorentzian
            * quality factor (centroid frequency / FWHM) of the second Lorentzian
            * mean count rate of the light curve
            
    deadtime : float or iterable
        The dead time. Either a single value (constant dead time)
        or an iterable with samples drawn from an empirical distribution.
        Ideally, the number of elements in this list should be much larger 
        than the number of photons to be simulated
        
    freq : numpy.ndarray
        An array of frequencies to use when generating a light curve from 
        a power spectrum
        
    tseg : float
        The length of the light curve
    
    dt_hires : float
        The resolution of the instrument to be simulated; generally higher 
        than the resolution of the output light curve. Used for simulating 
        dead time accurately
        
    dt : float
        The desired time resolution of the output light curve.
    
    deadtime_type : str in {"const" | "dist"}
        The type of dead time to use. If "const", then `deadtime` should be 
        a float. If `dist`, then `deadtime` is assumed to encode draws from 
        an empirical distribution.
    
    Returns
    -------
    lc1, lc2: stingray.Lightcurve objects
        Simulated light curves for two independent detectors, assuming no 
        dead time is present in the data
        
    lc1_dt, lc2_dt: stingray.Lightcurve objects
        Simulated light curves for two independent detectors, assuming 
        dead time is present
    
    """
    qpo1_amp = 1.0 # amplitude of the zero-point QPO in CSD
    qpo_rms = param[0] # absolute rms amplitude of the QPO
    qpo1_fwhm = param[1] # centroid position of the PSD
    qpo2_amp = param[2]
    qpo2_x0 = param[3]
    qpo2_qual = param[4] # quality factor for the QPO
    qpo2_fwhm = qpo2_x0 / qpo2_qual # Lorentzian FWHM, calculated from centroid and quality factor
    qpo3_amp = param[5] * param[2] # relative amplitude of the QPO harmonic
    mean_cr = param[6] # mean count rate in the light curve

    npoints = int(np.round(tseg/dt_hires)) # total number of points in original light curve

    # count rate in nustar bins
    mean_cr_hires = mean_cr * dt_hires

    if freq is None:
        df_hires = 1.0/tseg # frequency resolution of the PSD/CSD
        fmax_hires = 0.5/dt_hires # maximum frequency in the CSD/PSD

        # list of frequencies
        freq = np.arange(df_hires, fmax_hires+df_hires, df_hires)

    # generate theoretical spectrum
    mspec = two_lorentzian_harmonic(freq, qpo1_amp, qpo1_fwhm, qpo2_amp, qpo2_x0, qpo2_fwhm, qpo3_amp)

    lc = simulate_lc(mspec, dt_hires, npoints, mean_cr_hires, qpo_rms)
    lc.counts[lc.counts < 0] = 0.0

    # apply counts
    counts1 = np.random.poisson(lc.counts)
    counts2 = np.random.poisson(lc.counts)

    # generate events from counts
    times1, times2 = generate_events(lc.time, counts1, counts2)

    # apply dead time mask
    mask1 = filters.get_deadtime_mask(times1, deadtime, return_all=False)
    mask2 = filters.get_deadtime_mask(times2, deadtime, return_all=False)

    times1_dt = times1[mask1]
    times2_dt = times2[mask2]
    
    # create Lightcurve objects
    #lc1 = Lightcurve.make_lightcurve(times1, dt=dt, tseg=tseg, tstart=0.0)
    lc1_dt = Lightcurve.make_lightcurve(times1_dt, dt=dt, tseg=tseg, tstart=0.0)

    #lc2 = Lightcurve.make_lightcurve(times2, dt=dt, tseg=tseg, tstart=0.0)
    lc2_dt = Lightcurve.make_lightcurve(times2_dt, dt=dt, tseg=tseg, tstart=0.0)

    return lc1_dt, lc2_dt


def generate_simulator_function(tseg=10.0, dt_hires=1e-5, dt=0.005, 
                                deadtime=0.0025, segment_size=1.0, summary_type="psd",
                                f=0.01, deadtime_type="const"):
    def simulation(param):
        """
        Generate a simulated data set with a single QPO given a parameter set.

        Parameters
        ----------
        param : iterable
            A list of parameters:
                * fractional rms amplitude of the entire light curve
                * fwhm of the zero-point Lorentzian
                * relative amplitude of the second Lorentzian
                * centroid frequency of the second Lorentzian
                * quality factor (centroid frequency / FWHM) of the second Lorentzian
                * mean count rate of the light curve
                
        summary_type:
            What to return as a summary. Options are
                * "psd": return the unaveraged powers in the PSD
                * "avg": return averaged PSD, requires `segment_size`

        Returns
        -------
        summary : np.ndarray
            An array of summary statistics

        """

        param = np.array(param)


        lc1_dt, lc2_dt = simulate_deadtime(param, 
                                         freq=None, 
                                         tseg=tseg, 
                                         dt_hires=dt_hires, 
                                         dt=dt, 
                                         deadtime=deadtime,
                                         deadtime_type=deadtime_type)


        if summary_type == "psd":
            ps = Powerspectrum(lc1_dt+lc2_dt, norm="frac")
            return torch.as_tensor(ps.power)
        elif summary_type == "avg":
            aps = AveragedPowerspectrum(lc1_dt+lc2_dt, segment_size, 
                                        norm="frac", silent=True)
            return torch.as_tensor(aps.power)
        elif summary_type == "csd":
            cs = Crossspectrum(lc1_dt, lc2_dt, norm="frac")
            return torch.as_tensor(cs.power)
        elif summary_type == "logbin":
            ps = Powerspectrum(lc1_dt+lc2_dt, norm="frac")
            ps_bin = ps.rebin_log(f)
            return torch.as_tensor(ps_bin.power)
        elif summary_type == "avglogbin":
            aps = AveragedPowerspectrum(lc1_dt+lc2_dt, segment_size, 
                                        norm="frac", silent=True)
            aps_bin = aps.rebin_log(f)
            return torch.as_tensor(aps_bin.power)
        else:
            raise ValueError("Type of summary to be returned not recognized!")
            
    return simulation

def main():

    # read out first observation
    obsid = "80401312002"
    datadir = "/astro/users/dhuppenk/dhuppenk/data/grs1915/nustar/"
    fits_files = glob.glob(datadir + obsid + "/bary/*C.evt")
    
    # set time scales for light curves and (currently disabled) plotting
    dt = 0.5/300.0 # time resolution for analysis
    dt_plot = 5.0 # time resolution for plotting
    
    min_lc_len = 124.0 # minimum length of a GTI to be included
    
    segment_size = 16.0 # size of an LC segment within the averaged PSD
    nsegments = 15.0 # number of segments in each averaged PSD
    tseg_total = segment_size * nsegments # total length of LC for the avg PSD
    f = 0.01 # scale factor for logarithmic binning
    
    # I will split up simulations into individual runs so they're robust to failures
    num_sim = 5 # number of simulations per iteration
    num_iter = 3 # number of iterations for the simulation runs
    
    
    # generate light curves from observation
    lca, lcb = generate_lightcurves(fits_files,dt=dt, dt_plot=dt_plot)
    lc = lca + lcb
    
    # get the GTIs and keep only those of a certain length
    gti_diff = lc.gti[:,1] - lc.gti[:,0]
    gti_mask = (gti_diff > min_lc_len)
    lc.gti = lc.gti[gti_mask]
    
    # split list of light curves by GTI
    lc_list = lc.split_by_gti()
    
    # keep only enough of the first GTI to make the averaged PSD
    lc1_trunc = lc_list[0].truncate(start=lc_list[0].time[0], stop=lc_list[0].time[0]+tseg_total,
                                     method="time")
    
    # generate averaged PSD
    aps1_trunc = AveragedPowerspectrum(lc1_trunc, segment_size=segment_size, norm="frac")
    
    # generate logbinned version of averaged PSD
    aps1_bin = aps1_trunc.rebin_log(f=f)
    
    # set up priors for SBI
    rms_prior = [0.1, 0.6]
    fwhm0_prior = [1.0, 10.0]
    amp1_prior = [0.5, 20.0]
    nu1_prior = [1.0, 5.0]
    qual1_prior = [3, 100]
    amp3_prior = [0.01, 1.0]
    meancr_prior = [20, 500]
    
    lower_bounds = torch.tensor([rms_prior[0], fwhm0_prior[0], amp1_prior[0], 
                                 nu1_prior[0], qual1_prior[0], amp3_prior[0], 
                                 meancr_prior[0]])
    upper_bounds = torch.tensor([rms_prior[1], fwhm0_prior[1], amp1_prior[1], 
                                 nu1_prior[1], qual1_prior[1], amp3_prior[1], 
                                 meancr_prior[1]])
    
    prior = utils.BoxUniform(
            low = lower_bounds,
            high = upper_bounds
            )
    
    # relevant keyword arguments for SBI simulator
    simulation_kwargs = {"tseg":tseg_total, "dt_hires":0.0001, "dt":dt, "f":f,
                         "deadtime":0.0025, "summary_type":"avglogbin", "segment_size":segment_size}
    
    
    sim_func = generate_simulator_function(**simulation_kwargs)
    
    simulator, prior = prepare_for_sbi(sim_func, prior)
    
    
    inference = SNPE(prior=prior)
    
    for i in range(num_iter):
        print("I am on iteration %i"%i
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim)
    
        np.savetxt("grs1915_16s_15seg_theta%i.dat"%i, theta)
        np.savetxt("grs1915_16s_15seg_x%i.dat"%i, x)
    
    
    return
