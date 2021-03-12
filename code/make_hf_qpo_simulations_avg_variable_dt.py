import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_palette("rocket", n_colors=8)

import numpy as np
import pandas as pd
#from tqdm import tnrange, tqdm_notebook

import scipy.stats
import scipy.special
import scipy.fftpack
from tqdm import tqdm_notebook, tnrange 
import numba
from numba import jit, njit

from stingray import Lightcurve, Crossspectrum, Powerspectrum
from stingray import AveragedPowerspectrum
from stingray.simulator.simulator import Simulator
from stingray.events import EventList
from stingray.filters import DeadtimeFilterOutput, _paralyzable_dead_time, _non_paralyzable_dead_time

import warnings
warnings.filterwarnings('ignore')

from astropy.modeling import models
from stingray.modeling import PSDLogLikelihood, PSDParEst
from stingray import filters

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sbi import utils
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi

@jit
def lorentzian(x, amp, x0, fwhm):
    fac1 = amp * (fwhm/2)**2.
    fac2 = (fwhm/2)**2. + (x - x0)**2.
    return fac1/fac2

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

def get_deadtime_mask(ev_list, deadtime, bkg_ev_list=None,
                      dt_sigma=None, paralyzable=False,
                      return_all=False, verbose=False):
    """Filter an event list for a given dead time.

    Parameters
    ----------
    ev_list : array-like
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Other Parameters
    ----------------
    bkg_ev_list : array-like
        A background event list that affects dead time
    dt_sigma : float
        If specified, dead time will not have a single value but it will have
        a normal distribution with mean ``deadtime`` and standard deviation
        ``dt_sigma``.

    Returns
    -------
    mask : array-like, optional
        The mask that filters the input event list and produces the output
        event list.
    additional_output : object
        Object with all the following attributes. Only returned if
        `return_all` is True
        uf_events : array-like
            Unfiltered event list (events + background)
        is_event : array-like
            Boolean values; True if event, False if background
        deadtime : array-like
            Dead time values
        bkg : array-like
            The filtered background event list

    """
    additional_output = DeadtimeFilterOutput()

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(np.ones(len(ev_list), dtype=bool),
                            np.zeros(len(bkg_ev_list), dtype=bool))
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    additional_output.uf_events = tot_ev_list
    additional_output.is_event = ev_kind
    additional_output.deadtime = deadtime
    additional_output.uf_mask = np.ones(tot_ev_list.size, dtype=bool)
    additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]

    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    # if deadtime is a float, then it's either 
    # the constant dead time value or the mean 
    # of the distribution to be used for drawing 
    # dead time values
    if np.size(deadtime) == 1:
    
        if deadtime <= 0.:
            if deadtime < 0:
                raise ValueError("Dead time is less than 0. Please check.")
            retval = [np.ones(ev_list.size, dtype=bool), additional_output]
            return retval


        if dt_sigma is not None:
            deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
            deadtime_values[deadtime_values < 0] = 0.
        else:
            deadtime_values = np.zeros(nevents) + deadtime
    # otherwise it's a list of dead time values to be used 
    # as an empirical distribution to draw from:
    else:
        deadtime_values = np.random.choice(deadtime, replace=True, size=nevents)

    initial_len = len(tot_ev_list)
    # Note: saved_mask gives the mask that produces tot_ev_list_filt from
    # tot_ev_list. The same mask can be used to also filter all other arrays.
    if paralyzable:
        tot_ev_list_filt, saved_mask = \
            _paralyzable_dead_time(tot_ev_list, deadtime_values)

    else:
        tot_ev_list_filt, saved_mask = \
            _non_paralyzable_dead_time(tot_ev_list, deadtime_values)
    del tot_ev_list

    ev_kind = ev_kind[saved_mask]
    deadtime_values = deadtime_values[saved_mask]
    final_len = tot_ev_list_filt.size
    if verbose:
        log.info(
            'filter_for_deadtime: '
            '{0}/{1} events rejected'.format(initial_len - final_len,
                                             initial_len))

    retval = saved_mask[all_ev_kind]

    if return_all:
        # uf_events: source and background events together
        # ev_kind : kind of each event in uf_events.
        # bkg : Background events
        additional_output.uf_events = tot_ev_list_filt
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.bkg = tot_ev_list_filt[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval


@jit(nopython=False)
def simulate_deadtime(param, deadtime, freq=None, tseg=10.0, dt_hires=1e-5, dt=0.005, deadtime_type="const"):
    """
    Simulate a light curve with a QPO and with dead time. 
    
    Parameters
    ----------
    param : iterable
        The list of QPO parameters:
            * fractional rms amplitude
            * centroid frequency
            * quality factor (centroid frequency / FWHM)
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
    qpo_amp = 1.0 # amplitude of the QPO in CSD
    qpo_rms = param[0] # absolute rms amplitude of the QPO
    qpo_x0 = param[1] # centroid position of the PSD
    qpo_qual = param[2] # quality factor for the QPO
    qpo_fwhm = qpo_x0 / qpo_qual # Lorentzian FWHM, calculated from centroid and quality factor
    mean_cr = param[3] # mean count rate in the light curve

    npoints = int(np.round(tseg/dt_hires)) # total number of points in original light curve

    # count rate in nustar bins
    mean_cr_hires = mean_cr * dt_hires

    if freq is None:
        df_hires = 1.0/tseg # frequency resolution of the PSD/CSD
        fmax_hires = 0.5/dt_hires # maximum frequency in the CSD/PSD

        # list of frequencies
        freq = np.arange(df_hires, fmax_hires+df_hires, df_hires)

    # generate theoretical spectrum
    mspec = lorentzian(freq, qpo_amp, qpo_x0, qpo_fwhm)

    lc = simulate_lc(mspec, dt_hires, npoints, mean_cr_hires, qpo_rms)
    lc.counts[lc.counts < 0] = 0.0

    # apply counts
    counts1 = np.random.poisson(lc.counts)
    counts2 = np.random.poisson(lc.counts)

    # generate events from counts
    times1, times2 = generate_events(lc.time, counts1, counts2)

    # apply dead time mask
    mask1 = get_deadtime_mask(times1, deadtime, return_all=False)
    mask2 = get_deadtime_mask(times2, deadtime, return_all=False)

    times1_dt = times1[mask1]
    times2_dt = times2[mask2]
    
    # create Lightcurve objects
    lc1 = Lightcurve.make_lightcurve(times1, dt=dt, tseg=tseg, tstart=0.0)
    lc1_dt = Lightcurve.make_lightcurve(times1_dt, dt=dt, tseg=tseg, tstart=0.0)

    lc2 = Lightcurve.make_lightcurve(times2, dt=dt, tseg=tseg, tstart=0.0)
    lc2_dt = Lightcurve.make_lightcurve(times2_dt, dt=dt, tseg=tseg, tstart=0.0)

    return lc1, lc2, lc1_dt, lc2_dt, times1_dt, times2_dt

def generate_simulator_function(tseg=10.0, dt_hires=1e-5, dt=0.005, 
                                deadtime=0.0025, segment_size=1.0, 
                                summary_type="avgcsd", f=0.01, 
                                deadtime_type="const"):
    def simulation(param):
        """
        Generate a simulated data set with a single QPO given a parameter set.

        Parameters
        ----------
        param : iterable
            A list of parameters:
                * Fractional RMS amplitude of the QPO
                * centroid position x0 of teh QPO
                * quality factor (x0/fwhm) of the QPO
                * average count rate of the light curve

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
        #param = [rms, x0, qual, mean_cr]
        qpo_rms = param[0] # absolute rms amplitude of the QPO
        qpo_x0 = param[1] # centroid position of the PSD
        qpo_qual = param[2] # quality factor for the QPO
        qpo_fwhm = qpo_x0 / qpo_qual # Lorentzian FWHM, calculated from centroid and quality factor
        mean_cr = param[3] # mean count rate in the light curve


        lc1, lc2, lc1_dt, lc2_dt, times1_dt, times2_dt = simulate_deadtime(param, 
                                                     freq=None, 
                                                     tseg=tseg, 
                                                     dt_hires=dt_hires, 
                                                     dt=dt, 
                                                     deadtime=deadtime,
                                                     deadtime_type=deadtime_type)


        if summary_type == "psd":
            ps = Powerspectrum(lc1_dt+lc2_dt, norm="frac")
            return torch.as_tensor(ps.power)
        elif summary_type == "csd":
            cs = Crossspectrum(lc1_dt, lc2_dt, norm="frac")
            return torch.as_tensor(cs.power)
        elif summary_type == "avg":
            aps = AveragedPowerspectrum(lc1_dt+lc2_dt, segment_size, 
                                        norm="frac", silent=True)
            return torch.as_tensor(aps.power)
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
    
    lower_bounds = torch.tensor([0.1, 300.0, 5.0, 500])
    upper_bounds = torch.tensor([0.5, 500.0, 30.0, 1500.0])
    
    
    prior = utils.BoxUniform(
            low = lower_bounds,
            high = upper_bounds
            )

    f=0.01
    dead_time_obs = np.loadtxt("../data/deadtime_nustar.dat")    
    
    simulation_kwargs = {"tseg":10.0, "dt_hires":1e-5, "dt":0.5/1000.0, "deadtime":dead_time_obs, 
                         "summary_type":"avglogbin", "segment_size":1.0, "deadtime_type":"dist"}
    
    sim_func = generate_simulator_function(**simulation_kwargs) 
    simulator_wrapper, prior = prepare_for_sbi(sim_func, prior)

    # setup the inference procedure with the SNPE-C procedure
    inference = SNPE(prior=prior)
    
    num_sim = 5000
    num_iter = 10
    
    for i in range(num_iter):
        print("I am on simulation set %i."%i)
        theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_sim)
        np.savetxt("sim_hf_avgpsd_variabledt_logbin_theta%i.dat"%i, np.array(theta))
        np.savetxt("sim_hf_avgpsd_variabledt_logbin_x%i.dat"%i, np.array(x))


    return

if __name__ == "__main__":
     main()
