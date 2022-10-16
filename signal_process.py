#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:48:33 2021

@author: mahdi
"""
import os
import gc
import obspy
import sys
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.iris import Client as ClientI
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from pyrocko import obspy_compat, model, moment_tensor, util
from pyrocko.model import Event
from pulse_identifier import *
from scipy.signal import hilbert
from beat.config import BEATconfig, ProblemConfig, SeismicConfig, WaveformFitConfig, \
    SamplerConfig, dump_config, get_parameter, SeismicNoiseAnalyserConfig, SMCConfig
from collections import OrderedDict
from beat.heart import Parameter, ArrivalTaper, Filter, BandstopFilter,\
    ReferenceLocation
from theano import config as tconfig
from pyrocko.cake import load_model
from pyrocko.guts import List
from beat.heart import SeismicDataset
from beat.utility import dump_objects
from obspy.core.util.attribdict import AttribDict
from obspy import UTCDateTime


def arrivals(streams, inven, ev_lat, ev_lon, ev_dep):
    """
    arrivals(streams, inven, ev_lat, ev_lon, ev_dep)
    gives the required headers to the waveforms (Distnces, arrivals, backazimuth)
    it requires a velocity model file for obspy taup.

    Parameters
    ----------
    streams : obspy.stream
        waveforms.
    inven : obspy.inventory
        station responses.
    ev_lat : float
        event latitude.
    ev_lon : float
        event longitude.
    ev_dep : float
        event depth.

    Returns
    -------
    waveforms with headers.

    """
    client = ClientI()
    for net in inven:
        for stn in net:
            result = client.distaz(stalat=stn.latitude, stalon=stn.longitude,
                                   evtlat=ev_lat, evtlon=ev_lon)
            dist_deg = result['distance']
            model = TauPyModel(model='/home/mahdi/project/BEAT_automation/SLU.npz')
            arrivals = model.get_travel_times(source_depth_in_km=ev_dep,
                                              distance_in_degree=dist_deg,
                                              phase_list=['P', 'S'])
            P = [arrive.time for arrive in arrivals if arrive.name == 'P']
            S = [arrive.time for arrive in arrivals if arrive.name == 'S']
            if len(P) == 0 or len(S) == 0:
                P = dist_deg * 111.19 / 4.4
                S = dist_deg * 111.19 / 2.2
            else:
                P = np.min(P)
                S = np.min(S)
            for tr in streams.select(station=stn.code):
                tr.stats.back_azimuth = result['backazimuth']
                tr.stats.fp_arrival = P
                tr.stats.fs_arrival = S
                tr.stats.dist = dist_deg * 111.19
                tr.stats.latitude = stn.latitude
                tr.stats.longitude = stn.longitude
                tr.stats._format = 'AH'
                tr.stats.ah = AttribDict({'version': '1.0',
                                          'event': AttribDict({'latitude': ev_lat, 'longitude': ev_lon, 'depth': ev_dep, 'origin_time': None, 'comment': 'null'}),
                                          'station': AttribDict({'code': tr.stats.station, 'channel': tr.stats.channel, 'type': 'null', 'latitude': stn.latitude, 'longitude': stn.longitude, 'elevation': 0.0, 'gain': 0.0, 'normalization': 0.0, 'poles': [], 'zeros': []}),
                                          'record': AttribDict({'type': 1, 'ndata': tr.stats.npts, 'delta': tr.stats.delta, 'max_amplitude': 0, 'start_time': tr.stats.starttime, 'abscissa_min': 0.0, 'comment': 'null', 'log': 'null'}),
                                          'extras': [result['backazimuth'], P, S, dist_deg * 111.19]})


def resp_rot(st, inv):
    """
    resp_rot(st,inv)
    Rotate the traces from NEZ to RTZ

    Parameters
    ----------
    st : obspy.stream
        waveforms.
    inv : obsy.inventory
        station responses.

    Returns
    -------
    rotated waveforms.

    """
    st.merge(method=1, fill_value=0, interpolation_samples=-1)
    ts = np.max([tr.stats.starttime for tr in st])
    te = np.min([tr.stats.endtime for tr in st])
    st.trim(ts, te).rotate(method="->ZNE", inventory=inv)
    st.trim(ts, te).rotate(method="NE->RT", inventory=inv)


def filter_st(st, sampling_rate, freqmin, freqmax, bandstop=False):
    """
    filter_st(st, sampling_rate, freqmin, freqmax, bandstop = False)
    This function applies the filtering and decimating,
    a bandpass filter as one highpass and a lowpass,
    a bandreject for Ocean noise 0.12-0.25

    Parameters
    ----------
    st : obspy.stream
        signals.
    sampling_rate : int
        sampling rate of signals to be downsampled to .
    freqmin : float
        DESCRIPTION.
    freqmax : float
        DESCRIPTION.
    bandstop : Boolean, optional
        If applying ocean-noise filter is desirable. The default is False.

    Returns
    -------
    returning processed traces.

    """
    st.taper(0.02)
    st.filter('highpass', freq=freqmin, corners=3, zerophase=False)
    st.filter('lowpass', freq=freqmax, corners=3, zerophase=False)
    if bandstop:
        st.filter('bandstop', freqmin=0.12, freqmax=0.25,
                  corners=3, zerophase=True)
    for tr in st:
        ndec = int(tr.stats.sampling_rate/sampling_rate)
        tr.decimate(ndec, no_filter=True)


def read_stns(stationfile):
    """
    read_stns(stationfile)
    the function Loads station name and its network
    The input file should be like `NET.STA`
    PQ.NBC1 XL.MG03

    Parameters
    ----------
    stationfile : string
        input file name.

    Returns
    -------
    nets : list
        includes names of seismic networks (String).
    stns : list
        includes names of seismic stations (String).

    """
    file1 = open(stationfile, 'r')
    ln = file1.readline()
    file1.close()
    codes = ln.split()
    nets = []
    stns = []
    for code in codes:
        net, stn = code.split('.', 1)
        nets.append(net)
        stns.append(stn)
    return nets, stns


def rms(data):
    """
    rms(data)
    Compute RMS (Root mean square decibels)

    Parameters
    ----------
    data : numpy array
        signal.

    Returns
    -------
    float
        RMS value.

    """
    if np.any(data != data):
        data[np.isnan(data)] = 0
    return float(10*np.log10(np.sqrt(np.mean(data**2))))


def snr(datas, datan):
    return float(rms(datas) - rms(datan))


def snrfilt(traces, lb_sw, ub_sw, lb_nw, ub_nw,
            cutoff, orgntime, topic, plotting=False, removing=False):
    """
    snrfilt(traces, lb_sw, ub_sw, lb_nw, ub_nw,
            cutoff, orgntime, plotting = False, topic, removing = False)
    This function estimate the SNR regarding signal and noise windows.

    Parameters
    ----------
    traces : TYPE
        waveforms.
    lb_sw : float
        lower band of signal window with respect to p-wave arrival time, and average velocity.
    ub_sw : float
        upper band of signal window with respect to p-wave arrival time, and average velocity.
    lb_nw : float
        lower band of noise window with respect to p-wave arrival time (before p-wave).
    ub_nw : float
        upper band of noise window with respect to p-wave arrival time (before p-wave).
    cutoff : float
        threshold value for SNR to remove signals.
    orgntime : UTCDateTime
        event origin time.
    plotting : Boolean, optional
        if Plotting is necessary. The default is False.
    topic : String
        DESCRIPTION.
    removing : Boolean, optional
        if removing waveforms from dataset is desirable. The default is False.

    Returns
    -------
    None.

    """
    ncol = 5
    N = 3
    fig, axes = plt.subplots(nrows=N, ncols=ncol, sharex=True,
                             sharey=False, figsize=(21, 15))
    j = 0
    for t, tr in enumerate(traces):
        noiw = tr.copy().trim(orgntime+tr.stats.fp_arrival-lb_nw,
                              orgntime+tr.stats.fp_arrival - ub_nw)
        if topic == 'SNR_b4_filt':
            tr.stats.remove = []
            sigw = tr.copy().trim(orgntime+tr.stats.fp_arrival - lb_sw,
                                  orgntime+tr.stats.fp_arrival + ub_sw)
            SNR = snr(sigw.data, noiw.data)
            tr.stats.SNR_b4_filt = SNR
        elif topic == 'SNR_AF_filt':
            sigw = tr.copy().trim(orgntime+tr.stats.dist/3.3 - lb_sw,
                                  orgntime+tr.stats.dist/3.3 + ub_sw)
            SNR = snr(sigw.data, noiw.data)
            tr.stats.SNR_AF_filt = SNR
        if SNR <= cutoff:
            color = 'r'
            tr.stats['remove'].append(topic)
            if removing:
                traces.remove(tr)
        else:
            color = 'k'
        if plotting:
            k = int(j / ncol)
            jj = j % ncol
            if k == N:
                fig.tight_layout(pad=0.1)
                fig.savefig(fname='detained_{}_{}.png'.format(topic, t), format='PNG', dpi=200)
                plt.close(fig)
                j = 0
                k = int(j / ncol)
                jj = j % ncol
                fig, axes = plt.subplots(nrows=N, ncols=ncol, sharex=True,
                                         sharey=False, figsize=(21, 15))
                axes[k, jj].plot(tr.times("timestamp"), tr.data, color,
                                 sigw.times("timestamp"), sigw.data,
                                 noiw.times("timestamp"), noiw.data)
                axes[k, jj].legend(['{} {}\n{:4.2f}'.format(tr.stats.station,
                                                            tr.stats.channel,
                                                            SNR),
                                    'Signal W.', 'Noise W.'], loc=1)
                j += 1
                continue
            axes[k, jj].plot(tr.times("timestamp"), tr.data, color,
                             sigw.times("timestamp"), sigw.data,
                             noiw.times("timestamp"), noiw.data)
            axes[k, jj].legend(['{} {}\n{:4.2f}'.format(tr.stats.station,
                                                        tr.stats.channel,
                                                        SNR),
                                'Signal W.', 'Noise W.'], loc=1)
            j += 1
    if plotting:
        fig.tight_layout(pad=0.1)
        fig.savefig(fname='detailed_{}_{}.png'.format(topic, t), format='PNG', dpi=200)
        plt.close(fig)
        del fig


def condalngth(traces, delta, length, orgntime,
               plotting=False, removing=False):
    """
    condalngth(traces, delta, length, orgntime, plotting = False, removing = False)

    This function measures the length of Coda waves after
    S-wave arrival time until meeting quarter of mean of moving SNR window.
    A constant noise, and moving signal windows are considered
    to compute a smooth version of the waveform so that we can compute
    the length of the Coda.

    Parameters
    ----------
    traces : obspy.stream
        waveforms.
    delta : float
        time intervals.
    length : float
        length of moving signal window.
    orgntime : UTCDateTime
        Origin time of event.
    plotting : boolean, optional
        if plotting is desired. The default is False.
    removing : boolean, optional
        if removing from traces (dataset) is desired. The default is False.

    Returns
    -------
    None.

    """
    ncol = 5
    N = 3
    fig, axes = plt.subplots(nrows=N, ncols=ncol, sharex=True,
                             sharey=False, figsize=(21, 15))
    j = 0
    for t, tr in enumerate(traces):
        noisew = tr.copy().trim(orgntime+tr.stats.fp_arrival-50,
                                orgntime+tr.stats.fp_arrival)
        sarrive = tr.stats.fs_arrival
        startpoint = tr.stats.starttime
        endpoint = startpoint + length
        if int((tr.stats.endtime-endpoint)/delta) <= 0:
            print(tr.stats.station, tr.stats.channel)
        data = np.zeros((int((tr.stats.endtime-endpoint)/delta), 2))
        for ii in range(int((tr.stats.endtime-endpoint)/delta)):
            if endpoint >= tr.stats.endtime or len(tr.copy().trim(startpoint, endpoint)) == 0:
                break
            signalw = tr.copy().trim(startpoint, endpoint)
            data[ii, :] = startpoint.timestamp, (np.max(np.abs(signalw.data)) /
                                                 np.sqrt(np.mean(noisew.data**2)))
            startpoint += delta
            endpoint = startpoint + length
        data = data[data[:, 1] != 0]
        data2 = data[data[:, 0] > (orgntime+sarrive).timestamp]
        if len(data2) == 0 or
        len(data2[data2[:, 1] < np.mean(data[:, 1])/2]) == 0:
            codalength = 0
            tr.stats.CD = codalength
            xx = len(data2)
            pass
        elif data2[0, 1] > np.mean(data[:, 1])/2:
            xx = np.where(data2[:, 1] < np.mean(data[:, 1])/2)[0][0]
            codalength = data2[xx, 0] - data2[0, 0]
            if codalength < 20:
                xx = np.where(data2[:, 1] < np.mean(data[:, 1])/2)[0][-1]
                codalength = data2[xx, 0] - data2[0, 0]
        elif data2[0, 1] <= np.mean(data[:, 1])/2:
            xx = np.where(data2[:, 1] < np.mean(data[:, 1])/2)[0][-1]
            codalength = data2[xx, 0] - data2[0, 0]
        data2 = data2[:xx, :]
        # some limitation should be put here like: codalength > 100 (sec) -> traces.remove(tr)
        if codalength <= 10.0 or codalength > 250:
            tr.stats['remove'].append('CD')
            color = 'r'
            if removing:
                traces.remove(tr)
        else:
            color = 'k'
        tr.stats.CD = codalength
        if plotting:
            k = int(j / ncol)
            jj = j % ncol
            if k == N:
                fig.tight_layout(pad=0.1)
                fig.savefig(fname='CD_results_{}.png'.format(t), format='PNG', dpi=200)
                plt.close(fig)
                j = 0
                k = int(j / ncol)
                jj = j % ncol
                fig, axes = plt.subplots(nrows=N, ncols=ncol, sharex=True,
                                         sharey=False, figsize=(21, 15))
                axes[k, jj].plot(tr.times("timestamp"),
                                 np.max(data[:, 1])*tr.data/np.max(np.abs(tr.data)),
                                 color,
                                 data[:, 0], data[:, 1], data2[:, 0],
                                 data2[:, 1], data[:, 0],
                                 np.ones(data[:, 0].shape)*np.mean(data[:, 1])/2)
                axes[k, jj].legend(['{} {}'.format(tr.stats.station,
                                                   tr.stats.channel),
                                    'Moving W.', 'C. L.\n{:4.2f}'.format(codalength),
                                    'Mean(SNR)'], loc=1)
                j += 1
                continue
            axes[k, jj].plot(tr.times("timestamp"),
                             np.max(data[:, 1])*tr.data/np.max(np.abs(tr.data)),
                             color, data[:, 0], data[:, 1], data2[:, 0],
                             data2[:, 1], data[:, 0],
                             np.ones(data[:, 0].shape)*np.mean(data[:, 1])/2)
            axes[k, jj].legend(['{} {}'.format(tr.stats.station, tr.stats.channel),
                                'Moving W.', 'C. L.\n{:4.2f}'.format(codalength),
                                'Mean(SNR)'], loc=1)
            j += 1
    if plotting:
        fig.tight_layout(pad=0.1)
        fig.savefig(fname='CD_results_{}.png'.format(t), format='PNG', dpi=200)
        plt.close(fig)
        del fig


def morlet_with_phase(f, dt, data, phase):
    """
    morlet_with_phase(f, dt, data, phase)

    The function computes the shifting wavelet (fourth-order morlet)

    Parameters
    ----------
    f : float
        frequency of morlet we want to produce.
    dt : float
        sampling rate of signal (wavelet).
    data : numpy array
        signal (wavelet).
    phase : float
        phase shift (degree).

    Returns
    -------
    morlet_shifted : numpy array
        shifted wavelet.

    """
    # Create Ricker Wavelet
    y = np.real(morlet(f, dt, data, order=4))
    # Hilbert Transformation
    analytic_signal = hilbert(y)
    # Imaginary Part
    im = np.imag(analytic_signal)
    # Real Part
    re = np.real(analytic_signal)
    morlet_phased = np.cos(phase)*re - np.sin(phase)*im
    morlet_shifted = np.concatenate([morlet_phased[np.argmax(morlet_phased) -
                                                   np.argmax(data):],
                                     morlet_phased[:np.argmax(morlet_phased) -
                                                   np.argmax(data)]])
    return morlet_shifted


def ricker_with_phase(f, dt, data, phase):
    """
    ricker_with_phase(f, dt, data, phase)

    The function computes the shifting wavelet (second-order ricker)

    Parameters
    ----------
    f : float
        frequency of ricker we want to produce.
    dt : float
        sampling rate of signal (wavelet).
    data : numpy array
        signal (wavelet).
    phase : float
        phase shift (degree).

    Returns
    -------
    ricker_shifted : numpy array
        shifted wavelet.

    """
    # Create Ricker Wavelet
    t, y = ricker(f, dt, data)
    # Hilbert Transformation
    analytic_signal = hilbert(y)
    # Imaginary Part
    im = np.imag(analytic_signal)
    # Real Part
    re = np.real(analytic_signal)
    ricker_phased = np.cos(phase)*re - np.sin(phase)*im
    ricker_shifted = np.concatenate([ricker_phased[np.argmax(ricker_phased) -
                                                   np.argmax(data):],
                                     ricker_phased[:np.argmax(ricker_phased) -
                                                   np.argmax(data)]])
    return ricker_shifted


def fitting_wavelet(traces, orgntime, a, b, freqmin, freqmax,
                    plotting=False, removing=False):
    """
    fitting_ricker(traces, orgntime, a, b, freqmin, freqmax,
    plotting = False, removing = False)

    This function does wavelet analysis on the time windows. At first,
    it tries to find the pulse period in the time windows, after that
    It computes the pearson's correlation coefficients between these
    signal and (2-o Ricker, 4-o morelet) wavelets.(Ertuncay&Costa,2019).
     and selects the wavelet with maximum CC, and removes the signals
     with CC less 0.6 from traces.
    Parameters
    ----------
    traces : obspy.stream
        wvaeform datset.
    orgntime : UTCDateTime
        Origin time.
    a : float
        lower limit of time window.
    b : float
        upper limit.
    freqmin : float
        lower band of frequency range.
    freqmax : float
        upper band of frequency range.
    plotting : boolean, optional
        if plotting is desired. The default is False.
    removing : boolean, optional
        if removing from traces (dataset) is desired. The default is False.

    Returns
    -------
    gives the traces headers (CC, and remove elements).

    """
    ncol = 5
    N = 3
    fig, axes = plt.subplots(nrows=N, ncols=ncol, figsize=(21, 15))
    j = 0
    for t, tr in enumerate(traces):
        signal = tr.copy().trim(orgntime+tr.stats.dist/3.3-a,
                                orgntime+tr.stats.dist/3.3+b).taper(0.1)
        try:
            Tp, is_pulse_dz, fit_wavelet = vel_wf_detector([signal],
                                                           1/np.max(np.abs(signal.data)), True)
        except Exception:
            traces.remove(tr)
        # Energy drop
        en_drop_ricker = np.zeros((72, 1))
        en_drop_morlet = np.zeros((72, 1))
        # Phase angles
        phase_angles = np.arange(0, 360, 5)
        for i, angle in enumerate(phase_angles):
            # Phase Shift to Ricker Wavelet
            phase = np.deg2rad(angle)
            # Apply phase angle to ricker wavelet
            shifted_wavelet_ricker = ricker_with_phase(1/Tp, signal.stats.delta, signal.data, phase)
            # shifted_wavelet_morlet = morlet_with_phase(1/Tp, signal.stats.delta, signal.data, phase)
            # Correlation Coefficient
            en_drop_ricker[i] = np.corrcoef(signal.data, shifted_wavelet_ricker)[0][1]
            # en_drop_morlet[i] = np.corrcoef(signal.data,shifted_wavelet_morlet)[0][1]
        max_corr_ricker = np.where(en_drop_ricker == np.array(en_drop_ricker).max())[0][0]
        max_corr_morlet = np.where(en_drop_morlet == np.array(en_drop_morlet).max())[0][0]
        if max_corr_ricker > max_corr_morlet:
            phased_wl = ricker_with_phase(1/Tp, signal.stats.delta, signal.data,
                                          np.deg2rad(phase_angles[max_corr_ricker]))
        else:
            phased_wl = morlet_with_phase(1/Tp, signal.stats.delta, signal.data,
                                          np.deg2rad(phase_angles[max_corr_morlet]))
        CC = np.round(np.corrcoef(signal.data/np.max(np.abs(signal.data)),
                                  phased_wl/np.max(np.abs(phased_wl)))[0][1], 1)
        x = np.std((signal.data - phased_wl))
        z = np.std(phased_wl)
        if np.round(CC, 1) <= 0.5 or np.round(x/z, 1) >= 1.5:
            tr.stats['remove'].append('SYN')
            color = 'r'
            if removing:
                traces.remove(tr)
        else:
            color = 'k'
        tr.stats.SYN = CC
        if plotting:
            k = int(j / ncol)
            jj = j % ncol
            if k == N:
                fig.tight_layout(pad=0.1)
                fig.savefig(fname='SYN_results_{}.png'.format(t), format='PNG', dpi=200)
                plt.close(fig)
                j = 0
                k = int(j / ncol)
                jj = j % ncol
                fig, axes = plt.subplots(nrows=N, ncols=ncol, figsize=(21, 15))
                axes[k, jj].plot(signal.times("timestamp"),
                                 signal.data,
                                 color,
                                 signal.times("timestamp"),
                                 phased_wl, 'b')
                axes[k, jj].legend(['{}\n{}\nObs'.format(tr.stats.station,
                                                         tr.stats.channel),
                                    'Syn\nCC:{:4.2f}'.format(CC)], loc=1)
                j += 1
                continue
            axes[k, jj].plot(signal.times("timestamp"),
                             signal.data, color,
                             signal.times("timestamp"),
                             phased_wl, 'b')
            axes[k, jj].legend(['{}\n{}\nObs'.format(tr.stats.station,
                                                     tr.stats.channel),
                                'Syn\nCC:{:4.2f}'.format(CC)], loc=1)
            j += 1
    if plotting:
        fig.tight_layout(pad=0.1)
        fig.savefig(fname='SYN_results_{}.png'.format(t), format='PNG', dpi=200)
        plt.close(fig)
        del fig


def rm_extra_tr(traces, dataset, inventory):
    """
    rm_extra_st(traces, dataset, inventory)

    This function removes the unwanted station signals from inventory and metas
    that already removed from traces, in the future process, this consumes less
    memory and speeds up the process.
    Parameters
    ----------
    traces : obspy.stream
        signals.
    dataset : obspy.stream
        main waveforms for further processing (unfiltered).
    inventory : obspy.inventory
        includes station responses.

    Returns
    -------
    inventory : obspy.inventory
        doesn't includes removed-station reponses.

    """
    stations = [st.code for net in inventory for st in net]
    for st in stations:
        if len(traces.select(station=st)) == 0:
            inventory = inventory.remove(station=st)
            [dataset.remove(tr) for tr in dataset.select(station=st)]
        elif len(traces.select(station=st)) != 0:
            for chan in ['Z', 'R', 'T']:
                if len(traces.select(station=st, channel='??'+chan)) == 0:
                    inventory = inventory.remove(station=st, channel=chan)
                    [dataset.remove(tr) for tr in dataset.select(station=st, channel=chan)]
    # [dataset.remove(tr) for tr in dataset.select(channel='??[NE]')]
    # [dataset.remove(tr) for tr in dataset if 'SNR_b4_filt' in tr.stats.remove or 'CD' in tr.stats.remove]
    for tr in dataset.select(channel='??[NE]'):
        dataset.remove(tr)
        inventory = inventory.remove(station=tr.stats.station, channel=tr.stats.channel)
    for tr in dataset:
        if 'SNR_b4_filt' in tr.stats.remove or 'CD' in tr.stats.remove:
            dataset.remove(tr)
            inventory = inventory.remove(station=tr.stats.station, channel=tr.stats.channel)


def zero_cross(traces, orgntime, a, b, threshold=7, removing=False):
    """
    zero_cross(traces, orgntime, a, b, removing=False)

    This function computes the number of crosses a trace has with zero, and also number of turning points.

    Parameters
    ----------
    traces : obspy.stream
        Signals.
    orgntime : UTCDateTime
        origin time of the event.
    a : float
        lower limit for the time window we want to measure zcross, and turning points.
    b : float
        upper limit for the time window.
    removing : Boolean, optional
        This parameter determines that we want to remove
        unselected signals from the main dataset (traces). The default is False.

    Returns
    -------
    traces : obspy.stream
        processed traces (measured).

    """
    for tr in traces:
        signal = tr.copy().trim(orgntime+tr.stats.dist/3.3-a,
                                orgntime+tr.stats.dist/3.3+b)
        tr.stats.zcross = (np.diff(np.sign(signal.data)) != 0).sum() - (signal.data == 0).sum()
        tr.stats.turnp = (np.diff(np.sign(np.diff(np.diff(signal.data)))) != 0).sum()
        if tr.stats.turnp < threshold:
            tr.stats['remove'].append('turnp')
            if removing:
                traces.remove(tr)
    return traces


def plot_metrics(traces, orgntime, topic, threshold):
    """
    plot_metrics(traces, orgntime, topic, threshold)

    This function plots metrics values in ncol * N figures

    Parameters
    ----------
    traces : obspy.stream.
        Traces should have metrics values in their headers,
        like tr.stats.dist, or tr.stats.CD.
    orgntime : UTCDateTime.
        origin time of seismic events.
    topic : string.
        this variable determines the type of metrics you are trying to plot.
    threshold : float.
        threshold values for those metrics we  want to plot.

    Returns
    -------
    Save plots.

    """
    ncol = 6
    N = 4
    fig, axes = plt.subplots(nrows=N, ncols=ncol,
                             sharex=True, sharey=False,
                             figsize=(21, 15))
    j = 0
    traces.traces.sort(key=lambda x: x.stats[topic], reverse=False)
    max_dist = np.max([tr.stats.dist for tr in traces])
    for t, tr in enumerate(traces):
        signal = tr.copy().trim(orgntime+tr.stats.dist/3.3-20,
                                orgntime+tr.stats.dist/3.3+max_dist/3.3)
        if (signal.stats[topic] < threshold and topic != 'CD') or topic in signal.stats.remove:
            color = 'r'
            legend = ['{}-{}\n{}:{:4.2f}'.format(signal.stats.station,
                                                 signal.stats.channel,
                                                 topic, signal.stats[topic])]
        elif topic in ['dist'] and len(signal.stats.remove) != 0:
            color = 'r'
            legend = ['{}-{}\n{}:{:4.2f}\n{}'.format(signal.stats.station,
                                                     signal.stats.channel,
                                                     topic,
                                                     signal.stats[topic],
                                                     signal.stats.remove)]
        else:
            color = 'k'
            legend = ['{}-{}\n{}:{:4.2f}'.format(signal.stats.station,
                                                 signal.stats.channel,
                                                 topic,
                                                 signal.stats[topic])]
        k = int(j / ncol)
        jj = j % ncol
        if k == N:
            fig.tight_layout(pad=0.1)
            fig.savefig(fname='{}_{}.png'.format(topic, t), format='PNG', dpi=200)
            plt.close(fig)
            j = 0
            k = int(j / ncol)
            jj = j % ncol
            fig, axes = plt.subplots(nrows=N, ncols=ncol,
                                     sharex=True, sharey=False,
                                     figsize=(21, 15))
            axes[k, jj].plot(signal.times(), signal.data, color)
            axes[k, jj].legend(legend, loc=1)
            j += 1
            continue
        axes[k, jj].plot(signal.times(), signal.data, color)
        axes[k, jj].legend(legend, loc=1)
        j += 1
    fig.tight_layout(pad=0.1)
    fig.savefig(fname='{}_{}.png'.format(topic, t), format='PNG', dpi=200)
    plt.close(fig)
    del fig


def plot_all_in_one(traces, orgntime, topic, threshold):
    """
    plot_all_in_one(traces, orgntime, topic, threshold)
    This function plots all metrics values in one figure to have meaningful analysis.

    Parameters
    ----------
    traces : obspy.stream.
        Traces should have metrics values in their headers,
        like tr.stats.dist, or tr.stats.CD.
    orgntime : UTCDateTime.
        origin time of seismic events.
    topic : string.
        this variable determines the type of metrics you are trying to plot.
    threshold : float.
        threshold values for those metrics we  want to plot.

    Returns
    -------
    Save the plots.

    """
    traces.traces.sort(key=lambda x: x.stats[topic], reverse=False)
    stations = [tr.stats.station+'_'+tr.stats.channel for tr in traces]
    fig = plt.figure(figsize=(24, 21), dpi=200)
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.02, right=0.98, bottom=0.05, top=0.98,
                          wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, :2])
    ax_histx = fig.add_subplot(gs[0, :2], sharex=ax)
    max_dist = np.max([tr.stats.dist for tr in traces])
    for i, tr in enumerate(traces):
        signal = tr.copy().trim(orgntime+tr.stats.dist/3.3-20,
                                orgntime+tr.stats.dist/3.3+max_dist/3.3)
        if (tr.stats[topic] < threshold and topic != 'CD') or topic in tr.stats.remove:
            color = 'r'
        elif topic in ['dist'] and len(tr.stats.remove) != 0:
            color = 'r'
        else:
            color = 'k'
        ax.plot(signal.data/np.max(np.abs(signal.data))+i, signal.times(), color)
        ax_histx.plot(i, signal.stats[topic], color+'o')

    ax_histx.plot(range(len(traces)), threshold*np.ones((len(traces), 1)), 'b')
    ax.set_xticks(range(len(traces)))
    ax.set_xticklabels(stations, rotation='vertical')
    ax_histx.grid(b=True, which='both', axis='both')
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.title(topic)
    plt.savefig(fname='{}_all_in_one.png'.format(topic), format='PNG', dpi=200)
    plt.close(fig)
    del fig

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


def main(origin_time, evlat, evlon, evdep, mag, f1, f2, r, samp_rate):
    client = Client("IRIS")
    t1 = origin_time - 2.5 * (1/f1)                                     # starting time for signals (lower band for retrieving waveforms)
    t2 = origin_time + 2 * r * 111.19 / 3.3                              # ending time for signals (upper band for retrieving waveforms)
    event = Event(lat=evlat, lon=evlon, depth=evdep * 1000., time=origin_time)        # convert event's information to BEAT format (Pyrocko)
    stfile = 'stn.txt'                                                          # a station file includes network and station names like PQ.NBC1 XL.MG03 ...
    print("############ Fetching stations' information")
    if origin_time.month >= 3 and origin_time.month <= 10:
        noise_filter = True
    else:
        noise_filter = False
    if not os.path.exists(stfile):
        inv = client.get_stations(minlatitude=evlat-r,
                                  maxlatitude=evlat+r,
                                  maxlongitude=evlon+r,
                                  minlongitude=evlon-r,
                                  starttime=t1, endtime=t2,
                                  channel="BH?,HH?",
                                  level='channel')
        with open(stfile, 'w') as f:
            for net in inv:
                for st in net:
                    if st == 'WAPA':
                        continue
                    f.write('{}.{}\t'.format(net.code, st.code))
    nets, stns = read_stns(stfile)
    # if not os.path.exists('inv.xml'):
    inv = obspy.Inventory()
    for net, st in zip(nets, stns):
        try:
            # print('Working on:',net, st)
            inv.extend(client.get_stations(network=net, station=st,
                                           starttime=t1, endtime=t2,
                                           channel="BH?,HH?",
                                           level="response"))
        except Exception:
            # print('Not Working on:',net, st)
            continue
    inv.write('inv.xml', format="STATIONXML")
    # else:
    #     inv = obspy.read_inventory('inv.xml', format="STATIONXML")
    #     print("## IT'S DONE BEFORE ##")
    print('############## Retrieving waveforms ')
    # if os.path.exists('waveforms.ah'):
    #     st_all = obspy.read('waveforms.ah')
    #     print("## IT'S DONE BEFORE ##")
    # else:
    st_all = obspy.Stream()
    st_all.clear()
    for net in inv:
        for st in net:
            # print('Trying station:',net.code, st.code)
            try:
                st_all += client.get_waveforms(net.code, st.code, "*", "BH?,HH?", t1, t2, attach_response=True)
            except Exception:
                inv = inv.remove(station=st.code)
                # print('No data for',net.code, st.code)
                continue
    [st_all.remove(tr) for tr in st_all if (tr.stats.endtime - tr.stats.starttime) < np.fix(t2-t1)]
    st_all.detrend('spline', order=2, dspline=t2-t1)
    [st_all.remove(tr) for gap in st_all.get_gaps() if gap[6] < 50 for tr in st_all.select(station=gap[1], channel=gap[3])]
    arrivals(st_all, inv, evlat, evlon, evdep)

    print('############## Starting preprocessing the waveforms')

    pre_filt = (0.001, 0.004, 10.0, 20.0)
    st_all.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
    [resp_rot(st_all.select(station=st), inv) for st in [tr.stats.station for tr in st_all]]
    st_all.write('waveforms.ah', format='AH')

    print('# of Signals:', len(st_all))
    """
    APPLYING SNR FILTER, the noisw window is referred to orgntime
    and signal window in centerd on p-wave arrivals
    """
    snrfilt(st_all, 0.5*(1/f2), 1.5*(1/f2),  2.3*(1/f2), 0.3*(1/f2),
            4.0, origin_time, 'SNR_b4_filt', False, False)
    # plot_metrics(st_all, origin_time, 'SNR_b4_filt', 1.0)
    # plot_all_in_one(st_all, origin_time, 'SNR_b4_filt', 1.0)
    # print('First SNR filter:', len(st_all))
    condalngth(st_all, 2.5, 5.0, origin_time, False, False)
    st_all_unfilt = st_all.copy()
    filter_st(st_all, samp_rate, f1, f2, noise_filter)
    [st_all.remove(tr) for tr in st_all if (tr.stats.endtime - tr.stats.starttime) < np.fix(t2-t1)]
    # plot_metrics(st_all, origin_time, 'CD', 0.0)
    # plot_all_in_one(st_all, origin_time, 'CD', 0.0)
    # print('Coda-length filter:', len(st_all))
    zero_cross(st_all, origin_time, 0.5*(1/f1), 1.5*(1/f1), 7, False)
    # plot_metrics(st_all, origin_time, 'zcross', 7.0)
    # plot_all_in_one(st_all, origin_time, 'zcross', 7.0)
    # print('Zcross filter:', len(st_all))
    # plot_metrics(st_all, origin_time, 'turnp', 7.0)
    # plot_all_in_one(st_all, origin_time, 'turnp', 7.0)
    snrfilt(st_all, 0.5*(1/f1), 1.5*(1/f1),  2.3*(1/f1), 0.3*(1/f1),
            3.0, origin_time, 'SNR_AF_filt', False, False)
    # plot_metrics(st_all, origin_time, 'SNR_AF_filt', 5.0)
    # plot_all_in_one(st_all, origin_time, 'SNR_AF_filt', 5.0)
    # print('Second SNR filter:', len(st_all))
    fitting_wavelet(st_all, origin_time, 0.5*(1/f1),
                    1.5*(1/f1), f1, f2, False, False)
    # plot_metrics(st_all, origin_time, 'SYN', 0.61)
    # plot_all_in_one(st_all, origin_time, 'SYN', 0.61)
    # print('Wavelet filter:', len(st_all))
    # plot_all_in_one(st_all, origin_time, 'dist', 0.0)
    plot_metrics(st_all_unfilt, origin_time, 'fp_arrival', 0.0)
    # plot_all_in_one(st_all_unfilt, origin_time, 'fp_arrival',0.0)
    print('############## writing signals')
    rm_extra_tr(st_all, st_all_unfilt, inv)
    with open('myinfo.txt', 'w') as f:
        for net in inv:
            for st in net:
                f.write('    - {}\n'.format(st.code))
    plot_metrics(st_all_unfilt, origin_time, 'dist', 0.0)
    obspy_compat.plant()

    # convert unfiltered data to pyrocko traces and inventory to stations
    pyrocko_traces = st_all_unfilt.to_pyrocko_traces()
    stations = inv.to_pyrocko_stations()

    # add rotated channels to stations
    for s in stations:
        s.set_event_relative_data(event)
        pios = s.guess_projections_to_rtu(out_channels=('R', 'T', 'Z'))
        for (_, _, out_channels) in pios:
            for ch in out_channels:
                s.add_channel(ch)

    # convert pyrocko traces to beat traces
    beat_traces = []
    channel_mapping = {
        'HHR': 'R',
        'HHT': 'T',
        'HHZ': 'Z',
        'BHR': 'R',
        'BHT': 'T',
        'BHZ': 'Z',
    }
    for trc in pyrocko_traces:
        btrc = SeismicDataset.from_pyrocko_trace(trc)
        # adjust channel naming
        btrc.set_channel(channel_mapping[btrc.channel])
        beat_traces.append(btrc)
    # save to seismic_data.pkl
    dump_objects('seismic_data.pkl', outlist=[stations, beat_traces])
    print('####################### Setting up config file for BEAT')
    min_dist = np.min([tr.stats.dist for tr in st_all_unfilt])
    max_dist = np.max([tr.stats.dist for tr in st_all_unfilt])
    eventconfig = BEATconfig(name=origin_time.strftime("%Y%m%d.%H%M%S"),
                             date=origin_time.date.isoformat())
    eventconfig.event = Event(lat=evlat, lon=evlon, time=origin_time.isoformat(),
                              name=origin_time.strftime("%Y%m%d.%H%M%S"),
                              depth=4.0*1000.0, magnitude=mag, magnitude_type='Ml',
                              region="KSMMA", catalog='NRC', duration=0.6,
                              moment_tensor=moment_tensor.MomentTensor(mnn=-0.72,
                                                                       mee=-0.25,
                                                                       mdd=0.80,
                                                                       mne=-0.42,
                                                                       mnd=-0.27,
                                                                       med=0.35))
    eventconfig.project_dir = os.path.join(os.path.abspath('../'),
                                           origin_time.strftime("%Y%m%d.%H%M%S"))
    hyperparameters = OrderedDict()
    hyperparameters['h_any_P_0_Z'] = get_parameter('h_any_P_0_Z', 1, -1, 2)
    hyperparameters['h_any_P_1_R'] = get_parameter('h_any_P_1_R', 1, -1, 2)
    hyperparameters['h_any_P_2_T'] = get_parameter('h_any_P_2_T', 1, -1, 2)
    priors = OrderedDict()
    priors['depth'] = get_parameter('depth', 1, 0.1, 5)
    priors['duration'] = get_parameter('duration', 1, 0, 1)
    priors['east_shift'] = get_parameter('east_shift', 1, -5, 5)
    priors['h'] = get_parameter('h', 1, 0, 1)
    priors['kappa'] = get_parameter('kappa', 1, 0, 2*np.pi)
    priors['magnitude'] = get_parameter('magnitude', 1, 2.0, 3.5)
    priors['north_shift'] = get_parameter('north_shift', 1, -5, 5)
    priors['peak_ratio'] = get_parameter('peak_ratio', 1, 0, 0)
    priors['sigma'] = get_parameter('sigma', 1, -np.pi/2, np.pi/2)
    priors['time'] = get_parameter('time', 1, -3, 3)
    hierarchicals = OrderedDict()
    hierarchicals['time_shift'] = Parameter(name='time_shift',
                                            lower=np.ones(1, dtype=tconfig.floatX) * (-5),
                                            upper=np.ones(1, dtype=tconfig.floatX) * (5),
                                            testvalue=np.ones(1, dtype=tconfig.floatX) * (0))
    eventconfig.problem_config = ProblemConfig(n_sources=1, datatypes=['seismic'],
                                               mode='geometry', source_type='MTSource',
                                               stf_type='Triangular',
                                               hyperparameters=hyperparameters,
                                               priors=priors, hierarchicals=hierarchicals)
    eventconfig.problem_config.set_decimation_factor()
    eventconfig.seismic_config = SeismicConfig(station_corrections=True, wavenames=['any_P'],
                                               noise_estimator=SeismicNoiseAnalyserConfig(structure='variance', pre_arrival_time=10),
                                               waveforms=[WaveformFitConfig(include=True, preprocess_data=True,
                                                                            name='any_P', quantity='velocity',
                                                                            blacklist=[],
                                                                            filterer=[Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True),
                                                                                      BandstopFilter(lower_corner=0.12, upper_corner=0.25, order=3)] if noise_filter else [Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True)],
                                                                            arrival_taper=ArrivalTaper(a=np.round(min_dist/3.3-10.0), b=np.round(min_dist/3.3),
                                                                                                       c=np.round(max_dist/3.3-10), d=np.round(max_dist/3.3)),
                                                                            channels=['Z'], distances=(0.0, r), interpolation='multilinear'),
                                                          WaveformFitConfig(include=True, preprocess_data=True, name='any_P', quantity='velocity',
                                                                            blacklist=[], filterer=[Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True),
                                                                                                    BandstopFilter(lower_corner=0.12, upper_corner=0.25, order=3)] if noise_filter else [Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True)],
                                                                            arrival_taper=ArrivalTaper(a=np.round(min_dist/3.3-10.0), b=np.round(min_dist/3.3),
                                                                                                       c=np.round(max_dist/3.3-10), d=np.round(max_dist/3.3)),
                                                                            channels=['R'], distances=(0.0, r), interpolation='multilinear'),
                                                          WaveformFitConfig(include=True, preprocess_data=True, name='any_P', quantity='velocity',
                                                                            blacklist=[], filterer=[Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True),
                                                                                                    BandstopFilter(lower_corner=0.12, upper_corner=0.25, order=3)] if noise_filter else [Filter(lower_corner=f1, upper_corner=f2, order=3, stepwise=True)],
                                                                            arrival_taper=ArrivalTaper(a=np.round(min_dist/3.3-10.0), b=np.round(min_dist/3.3),
                                                                                                       c=np.round(max_dist/3.3-10), d=np.round(max_dist/3.3)),
                                                                            channels=['T'], distances=(0.0, r), interpolation='multilinear')])
    eventconfig.seismic_config.gf_config.reference_location =
    ReferenceLocation(lat=evlat,
                      lon=evlon, depth=evdep,
                      station='KSMMMA')
    eventconfig.seismic_config.gf_config.custom_velocity_model = \
        load_model('/home/mahdi/project/BEAT_automation/SLU.nd').\
        extract(depth_max=40. * 1000.0)
    eventconfig.seismic_config.gf_config.store_superdir = './{}'.format(origin_time.strftime("%Y%m%d.%H%M%S"))
    eventconfig.seismic_config.gf_config.use_crust2 = False
    eventconfig.seismic_config.gf_config.replace_water = False
    eventconfig.seismic_config.gf_config.nworkers = 20
    eventconfig.seismic_config.gf_config.earth_model_name = 'local'
    eventconfig.seismic_config.gf_config.source_depth_min = 0.1
    eventconfig.seismic_config.gf_config.source_depth_max = 5.0
    eventconfig.seismic_config.gf_config.source_depth_spacing = 0.5
    eventconfig.seismic_config.gf_config.source_distance_radius = r * 111.19+30
    eventconfig.seismic_config.gf_config.source_distance_spacing = 0.5
    eventconfig.seismic_config.gf_config.code = 'qseis'
    eventconfig.seismic_config.gf_config.sample_rate = 5.0
    sampling_params = SMCConfig(n_jobs=20,
                                rm_flag=True,
                                n_steps=300,
                                n_chains=2000,
                                proposal_dist='MultivariateCauchy')
    eventconfig.sampler_config = SamplerConfig(name='SMC',
                                               backend='bin',
                                               progressbar=False,
                                               buffer_size=1000,
                                               buffer_thinning=10,
                                               parameters=sampling_params)
    eventconfig.hyper_sampler_config = SamplerConfig(name='Metropolis')
    # eventconfig.update_hypers()
    # eventconfig.problem_config.validate_priors()
    eventconfig.regularize()
    eventconfig.validate()
    util.ensuredir(eventconfig.project_dir)
    dump_config(eventconfig)


main(obspy.UTCDateTime(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
     float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
     float(sys.argv[7]), 2, 5.0)
# main(origin_time, evlat, evlon, evdep, magnit, f1, f2, r, samp_rate)
