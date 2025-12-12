#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter design and plotting functions for the wireless communication system
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def design_bandpass_filter(fs, ftype='cheby1'):
    """
    Design bandpass filter for 1725-1875 Hz range
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    ftype : str
        Filter type (default: 'cheby1' - Chebyshev Type I for flat passband and linear phase)
    
    Returns
    -------
    b : array
        Numerator coefficients
    a : array
        Denominator coefficients
    """
    nyq = fs / 2.0
    Wp_bp = [1725/nyq, 1875/nyq]
    Ws_bp = [1650/nyq, 1950/nyq]
    Rp_bp = 0.5   # Max passband ripple (dB)
    Rs_bp = 40.0  # Min stopband attenuation (dB)
    
    b, a = signal.iirdesign(Wp_bp, Ws_bp, Rp_bp, Rs_bp, ftype=ftype)
    return b, a

def design_lowpass_filter(fs, ftype='butter'):
    """
    Design lowpass filter to remove 2*fc components
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    ftype : str
        Filter type (default: 'butter')
    
    Returns
    -------
    b : array
        Numerator coefficients
    a : array
        Denominator coefficients
    """
    nyq = fs / 2.0
    Wp_lp = 100.0 / nyq
    Ws_lp = 3000.0 / nyq
    Rp_lp = 1.0
    Rs_lp = 40.0
    
    b, a = signal.iirdesign(Wp_lp, Ws_lp, Rp_lp, Rs_lp, ftype=ftype)
    return b, a

def plot_bandpass_filter(b_bp, a_bp, n_bp, fs):
    """Plot Bode diagram for bandpass filter"""
    w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=8192, fs=fs)
    h_bp_mag = np.abs(h_bp)
    h_bp_mag_normalized = h_bp_mag / np.max(h_bp_mag)
    h_bp_mag_db = 20 * np.log10(np.maximum(h_bp_mag_normalized, 1e-10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Magnitude
    ax1.plot(w_bp, h_bp_mag_db, 'b', linewidth=2, label='Filter response')
    ax1.axvline(1725, color='r', linestyle='--', alpha=0.7, label='Passband: 1725-1875 Hz')
    ax1.axvline(1875, color='r', linestyle='--', alpha=0.7)
    ax1.axvline(1650, color='orange', linestyle='--', alpha=0.5, label='Stopband edges')
    ax1.axvline(1950, color='orange', linestyle='--', alpha=0.5)
    ax1.axhline(-60, color='m', linestyle=':', alpha=0.5, label='Rs=-60 dB')
    ax1.set_title(f'Bandpass Filter Magnitude (Order={n_bp})')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlim(1400, 2200)
    ax1.set_ylim(-100, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Phase
    ax2.plot(w_bp, np.angle(h_bp), 'b', linewidth=2)
    ax2.set_title(f'Bandpass Filter Phase (Order={n_bp})')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_xlim(1400, 2200)
    ax2.grid(True, alpha=0.3)
    
    return fig

def plot_lowpass_filter(b_lp, a_lp, n_lp, f_lp, fs):
    """Plot Bode diagram for lowpass filter"""
    w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=8192, fs=fs)
    h_lp_mag = np.abs(h_lp)
    h_lp_mag_normalized = h_lp_mag / np.max(h_lp_mag)
    h_lp_mag_db = 20 * np.log10(np.maximum(h_lp_mag_normalized, 1e-10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Magnitude
    ax1.plot(w_lp, h_lp_mag_db, 'b', linewidth=2, label='Filter response')
    ax1.axvline(f_lp, color='r', linestyle='--', alpha=0.7, label=f'Cutoff: {f_lp} Hz')
    ax1.set_title(f'Lowpass Filter Magnitude (Order={n_lp})')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlim(0, 4000)
    ax1.set_ylim(-100, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Phase
    ax2.plot(w_lp, np.angle(h_lp), 'b', linewidth=2)
    ax2.set_title(f'Lowpass Filter Phase (Order={n_lp})')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_xlim(0, 4000)
    ax2.grid(True, alpha=0.3)
    
    return fig
