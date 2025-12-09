#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receiver template for the wireless communication system project in Signals and
transforms

2022-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import argparse
import numpy as np
from scipy import signal
import sounddevice as sd
import matplotlib.pyplot as plt
import os

import wcslib as wcs

from parameters import Tb, dt, fc, channel_id, fs

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
    ax1.set_title(f'Bandpass Filter Magnitude (Order={n_bp})')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlim(1400, 2200)
    ax1.set_ylim(-80, 5)
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
    ax1.set_xlim(0, 500)
    ax1.set_ylim(-100, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Phase
    ax2.plot(w_lp, np.angle(h_lp), 'b', linewidth=2)
    ax2.set_title(f'Lowpass Filter Phase (Order={n_lp})')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_xlim(0, 500)
    ax2.grid(True, alpha=0.3)
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        prog='receiver',
        description='Acoustic wireless communication system -- receiver.'
    )
    parser.add_argument(
        '-d',
        '--duration',
        help='receiver recording duration',
        type=float,
        default=10
    )
    args = parser.parse_args()

    # Set parameters
    T = args.duration

    # Receive signal
    print(f'Receiving for {T} s.')
    yr = sd.rec(int(T/dt), samplerate=1/dt, channels=1, blocking=True)
    yr = yr[:, 0]           # Remove second channel

    # ============ TILLAGD KOD ============
    
    # Bandpass filter: 1725–1875 Hz (using iirdesign for optimal order)
    nyq = fs / 2.0
    Wp_bp = [1725/nyq, 1875/nyq]  # Passband edges
    Ws_bp = [1650/nyq, 1950/nyq]  # Stopband edges
    Rp_bp = 1.0   # Max passband ripple (dB)
    Rs_bp = 20.0  # Min stopband attenuation (dB)
    
    b_bp, a_bp = signal.iirdesign(Wp_bp, Ws_bp, Rp_bp, Rs_bp, ftype='cheby1')
    y_bp = signal.lfilter(b_bp, a_bp, yr)
    n_bp = len(a_bp) - 1  # Actual bandpass order for plotting

    #IQ-demodulation
    n = np.arange(len(y_bp))
    t = n * dt
    cos_c = np.cos(2 * np.pi * fc * t)
    sin_c = np.sin(2 * np.pi * fc * t)

    yI_d = 2.0 * y_bp * cos_c
    yQ_d = 2.0 * y_bp * sin_c

    # Lowpass filter: remove 2*fc components (using iirdesign for optimal order)
    Wp_lp = 150.0 / nyq   # Passband edge
    Ws_lp = 3000.0 / nyq  # Stopband edge (to remove 2*fc = 3600 Hz)
    Rp_lp = 1.0   # Max passband ripple (dB)
    Rs_lp = 30.0  # Min stopband attenuation (dB)
    
    b_lp, a_lp = signal.iirdesign(Wp_lp, Ws_lp, Rp_lp, Rs_lp, ftype='butter')
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)
    n_lp = len(a_lp) - 1  # Actual lowpass order for plotting
    f_lp = 150.0  # For plotting reference

    # Complex baseband signal
    yb = yI_b + 1j*yQ_b

   
    br = wcs.decode_baseband_signal(yb, Tb, fs)
    data_rx = wcs.decode_string(br)
    print(f'Received: {data_rx} (no of bits: {len(br)}).')

    # Generate Bode plots
    os.makedirs('filters', exist_ok=True)
    
    fig_bp = plot_bandpass_filter(b_bp, a_bp, n_bp, fs)
    fig_bp.savefig('filters/bandpass_bode.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bp)
    
    fig_lp = plot_lowpass_filter(b_lp, a_lp, n_lp, f_lp, fs)
    fig_lp.savefig('filters/lowpass_bode.png', dpi=150, bbox_inches='tight')
    plt.close(fig_lp)
    
    print("Bode plots saved to filters/bandpass_bode.png and filters/lowpass_bode.png")





if __name__ == "__main__":    
    main()
