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
import filters

from parameters import Tb, dt, fc, channel_id, fs

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
    yr = sd.rec(int(T*fs), samplerate=fs, channels=1, blocking=True)
    yr = yr[:, 0]           # Remove second channel

    # ============ TILLAGD KOD ============
    
    # Bandpass filter: 1725–1875 Hz (using iirdesign for optimal order)
    b_bp, a_bp = filters.design_bandpass_filter(fs)
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
    b_lp, a_lp = filters.design_lowpass_filter(fs)
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)
    n_lp = len(a_lp) - 1  # Actual lowpass order for plotting
    f_lp = 100.0  # For plotting reference

    # Complex baseband signal
    yb = yI_b + 1j*yQ_b

    # Check signal strength before decoding
    signal_amplitude = np.max(np.abs(yb))
    noise_floor = np.median(np.abs(yb))
    amplitude_threshold = 0.003  # Lowered to catch weaker signals
    
    print(f'Signal amplitude: {signal_amplitude:.4f}, Noise floor: {noise_floor:.4f}')
    
    if signal_amplitude < amplitude_threshold:
        print(f'No signal detected (amplitude {signal_amplitude:.4f} < threshold {amplitude_threshold}).')
        br = np.array([], dtype=int)
        data_rx = ""
    else:
        print(f'Signal detected! Decoding... (amplitude {signal_amplitude:.4f})')
        br = wcs.decode_baseband_signal(yb, Tb, fs)
        data_rx = wcs.decode_string(br)
    
    print(f'Received: {data_rx} (no of bits: {len(br)}).')

    # Generate Bode plots
    os.makedirs('filters', exist_ok=True)
    
    fig_bp = filters.plot_bandpass_filter(b_bp, a_bp, n_bp, fs)
    fig_bp.savefig('filters/bandpass_bode.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bp)
    
    fig_lp = filters.plot_lowpass_filter(b_lp, a_lp, n_lp, f_lp, fs)
    fig_lp.savefig('filters/lowpass_bode.png', dpi=150, bbox_inches='tight')
    plt.close(fig_lp)
    
    print("Bode plots saved to filters/bandpass_bode.png and filters/lowpass_bode.png")


if __name__ == "__main__":    
    main()
