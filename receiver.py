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

import wcslib as wcs

# TODO: Add relevant parameters to parameters.py
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
    yr = sd.rec(int(T/dt), samplerate=1/dt, channels=1, blocking=True)
    yr = yr[:, 0]           # Remove second channel

    # ============ TILLAGD KOD ============

    #Bandpass runt 1725–1875 Hz
    fl = wcs._channels[0, channel_id] # 1725 Hz
    fu = wcs._channels[1, channel_id] # 1875 Hz
    nyq = fs / 2.0 # Nyquist-frekvens
    b_bp, a_bp = signal.butter(4, [fl/nyq, fu/nyq], btype="band")
    y_bp = signal.lfilter(b_bp, a_bp, yr)

    #IQ-demodulation
    n = np.arange(len(y_bp))
    t = n * dt
    cos_c = np.cos(2 * np.pi * fc * t)
    sin_c = np.sin(2 * np.pi * fc * t)

    yI_d = 2.0 * y_bp * cos_c
    yQ_d = 2.0 * y_bp * sin_c

    # Vi får skräp +- 3600 Hz, tas bort med lowpass filter 
    #Lowpass till baseband
    f_lp = 100.0  # Hz, > bit rate (25 Hz) men << fc 
    #TODO: öka? 
    b_lp, a_lp = signal.butter(4, f_lp/nyq, btype="low")
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)

    #Komplex baseband
    yb = yI_b + 1j * yQ_b

    #======================================

   
    br = wcs.decode_baseband_signal(yb, Tb, fs)
    data_rx = wcs.decode_string(br)
    print(f'Received: {data_rx} (no of bits: {len(br)}).')


if __name__ == "__main__":    
    main()
