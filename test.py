#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for local testing of transmitter and receiver without actual audio transmission
"""

import numpy as np
from scipy import signal

import wcslib as wcs
import filters
from parameters import Tb, Ac, dt, fc, fs, channel_id

def test_transmission(message="Hello World!", binary=False, SNR=20.0, dmax=5.0):
    """
    Test the complete transmission and reception chain locally
    
    Parameters
    ----------
    message : str
        Message to transmit (text or binary string)
    binary : bool
        Whether message is binary (default: False)
    SNR : float
        Signal-to-noise ratio in dB (default: 20.0)
    dmax : float
        Maximum distance in meters (default: 5.0)
    """
    
    print("="*60)
    print("LOCAL TRANSMISSION TEST")
    print("="*60)
    
    # === TRANSMITTER SIDE ===
    print("\n[TRANSMITTER]")
    
    # Convert message to bits
    if binary:
        bs = np.array([bit for bit in map(int, message)])
    else:
        bs = wcs.encode_string(message)
    
    print(f'Sending: {message}')
    print(f'Number of bits: {len(bs)}')
    print(f'Message duration: {np.round(len(bs)*Tb, 1)} s')
    
    # Encode baseband signal
    xb = wcs.encode_baseband_signal(bs, Tb, fs)
    n = np.arange(len(xb))
    t = n * dt
    carrier = Ac * np.sin(2 * np.pi * fc * t)
    xt = xb * carrier
    
    # === CHANNEL SIMULATION ===
    print("\n[CHANNEL]")
    print(f'Simulating channel with SNR={SNR} dB, distance={dmax} m')
    yr = wcs.simulate_channel(xt, fs, channel_id, SNR=SNR, dmax=dmax)
    
    # === RECEIVER SIDE ===
    print("\n[RECEIVER]")
    
    # Bandpass filter
    b_bp, a_bp = filters.design_bandpass_filter(fs)
    y_bp = signal.lfilter(b_bp, a_bp, yr)
    
    # IQ-demodulation
    n = np.arange(len(y_bp))
    t = n * dt
    cos_c = np.cos(2 * np.pi * fc * t)
    sin_c = np.sin(2 * np.pi * fc * t)
    
    yI_d = 2.0 * y_bp * cos_c
    yQ_d = 2.0 * y_bp * sin_c
    
    # Lowpass filter
    b_lp, a_lp = filters.design_lowpass_filter(fs)
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)
    
    # Complex baseband signal
    yb = yI_b + 1j*yQ_b
    
    # Decode
    br = wcs.decode_baseband_signal(yb, Tb, fs)
    data_rx = wcs.decode_string(br)
    
    print(f'Received: {data_rx}')
    print(f'Number of bits: {len(br)}')
    
    # === RESULTS ===
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f'Original:  {message}')
    print(f'Received:  {data_rx}')
    print(f'Match:     {message == data_rx}')
    print(f'Bit errors: {np.sum(bs != br[:len(bs)])} / {len(bs)}')
    print("="*60)
    
    return message == data_rx

if __name__ == "__main__":
    
    print("\n\n### TEST 3: Longer Message ###\n")
    test_transmission("This is a longer message for testing.", SNR=20.0, dmax=10.0)
