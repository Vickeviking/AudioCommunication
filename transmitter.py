#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transmitter template for the wireless communication system project in Signals and
transforms

For plain text inputs, run:
$ python3 transmitter.py "Hello World!"

For binary inputs, run:
$ python3 transmitter.py -b 010010000110100100100001

2022-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import argparse
import numpy as np
from scipy import signal
import sounddevice as sd

import wcslib as wcs

from parameters import Tb, Ac, dt, fc, fs



def main():
    parser = argparse.ArgumentParser(
        prog='transmitter',
        description='Acoustic wireless communication system -- transmitter.'
    )
    parser.add_argument(
        '-b',
        '--binary',
        help='message is a binary sequence',
        action='store_true'
    )
    parser.add_argument('message', help='message to transmit', nargs='?')
    args = parser.parse_args()

    if args.message is None:
        args.message = 'Hello World!'

    # Set parameters
    data = args.message

    # Convert string to bit sequence or string bit sequence to numeric bit
    # sequence
    if args.binary:
        bs = np.array([bit for bit in map(int, data)])
    else:
        bs = wcs.encode_string(data)
    
    # Add trailing zeros (3 bytes = 24 bits) to ensure full message decoding
    trailing_zeros = np.zeros(24, dtype=int)
    bs = np.concatenate([bs, trailing_zeros])

    # Transmit signal
    print(f'Sending: {data} (no of bits: {len(bs)}; message duration: {np.round(len(bs)*Tb, 1)} s).')


    # === TILLAGD KOD ====

    # Encode baseband signal
    xb = wcs.encode_baseband_signal(bs, Tb, fs)
    n = np.arange(len(xb))
    t = n * dt
    carrier = Ac * np.sin(2 * np.pi * fc * t)
    xt = xb * carrier

    # ====================


    # Ensure the signal is mono, then play through speakers
    xt = np.stack((xt, np.zeros(xt.shape)), axis=1)
    sd.play(xt, 1/dt, blocking=True)


if __name__ == "__main__":    
    main()
