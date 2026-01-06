Tb = 0.04             # s, tid per bit
Ac = 2.0              # carrier amplitud
fc = 1800.0           # Hz, carrier frequens (1725+1875)/2
fs = 36000.0          # Hz, samplingsfrekvens (standard audio sampling rate)
dt = 1.0 / fs         # s, tidssteg
channel_id = 5        # band id, använder 1725–1875 Hz i wcslib.channels
data = "Hello World!" # Data to transmit

