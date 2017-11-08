NRCHID = 1440
NRCELL = 1024
NRPATCH = 160
NRTEMPSENSOR = 160
ROI = 300
PEAFACTOR = 0.1  # [singlePhotonAmplitude/mV]
TRIGGERFREQUENZ = 80.0  # [Hz]
# 12 Bit ADC with 2.0 V range
ADCCOUNTSTOMILIVOLT = 2000.0 / 4096.0  # [mV/adc]
# 16 Bit DAC with 2.5 V range and an input of 50000
DACfactor = 2500/pow(2, 16)*50000  # ca. 1907.35 mV
