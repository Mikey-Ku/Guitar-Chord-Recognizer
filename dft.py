import math

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = (2 * math.pi * k * n) / N
            real_part += signal[n] * math.cos(angle)
            imag_part += signal[n] * math.sin(angle)
        magnitude = math.sqrt(real_part**2 + imag_part**2)
        result.append(magnitude)
    return result