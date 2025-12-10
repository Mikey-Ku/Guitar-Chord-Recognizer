"""
Manual implementation of Discrete Fourier Transform
"""

from math import cos, sin, pi, sqrt


class ManualDFT:
    """Manual implementation of Discrete Fourier Transform"""
    
    @staticmethod
    def dft(signal):
        """
        Compute DFT 
        Returns complex frequency components
        """
        N = len(signal)
        frequencies = []
        
        for k in range(N // 2):
            real_part = 0.0
            imag_part = 0.0
            
            for n in range(N):
                angle = 2 * pi * k * n / N
                real_part += signal[n] * cos(angle)
                imag_part -= signal[n] * sin(angle)
            
            frequencies.append((real_part, imag_part))
        
        return frequencies
    
    @staticmethod
    def get_magnitude_spectrum(frequencies):
        """Calculate magnitude spectrum from complex frequencies"""
        magnitudes = []
        for real, imag in frequencies:
            magnitude = sqrt(real * real + imag * imag)
            magnitudes.append(magnitude)
        return magnitudes
