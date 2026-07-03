"""
utils/espectral.py
===================
Análisis espectral de señales EMG: Transformada Rápida de Fourier (FFT),
Densidad Espectral de Potencia mediante el método de Welch (PSD) y
espectrograma mediante Transformada de Fourier de Tiempo Corto (STFT).

Este módulo se agrega como extensión de la arquitectura existente
(utils/) para mantener una única responsabilidad por archivo: el análisis
en el dominio de la frecuencia se mantiene separado del preprocesamiento
en el dominio del tiempo (utils/filtros.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from config import (
    STFT_NOVERLAP_DEFAULT,
    STFT_NPERSEG_DEFAULT,
    WELCH_NPERSEG_DEFAULT,
)


@dataclass(frozen=True)
class ResultadoFFT:
    """Resultado del cálculo de la FFT de un canal."""

    frecuencias: np.ndarray   # Hz
    magnitudes: np.ndarray    # mV (amplitud, espectro unilateral)


@dataclass(frozen=True)
class ResultadoPSD:
    """Resultado del cálculo de la Densidad Espectral de Potencia (Welch)."""

    frecuencias: np.ndarray   # Hz
    potencia: np.ndarray      # mV^2/Hz


@dataclass(frozen=True)
class ResultadoEspectrograma:
    """Resultado del cálculo del espectrograma (STFT)."""

    frecuencias: np.ndarray   # Hz
    tiempos: np.ndarray       # s
    magnitud: np.ndarray      # mV, matriz [frecuencias x tiempos]


def calcular_fft(senal: np.ndarray, fs: int) -> ResultadoFFT:
    """Calcula el espectro de amplitud unilateral de la señal mediante la
    Transformada Rápida de Fourier (FFT).

    Se normaliza por el número de muestras para obtener una magnitud
    comparable independientemente de la duración del registro, y se
    multiplica por 2 (excepto la componente DC) para compensar el
    descarte de la mitad negativa del espectro.
    """
    n = len(senal)
    if n == 0:
        return ResultadoFFT(frecuencias=np.array([]), magnitudes=np.array([]))

    espectro = np.fft.rfft(senal)
    frecuencias = np.fft.rfftfreq(n, d=1.0 / fs)

    magnitudes = np.abs(espectro) / n
    if len(magnitudes) > 1:
        magnitudes[1:] *= 2  # compensar espectro unilateral (excepto DC)

    return ResultadoFFT(frecuencias=frecuencias, magnitudes=magnitudes)


def calcular_psd_welch(
    senal: np.ndarray, fs: int, nperseg: int = WELCH_NPERSEG_DEFAULT
) -> ResultadoPSD:
    """Calcula la Densidad Espectral de Potencia (PSD) mediante el método
    de Welch (promediado de periodogramas con ventanas solapadas), que
    reduce la varianza respecto a un periodograma simple.
    """
    nperseg_efectivo = min(nperseg, len(senal))
    if nperseg_efectivo < 2:
        return ResultadoPSD(frecuencias=np.array([]), potencia=np.array([]))

    frecuencias, potencia = signal.welch(senal, fs=fs, nperseg=nperseg_efectivo)
    return ResultadoPSD(frecuencias=frecuencias, potencia=potencia)


def calcular_espectrograma(
    senal: np.ndarray,
    fs: int,
    nperseg: int = STFT_NPERSEG_DEFAULT,
    noverlap: int = STFT_NOVERLAP_DEFAULT,
) -> ResultadoEspectrograma:
    """Calcula el espectrograma de la señal mediante la Transformada de
    Fourier de Tiempo Corto (STFT), útil para observar cómo cambia el
    contenido en frecuencia a lo largo del registro.
    """
    nperseg_efectivo = min(nperseg, len(senal))
    noverlap_efectivo = min(noverlap, max(nperseg_efectivo - 1, 0))

    if nperseg_efectivo < 2:
        return ResultadoEspectrograma(
            frecuencias=np.array([]), tiempos=np.array([]), magnitud=np.array([[]])
        )

    frecuencias, tiempos, sxx = signal.spectrogram(
        senal, fs=fs, nperseg=nperseg_efectivo, noverlap=noverlap_efectivo
    )
    return ResultadoEspectrograma(frecuencias=frecuencias, tiempos=tiempos, magnitud=sxx)
