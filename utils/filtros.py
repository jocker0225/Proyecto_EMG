"""
utils/filtros.py
=================
Filtros digitales aplicables a la señal EMG cruda:

- Rectificación de onda completa (valor absoluto).
- Filtro pasa banda Butterworth (configurable).
- Filtro notch (rechazo de banda angosta) para eliminar la interferencia
  de la red eléctrica (60 Hz).

Todas las funciones son puras (no dependen ni modifican estado externo) y
reciben la frecuencia de muestreo como parámetro explícito.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from config import (
    BANDPASS_HIGH_DEFAULT,
    BANDPASS_LOW_DEFAULT,
    BANDPASS_ORDER_DEFAULT,
    NOTCH_FREQ,
    NOTCH_Q_DEFAULT,
)


def rectificar_onda_completa(senal: np.ndarray) -> np.ndarray:
    """Aplica rectificación de onda completa (valor absoluto) a la señal.

    Es un paso habitual en el preprocesamiento EMG antes de calcular
    envolventes o parámetros de amplitud.
    """
    return np.abs(senal)


def filtro_pasa_banda(
    senal: np.ndarray,
    fs: int,
    f_baja: float = BANDPASS_LOW_DEFAULT,
    f_alta: float = BANDPASS_HIGH_DEFAULT,
    orden: int = BANDPASS_ORDER_DEFAULT,
) -> np.ndarray:
    """Aplica un filtro pasa banda Butterworth de fase cero (filtfilt) a la
    señal EMG.

    Parámetros
    ----------
    senal : np.ndarray
        Señal de entrada.
    fs : int
        Frecuencia de muestreo en Hz.
    f_baja, f_alta : float
        Frecuencias de corte inferior y superior en Hz.
    orden : int
        Orden del filtro Butterworth.
    """
    nyquist = fs / 2.0

    # Se limita f_alta a un valor levemente inferior a Nyquist para evitar
    # errores numéricos si el usuario introduce un valor demasiado alto.
    f_alta_segura = min(f_alta, nyquist * 0.99)
    f_baja_segura = max(f_baja, 0.1)

    sos = signal.butter(
        orden,
        [f_baja_segura, f_alta_segura],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return signal.sosfiltfilt(sos, senal)


def filtro_notch(
    senal: np.ndarray,
    fs: int,
    f_notch: float = NOTCH_FREQ,
    q: float = NOTCH_Q_DEFAULT,
) -> np.ndarray:
    """Aplica un filtro notch (rechaza-banda angosto) centrado en
    ``f_notch`` (por defecto 60 Hz) para eliminar la interferencia de la
    red eléctrica.
    """
    b, a = signal.iirnotch(f_notch, q, fs)
    return signal.filtfilt(b, a, senal)


def aplicar_pipeline(
    senal: np.ndarray,
    fs: int,
    usar_rectificacion: bool = False,
    usar_pasa_banda: bool = False,
    usar_notch: bool = False,
    f_baja: float = BANDPASS_LOW_DEFAULT,
    f_alta: float = BANDPASS_HIGH_DEFAULT,
    orden_pasa_banda: int = BANDPASS_ORDER_DEFAULT,
    f_notch: float = NOTCH_FREQ,
    q_notch: float = NOTCH_Q_DEFAULT,
) -> np.ndarray:
    """Aplica de forma secuencial los filtros seleccionados por el usuario.

    Orden de aplicación (recomendado en procesamiento EMG):
    1. Filtro pasa banda (elimina ruido de baja frecuencia y de alta
       frecuencia fuera del contenido útil del EMG).
    2. Filtro notch (elimina interferencia de línea de 60 Hz).
    3. Rectificación de onda completa (para análisis de amplitud/envolvente).

    Cada paso es opcional e independiente, controlado por los flags
    booleanos correspondientes.
    """
    senal_procesada = senal.copy()

    if usar_pasa_banda:
        senal_procesada = filtro_pasa_banda(
            senal_procesada, fs, f_baja=f_baja, f_alta=f_alta, orden=orden_pasa_banda
        )

    if usar_notch:
        senal_procesada = filtro_notch(senal_procesada, fs, f_notch=f_notch, q=q_notch)

    if usar_rectificacion:
        senal_procesada = rectificar_onda_completa(senal_procesada)

    return senal_procesada
