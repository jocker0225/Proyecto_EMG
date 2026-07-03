"""
utils/estadisticas.py
======================
Cálculo de parámetros estadísticos y de amplitud sobre señales EMG, y del
índice de asimetría bilateral basado en el RMS de ambos canales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    ASIMETRIA_LEVE_MAX,
    ASIMETRIA_MODERADA_MAX,
    ASIMETRIA_SIMETRICO_MAX,
)


@dataclass(frozen=True)
class EstadisticasCanal:
    """Conjunto de estadísticos descriptivos calculados sobre un canal EMG.

    Todos los valores (excepto varianza y energía, que están en mV^2)
    se expresan en milivoltios (mV), unidad de la señal ya convertida.
    """

    media: float
    mediana: float
    desviacion_estandar: float
    maximo: float
    minimo: float
    rms: float
    mav: float
    varianza: float
    energia: float


def calcular_estadisticas(senal: np.ndarray) -> EstadisticasCanal:
    """Calcula el conjunto completo de estadísticos descriptivos para una
    señal EMG de un solo canal (valores en mV).

    - RMS: raíz del valor cuadrático medio, proporcional a la energía de
      activación muscular.
    - MAV: valor absoluto medio, estimador robusto de la amplitud.
    - Energía: suma de los valores al cuadrado de la señal (mV^2).
    """
    return EstadisticasCanal(
        media=float(np.mean(senal)),
        mediana=float(np.median(senal)),
        desviacion_estandar=float(np.std(senal)),
        maximo=float(np.max(senal)),
        minimo=float(np.min(senal)),
        rms=float(np.sqrt(np.mean(np.square(senal)))),
        mav=float(np.mean(np.abs(senal))),
        varianza=float(np.var(senal)),
        energia=float(np.sum(np.square(senal))),
    )


@dataclass(frozen=True)
class ResultadoAsimetria:
    """Resultado del cálculo de asimetría bilateral entre ambos canales."""

    indice_porcentaje: float
    clasificacion: str


def calcular_indice_asimetria(rms_derecho: float, rms_izquierdo: float) -> ResultadoAsimetria:
    """Calcula el índice de asimetría bilateral (%) a partir del RMS de
    ambos canales, usando la fórmula simétrica habitual en literatura de
    biomecánica:

        Asimetría (%) = |RMS_d - RMS_i| / ((RMS_d + RMS_i) / 2) * 100

    Esta fórmula normaliza la diferencia respecto al promedio de ambos
    lados, evitando sesgos hacia el canal de mayor magnitud.
    """
    promedio = (rms_derecho + rms_izquierdo) / 2.0

    if promedio == 0:
        indice = 0.0
    else:
        indice = abs(rms_derecho - rms_izquierdo) / promedio * 100.0

    clasificacion = _clasificar_asimetria(indice)

    return ResultadoAsimetria(indice_porcentaje=indice, clasificacion=clasificacion)


def _clasificar_asimetria(indice_porcentaje: float) -> str:
    """Clasifica el índice de asimetría según rangos razonables definidos
    en config.py.
    """
    if indice_porcentaje <= ASIMETRIA_SIMETRICO_MAX:
        return "Simetría adecuada"
    if indice_porcentaje <= ASIMETRIA_LEVE_MAX:
        return "Asimetría leve"
    if indice_porcentaje <= ASIMETRIA_MODERADA_MAX:
        return "Asimetría moderada"
    return "Asimetría severa"
