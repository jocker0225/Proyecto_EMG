"""
utils/procesamiento.py
=======================
Capa de orquestación entre la lectura de la señal (utils.lectura) y los
filtros (utils.filtros). Define la estructura de configuración de
procesamiento seleccionada por el usuario en la interfaz y aplica dicho
procesamiento a ambos canales EMG de forma consistente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    BANDPASS_HIGH_DEFAULT,
    BANDPASS_LOW_DEFAULT,
    BANDPASS_ORDER_DEFAULT,
    NOTCH_FREQ,
    NOTCH_Q_DEFAULT,
)
from utils.filtros import aplicar_pipeline
from utils.lectura import SenalEMG


@dataclass(frozen=True)
class ConfiguracionProcesamiento:
    """Agrupa todas las opciones de procesamiento seleccionables desde la
    barra lateral de la aplicación.
    """

    usar_rectificacion: bool = False
    usar_pasa_banda: bool = False
    usar_notch: bool = False
    f_baja: float = BANDPASS_LOW_DEFAULT
    f_alta: float = BANDPASS_HIGH_DEFAULT
    orden_pasa_banda: int = BANDPASS_ORDER_DEFAULT
    f_notch: float = NOTCH_FREQ
    q_notch: float = NOTCH_Q_DEFAULT


@dataclass(frozen=True)
class SenalProcesada:
    """Resultado de aplicar una configuración de procesamiento a una
    :class:`~utils.lectura.SenalEMG`.
    """

    tiempo: np.ndarray
    canal_derecho: np.ndarray
    canal_izquierdo: np.ndarray
    fs: int


def procesar_senal(
    senal: SenalEMG, config: ConfiguracionProcesamiento
) -> SenalProcesada:
    """Aplica la configuración de procesamiento a ambos canales de una
    señal EMG cargada y devuelve un nuevo objeto con las señales
    resultantes, dejando la señal original (``senal``) intacta.
    """
    derecho_procesado = aplicar_pipeline(
        senal.canal_derecho,
        senal.fs,
        usar_rectificacion=config.usar_rectificacion,
        usar_pasa_banda=config.usar_pasa_banda,
        usar_notch=config.usar_notch,
        f_baja=config.f_baja,
        f_alta=config.f_alta,
        orden_pasa_banda=config.orden_pasa_banda,
        f_notch=config.f_notch,
        q_notch=config.q_notch,
    )

    izquierdo_procesado = aplicar_pipeline(
        senal.canal_izquierdo,
        senal.fs,
        usar_rectificacion=config.usar_rectificacion,
        usar_pasa_banda=config.usar_pasa_banda,
        usar_notch=config.usar_notch,
        f_baja=config.f_baja,
        f_alta=config.f_alta,
        orden_pasa_banda=config.orden_pasa_banda,
        f_notch=config.f_notch,
        q_notch=config.q_notch,
    )

    return SenalProcesada(
        tiempo=senal.tiempo,
        canal_derecho=derecho_procesado,
        canal_izquierdo=izquierdo_procesado,
        fs=senal.fs,
    )


def recortar_por_intervalo(
    senal: SenalProcesada, t_inicio: float, t_fin: float
) -> SenalProcesada:
    """Recorta una señal procesada a un intervalo temporal [t_inicio, t_fin]
    en segundos. Se utiliza para el análisis de una ventana seleccionada
    interactivamente por el usuario.
    """
    fs = senal.fs
    idx_inicio = max(int(round(t_inicio * fs)), 0)
    idx_fin = min(int(round(t_fin * fs)), len(senal.tiempo))
    idx_fin = max(idx_fin, idx_inicio + 1)  # garantizar al menos una muestra

    return SenalProcesada(
        tiempo=senal.tiempo[idx_inicio:idx_fin],
        canal_derecho=senal.canal_derecho[idx_inicio:idx_fin],
        canal_izquierdo=senal.canal_izquierdo[idx_inicio:idx_fin],
        fs=fs,
    )


def normalizar_respecto_a_referencia(
    senal: SenalProcesada, rms_ref_derecho: float, rms_ref_izquierdo: float
) -> SenalProcesada:
    """Normaliza ambos canales de una señal procesada respecto al RMS de
    una señal de referencia (típicamente la condición 'Basal' del mismo
    participante, procesada con la misma configuración de filtros).

    El resultado se expresa como porcentaje de la actividad basal
    (0 % = sin actividad, 100 % = igual a la actividad basal de referencia).
    Si el RMS de referencia es cero, se devuelve la señal sin modificar
    para evitar una división por cero.
    """
    derecho_normalizado = (
        senal.canal_derecho / rms_ref_derecho * 100.0
        if rms_ref_derecho > 0
        else senal.canal_derecho
    )
    izquierdo_normalizado = (
        senal.canal_izquierdo / rms_ref_izquierdo * 100.0
        if rms_ref_izquierdo > 0
        else senal.canal_izquierdo
    )

    return SenalProcesada(
        tiempo=senal.tiempo,
        canal_derecho=derecho_normalizado,
        canal_izquierdo=izquierdo_normalizado,
        fs=senal.fs,
    )
