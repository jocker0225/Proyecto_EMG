"""
utils/lectura.py
=================
Funciones encargadas de localizar y leer los archivos crudos exportados por
OpenSignals (BITalino), ignorando la cabecera y extrayendo únicamente los
canales EMG de interés (A1 = Trapecio Derecho, A2 = Trapecio Izquierdo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    ADC_N_BITS,
    ADC_VCC,
    COL_A1,
    COL_A2,
    CONDICIONES,
    DATA_DIR,
    EMG_GAIN,
    FS_HZ,
    HEADER_LINES,
    PARTICIPANTES,
)


class ArchivoNoEncontradoError(FileNotFoundError):
    """Se lanza cuando no existe el archivo esperado para una combinación
    participante/condición."""


def adc_a_mv(
    valores_adc: np.ndarray,
    n_bits: int = ADC_N_BITS,
    vcc: float = ADC_VCC,
    ganancia: float = EMG_GAIN,
) -> np.ndarray:
    """Convierte cuentas ADC crudas del sensor EMG de BITalino a milivoltios.

    Aplica la fórmula estándar de PLUX/OpenSignals:

        EMG(V) = ((ADC / 2^n - 1/2) * VCC) / GAIN

    y expresa el resultado en mV, unidad utilizada en toda la aplicación.
    """
    resolucion = 2 ** n_bits
    voltios = ((valores_adc / resolucion) - 0.5) * vcc / ganancia
    return voltios * 1000.0  # V -> mV


@dataclass(frozen=True)
class SenalEMG:
    """Contenedor inmutable con la señal EMG ya cargada y lista para usar.

    Atributos
    ---------
    tiempo : np.ndarray
        Vector de tiempo en segundos, generado a partir de fs.
    canal_derecho : np.ndarray
        Señal del canal A1 (Trapecio Derecho), ya convertida a milivoltios (mV).
    canal_izquierdo : np.ndarray
        Señal del canal A2 (Trapecio Izquierdo), ya convertida a milivoltios (mV).
    fs : int
        Frecuencia de muestreo en Hz.
    participante : str
        Nombre del participante (tal como se muestra en la UI).
    condicion : str
        Nombre de la condición experimental (tal como se muestra en la UI).
    nombre_archivo : str
        Nombre del archivo fuente, útil para trazabilidad y exportación.
    """

    tiempo: np.ndarray
    canal_derecho: np.ndarray
    canal_izquierdo: np.ndarray
    fs: int
    participante: str
    condicion: str
    nombre_archivo: str

    @property
    def duracion_segundos(self) -> float:
        """Duración total de la señal en segundos."""
        return len(self.canal_derecho) / self.fs

    @property
    def n_muestras(self) -> int:
        """Número total de muestras de la señal."""
        return len(self.canal_derecho)


def construir_nombre_archivo(participante: str, condicion: str) -> str:
    """Construye el nombre de archivo esperado a partir de los nombres
    legibles de participante y condición mostrados en la interfaz.

    Ejemplo: ("Andre", "Basal") -> "basal_trapecio_andre.txt"
    """
    if participante not in PARTICIPANTES:
        raise ValueError(f"Participante desconocido: {participante}")
    if condicion not in CONDICIONES:
        raise ValueError(f"Condición desconocida: {condicion}")

    sufijo_participante = PARTICIPANTES[participante]
    prefijo_condicion = CONDICIONES[condicion]
    return f"{prefijo_condicion}_{sufijo_participante}.txt"


def localizar_archivo(participante: str, condicion: str, data_dir: Path = DATA_DIR) -> Path:
    """Devuelve la ruta completa al archivo correspondiente a la combinación
    participante/condición, verificando que exista.

    Lanza ArchivoNoEncontradoError si el archivo no está presente en
    ``data_dir``.
    """
    nombre_archivo = construir_nombre_archivo(participante, condicion)
    ruta = data_dir / nombre_archivo

    if not ruta.exists():
        raise ArchivoNoEncontradoError(
            f"No se encontró el archivo '{nombre_archivo}' en '{data_dir}'. "
            "Verifica que el archivo exista y que el nombre siga el patrón "
            "esperado (ej: basal_trapecio_andre.txt)."
        )
    return ruta


def leer_senal_emg(
    participante: str,
    condicion: str,
    data_dir: Path = DATA_DIR,
    fs: int = FS_HZ,
) -> SenalEMG:
    """Lee el archivo .txt correspondiente, ignora las líneas de cabecera y
    extrae los canales A1 y A2, devolviendo un objeto :class:`SenalEMG`
    listo para procesar y graficar.
    """
    ruta = localizar_archivo(participante, condicion, data_dir=data_dir)

    # np.loadtxt es suficiente aquí porque los datos están separados por
    # tabulaciones/espacios en columnas homogéneas de números.
    datos = np.loadtxt(ruta, skiprows=HEADER_LINES)

    if datos.ndim == 1:
        # Caso borde: archivo con una sola fila de datos.
        datos = datos.reshape(1, -1)

    canal_derecho = adc_a_mv(datos[:, COL_A1].astype(float))
    canal_izquierdo = adc_a_mv(datos[:, COL_A2].astype(float))

    n_muestras = len(canal_derecho)
    tiempo = np.arange(n_muestras) / fs

    return SenalEMG(
        tiempo=tiempo,
        canal_derecho=canal_derecho,
        canal_izquierdo=canal_izquierdo,
        fs=fs,
        participante=participante,
        condicion=condicion,
        nombre_archivo=ruta.name,
    )


def cargar_senal_basal(
    participante: str, data_dir: Path = DATA_DIR, fs: int = FS_HZ
) -> SenalEMG:
    """Carga la señal de la condición 'Basal' del mismo participante.

    Se utiliza como referencia para la normalización de las condiciones
    'Una Asa' y 'Doble Asa' (normalización respecto a la actividad basal).
    """
    return leer_senal_emg(participante, "Basal", data_dir=data_dir, fs=fs)


def listar_archivos_disponibles(data_dir: Path = DATA_DIR) -> list[str]:
    """Lista los archivos .txt presentes en la carpeta de datos. Útil para
    mostrar diagnósticos en la interfaz cuando falta algún registro."""
    if not data_dir.exists():
        return []
    return sorted(p.name for p in data_dir.glob("*.txt"))
