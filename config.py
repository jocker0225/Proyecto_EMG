"""
config.py
=========
Configuración global de la aplicación de análisis EMG.

Este módulo centraliza todas las constantes del proyecto (rutas, frecuencia
de muestreo, mapeos de participantes/condiciones a nombres de archivo,
parámetros por defecto de los filtros y umbrales de asimetría) para que
el resto de módulos NO dependan de valores mágicos dispersos en el código.

No se definen variables globales mutables: todo son constantes (mayúsculas)
o estructuras inmutables usadas solo para lectura.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# BASE_DIR apunta a la carpeta donde vive este archivo (raíz del proyecto),
# así la app funciona sin importar desde qué directorio se ejecute Streamlit.
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "Datos"

# ---------------------------------------------------------------------------
# Parámetros de adquisición (BITalino + OpenSignals)
# ---------------------------------------------------------------------------
FS_HZ: int = 1000          # Frecuencia de muestreo en Hz
HEADER_LINES: int = 3      # Líneas de cabecera a ignorar en los .txt

# Índices de columnas (0-indexado) del archivo OpenSignals:
# nSeq  I1  I2  O1  O2  A1  A2
COL_NSEQ = 0
COL_A1 = 5   # Trapecio Derecho
COL_A2 = 6   # Trapecio Izquierdo

# ---------------------------------------------------------------------------
# Conversión de unidades: cuentas ADC crudas -> milivoltios (mV)
# ---------------------------------------------------------------------------
# Fórmula de conversión estándar para el sensor EMG de BITalino (documentada
# por PLUX / OpenSignals):
#
#     EMG(V) = ((ADC / 2^n - 1/2) * VCC) / GAIN
#
# donde:
#   n     = resolución del ADC en bits
#   VCC   = voltaje de referencia de la placa
#   GAIN  = ganancia del sensor EMG
#
# El resultado se multiplica por 1000 para expresarlo en milivoltios.
ADC_N_BITS: int = 10        # Resolución del ADC de BITalino (2^10 = 1024)
ADC_VCC: float = 3.3        # Voltaje de referencia (V)
EMG_GAIN: float = 1009.0    # Ganancia del sensor EMG de BITalino
UNIDAD_SENAL: str = "mV"    # Unidad en la que se expresan las señales EMG

# ---------------------------------------------------------------------------
# Parámetros por defecto del análisis espectral
# ---------------------------------------------------------------------------
WELCH_NPERSEG_DEFAULT: int = 1024   # Longitud de segmento para el método de Welch (PSD)
STFT_NPERSEG_DEFAULT: int = 256     # Longitud de ventana para el espectrograma (STFT)
STFT_NOVERLAP_DEFAULT: int = 128    # Traslape entre ventanas del espectrograma

# ---------------------------------------------------------------------------
# Participantes y condiciones experimentales
# ---------------------------------------------------------------------------
# Nombre visible (UI) -> sufijo usado en el nombre de archivo
PARTICIPANTES: dict[str, str] = {
    "Ale": "ale",
    "Andre": "andre",
    "Sandro": "sandro",
    "Vallejo": "vallejo",
    "Renzo": "renzo",
}

# Nombre visible (UI) -> prefijo usado en el nombre de archivo
CONDICIONES: dict[str, str] = {
    "Basal": "basal_trapecio",
    "Una Asa": "una_asa_trapecio",
    "Doble Asa": "doble_asa_trapecio",
}

# Nombres de los canales EMG usados en toda la interfaz
CANAL_DERECHO = "Trapecio Derecho (A1)"
CANAL_IZQUIERDO = "Trapecio Izquierdo (A2)"

# ---------------------------------------------------------------------------
# Parámetros por defecto de los filtros digitales
# ---------------------------------------------------------------------------
BANDPASS_LOW_DEFAULT: float = 20.0     # Hz - límite inferior típico en EMG de superficie
BANDPASS_HIGH_DEFAULT: float = 450.0   # Hz - límite superior típico en EMG de superficie
BANDPASS_ORDER_DEFAULT: int = 4
NOTCH_FREQ: float = 60.0               # Hz - interferencia de línea eléctrica
NOTCH_Q_DEFAULT: float = 30.0          # Factor de calidad del filtro notch

# ---------------------------------------------------------------------------
# Umbrales para la clasificación del índice de asimetría bilateral (%)
# ---------------------------------------------------------------------------
ASIMETRIA_SIMETRICO_MAX = 10.0
ASIMETRIA_LEVE_MAX = 20.0
ASIMETRIA_MODERADA_MAX = 40.0
# Por encima de ASIMETRIA_MODERADA_MAX se considera "Asimetría severa"

# ---------------------------------------------------------------------------
# Metadatos de la aplicación
# ---------------------------------------------------------------------------
APP_TITLE = "Análisis EMG - Estabilizadores del Tronco"
APP_SUBTITLE = (
    "Activación y asimetría bilateral de los músculos estabilizadores del "
    "tronco durante una simulación de transporte público"
)
