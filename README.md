# Análisis EMG — Estabilizadores del Tronco

**Activación y asimetría bilateral de los músculos estabilizadores del tronco
durante una simulación de transporte público: estudio piloto mediante
electromiografía de superficie.**

Proyecto universitario — Introducción a Señales Biomédicas.

## Descripción

Aplicación web interactiva desarrollada en **Streamlit** para visualizar y
analizar señales EMG de superficie (trapecio derecho e izquierdo)
adquiridas con **BITalino** a través de **OpenSignals**, durante tres
condiciones experimentales que simulan el uso de transporte público:
Basal, Una Asa y Doble Asa.

## Estructura del proyecto

```
Proyecto_EMG/
├── app.py                  # Aplicación principal Streamlit
├── config.py                # Constantes, rutas y parámetros globales
├── utils/
│   ├── lectura.py           # Localización, lectura y conversión ADC->mV de los archivos .txt
│   ├── procesamiento.py     # Orquestación del pipeline de filtros, recorte y normalización
│   ├── estadisticas.py      # Estadísticos descriptivos y asimetría bilateral
│   ├── espectral.py         # FFT, PSD (Welch) y espectrograma (STFT)
│   ├── graficas.py          # Figuras interactivas con Plotly
│   └── filtros.py           # Filtros digitales (pasa banda, notch, rectificación)
├── Datos/                    # Archivos .txt de OpenSignals (no incluidos)
├── requirements.txt
└── README.md
```

## Formato de datos esperado

Cada archivo `.txt` exportado por OpenSignals debe seguir el patrón:

```
<condicion>_trapecio_<participante>.txt
```

Ejemplos: `basal_trapecio_andre.txt`, `una_asa_trapecio_ale.txt`,
`doble_asa_trapecio_sandro.txt`.

Deben colocarse dentro de la carpeta `Datos/`. Las tres primeras líneas de
cada archivo (cabecera de OpenSignals) se ignoran automáticamente. Las
columnas usadas son A1 (Trapecio Derecho) y A2 (Trapecio Izquierdo),
muestreadas a 1000 Hz.

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador
(por defecto en `http://localhost:8501`).

## Funcionalidades principales

- **Selección de registro**: participante, condición y carga bajo demanda.
- **Conversión automática a milivoltios (mV)**: las cuentas ADC crudas de
  BITalino se convierten a mV mediante la fórmula estándar de PLUX/OpenSignals;
  todas las gráficas, estadísticas y exportaciones usan esta unidad.
- **Selección interactiva de intervalo temporal**: un control deslizante
  permite acotar el análisis a una ventana específica del registro; todas
  las estadísticas, gráficas y el análisis espectral se recalculan sobre
  ese intervalo.
- **Visualización interactiva** (Plotly): canal derecho, canal izquierdo y
  comparación sincronizada (zoom compartido entre ambos canales).
- **Procesamiento configurable**: rectificación de onda completa, filtro
  pasa banda Butterworth (frecuencias y orden ajustables) y filtro notch de
  60 Hz para interferencia de línea.
- **Normalización respecto a Basal**: expresa la señal como porcentaje del
  RMS de la condición Basal del mismo participante (cargada automáticamente).
- **Estadísticas descriptivas** por canal: media, mediana, desviación
  estándar, máximo, mínimo, RMS, MAV, varianza y energía, en mV.
- **Asimetría bilateral**: sección destacada con indicador tipo gauge,
  RMS de ambos canales e interpretación automática (Simetría adecuada /
  Asimetría leve / moderada / severa).
- **Análisis espectral**: FFT, Densidad Espectral de Potencia (método de
  Welch) y espectrograma (STFT), para ambos canales.
- **Exportación a CSV** del resumen de resultados del intervalo analizado.

## Notas técnicas

- El proyecto sigue una arquitectura modular sin variables globales
  mutables: toda la configuración vive en `config.py` como constantes, y el
  estado de la sesión de usuario se maneja mediante `st.session_state` de
  Streamlit.
- Los filtros se implementan con `scipy.signal` utilizando secciones de

**Streamlit.app: https://proyectoemg-theemgineers.streamlit.app/

  segundo orden (`sosfiltfilt`) para el pasa banda, garantizando estabilidad
  numérica y fase cero.
- El análisis espectral (`utils/espectral.py`) usa `numpy.fft`, `scipy.signal.welch`
  y `scipy.signal.spectrogram`.
- La conversión ADC -> mV usa la resolución del ADC, el voltaje de referencia
  y la ganancia del sensor EMG de BITalino, definidas en `config.py`.
