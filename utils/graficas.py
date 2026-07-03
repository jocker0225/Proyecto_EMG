"""
utils/graficas.py
==================
Generación de figuras interactivas (Plotly) para visualizar las señales
EMG: gráfico individual por canal, comparación sincronizada, indicador de
asimetría bilateral y análisis espectral (FFT, PSD, espectrograma).

Paleta sobria orientada a un contexto de investigación biomédica: sin
colores llamativos ni elementos decorativos.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    ASIMETRIA_LEVE_MAX,
    ASIMETRIA_MODERADA_MAX,
    ASIMETRIA_SIMETRICO_MAX,
    CANAL_DERECHO,
    CANAL_IZQUIERDO,
    UNIDAD_SENAL,
)

# Paleta de colores sobria, consistente en toda la aplicación.
COLOR_DERECHO = "#1F4E79"     # azul acero oscuro
COLOR_IZQUIERDO = "#8C1D18"   # granate oscuro
COLOR_NEUTRO = "#4B5563"      # gris pizarra

_LAYOUT_BASE = dict(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=45, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Georgia, 'Times New Roman', serif", size=13),
)


def figura_canal_individual(
    tiempo: np.ndarray, senal: np.ndarray, nombre_canal: str, color: str,
    unidad: str = UNIDAD_SENAL,
) -> go.Figure:
    """Crea una figura Plotly para un único canal EMG con zoom interactivo."""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=tiempo,
            y=senal,
            mode="lines",
            name=nombre_canal,
            line=dict(color=color, width=1),
        )
    )
    fig.update_layout(
        title=f"Señal EMG - {nombre_canal}",
        xaxis_title="Tiempo (s)",
        yaxis_title=f"Amplitud ({unidad})",
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def figura_trapecio_derecho(tiempo: np.ndarray, senal: np.ndarray, unidad: str = UNIDAD_SENAL) -> go.Figure:
    """Atajo para generar la figura del canal A1 (Trapecio Derecho)."""
    return figura_canal_individual(tiempo, senal, CANAL_DERECHO, COLOR_DERECHO, unidad=unidad)


def figura_trapecio_izquierdo(tiempo: np.ndarray, senal: np.ndarray, unidad: str = UNIDAD_SENAL) -> go.Figure:
    """Atajo para generar la figura del canal A2 (Trapecio Izquierdo)."""
    return figura_canal_individual(tiempo, senal, CANAL_IZQUIERDO, COLOR_IZQUIERDO, unidad=unidad)


def figura_comparativa(
    tiempo: np.ndarray, senal_derecha: np.ndarray, senal_izquierda: np.ndarray,
    unidad: str = UNIDAD_SENAL,
) -> go.Figure:
    """Crea una figura Plotly que superpone ambos canales para comparación
    visual directa de activación y asimetría."""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=tiempo, y=senal_derecha, mode="lines",
            name=CANAL_DERECHO, line=dict(color=COLOR_DERECHO, width=1),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=tiempo, y=senal_izquierda, mode="lines",
            name=CANAL_IZQUIERDO, line=dict(color=COLOR_IZQUIERDO, width=1),
        )
    )
    fig.update_layout(
        title="Comparación bilateral: Trapecio Derecho vs. Izquierdo",
        xaxis_title="Tiempo (s)",
        yaxis_title=f"Amplitud ({unidad})",
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def figura_comparativa_sincronizada(
    tiempo: np.ndarray, senal_derecha: np.ndarray, senal_izquierda: np.ndarray,
    unidad: str = UNIDAD_SENAL,
) -> go.Figure:
    """Crea dos subgráficos apilados (Trapecio Derecho / Izquierdo) con eje
    temporal compartido: el zoom o desplazamiento aplicado en uno se
    refleja automáticamente en el otro, facilitando la comparación bilateral
    directa.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(CANAL_DERECHO, CANAL_IZQUIERDO),
    )

    fig.add_trace(
        go.Scattergl(
            x=tiempo, y=senal_derecha, mode="lines",
            name=CANAL_DERECHO, line=dict(color=COLOR_DERECHO, width=1),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=tiempo, y=senal_izquierda, mode="lines",
            name=CANAL_IZQUIERDO, line=dict(color=COLOR_IZQUIERDO, width=1),
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text=f"Amplitud ({unidad})", row=1, col=1)
    fig.update_yaxes(title_text=f"Amplitud ({unidad})", row=2, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig.update_layout(
        title="Comparación sincronizada entre canales",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=45, r=20, t=60, b=40),
        font=dict(family="Georgia, 'Times New Roman', serif", size=13),
        height=550,
    )
    return fig


def figura_barras_rms(rms_derecho: float, rms_izquierdo: float, unidad: str = UNIDAD_SENAL) -> go.Figure:
    """Gráfico de barras comparando el RMS de ambos canales, útil como
    apoyo visual del índice de asimetría."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[CANAL_DERECHO, CANAL_IZQUIERDO],
                y=[rms_derecho, rms_izquierdo],
                marker_color=[COLOR_DERECHO, COLOR_IZQUIERDO],
                text=[f"{rms_derecho:.3f}", f"{rms_izquierdo:.3f}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"RMS por canal ({unidad})",
        yaxis_title=f"RMS ({unidad})",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=45, r=20, t=50, b=40),
        font=dict(family="Georgia, 'Times New Roman', serif", size=13),
    )
    return fig


def figura_indicador_asimetria(indice_porcentaje: float) -> go.Figure:
    """Crea un indicador tipo gauge (aguja) que muestra el índice de
    asimetría bilateral (%) sobre un fondo con las cuatro zonas de
    clasificación, para una lectura visual inmediata y prominente.
    """
    limite_superior = max(ASIMETRIA_MODERADA_MAX * 1.5, indice_porcentaje * 1.2, 60.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=indice_porcentaje,
            number={"suffix": " %", "font": {"size": 44}},
            title={"text": "Índice de Asimetría Bilateral", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, limite_superior]},
                "bar": {"color": COLOR_NEUTRO, "thickness": 0.35},
                "steps": [
                    {"range": [0, ASIMETRIA_SIMETRICO_MAX], "color": "#D9E2D9"},
                    {"range": [ASIMETRIA_SIMETRICO_MAX, ASIMETRIA_LEVE_MAX], "color": "#EDE6C8"},
                    {"range": [ASIMETRIA_LEVE_MAX, ASIMETRIA_MODERADA_MAX], "color": "#E8D2B0"},
                    {"range": [ASIMETRIA_MODERADA_MAX, limite_superior], "color": "#E3C0BE"},
                ],
                "threshold": {
                    "line": {"color": "#1F1F1F", "width": 3},
                    "thickness": 0.85,
                    "value": indice_porcentaje,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=10),
        height=320,
        font=dict(family="Georgia, 'Times New Roman', serif"),
    )
    return fig


def figura_fft(frecuencias: np.ndarray, magnitudes: np.ndarray, nombre_canal: str, color: str) -> go.Figure:
    """Genera la figura del espectro de amplitud (FFT) de un canal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frecuencias, y=magnitudes, mode="lines",
            name=nombre_canal, line=dict(color=color, width=1),
        )
    )
    fig.update_layout(
        title=f"Espectro de Frecuencia (FFT) - {nombre_canal}",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title=f"Magnitud ({UNIDAD_SENAL})",
        **_LAYOUT_BASE,
    )
    return fig


def figura_psd(frecuencias: np.ndarray, potencia: np.ndarray, nombre_canal: str, color: str) -> go.Figure:
    """Genera la figura de la Densidad Espectral de Potencia (Welch) de un canal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frecuencias, y=potencia, mode="lines",
            name=nombre_canal, line=dict(color=color, width=1),
        )
    )
    fig.update_layout(
        title=f"Densidad Espectral de Potencia (Welch) - {nombre_canal}",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title=f"Potencia ({UNIDAD_SENAL}²/Hz)",
        yaxis_type="log",
        **_LAYOUT_BASE,
    )
    return fig


def figura_espectrograma(
    frecuencias: np.ndarray, tiempos: np.ndarray, magnitud: np.ndarray, nombre_canal: str
) -> go.Figure:
    """Genera la figura del espectrograma (STFT) de un canal."""
    magnitud_db = 10 * np.log10(magnitud + 1e-12)

    fig = go.Figure(
        data=go.Heatmap(
            x=tiempos,
            y=frecuencias,
            z=magnitud_db,
            colorscale="Greys",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=f"Espectrograma - {nombre_canal}",
        xaxis_title="Tiempo (s)",
        yaxis_title="Frecuencia (Hz)",
        template="plotly_white",
        margin=dict(l=45, r=20, t=50, b=40),
        font=dict(family="Georgia, 'Times New Roman', serif", size=13),
    )
    return fig
