"""
app.py
======
Aplicación Streamlit para el análisis de activación y asimetría bilateral
de los músculos estabilizadores del tronco (trapecios) durante una
simulación de transporte público, mediante electromiografía de superficie.

Proyecto de Introducción a Señales Biomédicas.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    CONDICIONES,
    DATA_DIR,
    PARTICIPANTES,
    UNIDAD_SENAL,
)
from utils.espectral import calcular_espectrograma, calcular_fft, calcular_psd_welch
from utils.estadisticas import calcular_estadisticas, calcular_indice_asimetria
from utils.graficas import (
    COLOR_DERECHO,
    COLOR_IZQUIERDO,
    figura_barras_rms,
    figura_comparativa_sincronizada,
    figura_espectrograma,
    figura_fft,
    figura_indicador_asimetria,
    figura_psd,
    figura_trapecio_derecho,
    figura_trapecio_izquierdo,
)
from utils.lectura import (
    ArchivoNoEncontradoError,
    leer_senal_emg,
    listar_archivos_disponibles,
)
from utils.procesamiento import (
    ConfiguracionProcesamiento,
    normalizar_respecto_a_referencia,
    procesar_senal,
    recortar_por_intervalo,
)

# ---------------------------------------------------------------------------
# Configuración de la página (debe ser la primera llamada de Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inyectar_estilos() -> None:
    """Inyecta CSS personalizado para dar una apariencia sobria, seria y
    de laboratorio de investigación biomédica: sin colores llamativos ni
    elementos decorativos.

    Importante: se usan las variables de tema que expone Streamlit
    (--text-color, --background-color, --secondary-background-color) en
    lugar de colores fijos, para que el diseño se vea correctamente tanto
    en tema claro como en tema oscuro. Fijar colores de fondo claros a mano
    mientras el texto hereda el color del tema (blanco en modo oscuro)
    es lo que provocaba letras invisibles al cambiar de tema.
    """
    st.markdown(
        """
        <style>
        .main > div {padding-top: 1.5rem;}
        h1, h2, h3, h4 {
            font-family: Georgia, 'Times New Roman', serif;
            color: var(--text-color);
        }
        p, div, span, label {font-family: 'Segoe UI', Arial, sans-serif;}
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 4px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] {font-weight: 600; color: var(--text-color);}
        div[data-testid="stMetricValue"] {color: var(--text-color);}
        .stTabs [data-baseweb="tab-list"] {gap: 6px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
        }
        .bloque-asimetria {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-left: 5px solid #1F4E79;
            border-radius: 4px;
            padding: 20px 24px;
            color: var(--text-color);
        }
        .etiqueta-condicion {
            display: inline-block;
            background-color: #1F4E79;
            color: #FFFFFF;
            padding: 4px 14px;
            border-radius: 3px;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _barra_lateral() -> tuple[str, str, ConfiguracionProcesamiento, bool, bool]:
    """Construye la barra lateral y devuelve las selecciones del usuario:
    participante, condición, configuración de procesamiento, si se pulsó
    el botón de carga y si se solicitó normalización respecto a Basal.
    """
    with st.sidebar:
        st.markdown("## Panel de control")
        st.caption("Selección del registro EMG a analizar")

        participante = st.selectbox("Participante", options=list(PARTICIPANTES.keys()))
        condicion = st.selectbox("Condición experimental", options=list(CONDICIONES.keys()))

        cargar = st.button("Cargar señal", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### Procesamiento de la señal")

        st.checkbox("Señal original", value=True, disabled=True,
                     help="La señal original siempre está disponible como referencia base.")
        usar_rectificacion = st.checkbox("Rectificación de onda completa")
        usar_pasa_banda = st.checkbox("Filtro pasa banda configurable")

        f_baja, f_alta, orden = 20.0, 450.0, 4
        if usar_pasa_banda:
            col_a, col_b = st.columns(2)
            with col_a:
                f_baja = st.number_input("F. corte baja (Hz)", value=20.0, min_value=0.1, step=1.0)
            with col_b:
                f_alta = st.number_input("F. corte alta (Hz)", value=450.0, min_value=1.0, step=10.0)
            orden = st.slider("Orden del filtro", min_value=2, max_value=8, value=4, step=2)

        usar_notch = st.checkbox("Filtro notch 60 Hz")

        st.divider()
        st.markdown("### Normalización")
        usar_normalizacion = st.checkbox(
            "Normalizar respecto a la condición Basal",
            help=(
                "Expresa la señal como porcentaje del RMS de la condición "
                "Basal del mismo participante (100% = igual a la actividad "
                "basal de referencia)."
            ),
        )

        st.divider()
        with st.expander("Acerca del proyecto"):
            st.write(
                "Estudio piloto de electromiografía de superficie sobre "
                "los trapecios derecho e izquierdo, adquirido con BITalino "
                "a 1000 Hz durante la simulación de tres condiciones de "
                "transporte público: Basal, Una Asa y Doble Asa. Señales "
                f"expresadas en {UNIDAD_SENAL}."
            )

        config = ConfiguracionProcesamiento(
            usar_rectificacion=usar_rectificacion,
            usar_pasa_banda=usar_pasa_banda,
            usar_notch=usar_notch,
            f_baja=f_baja,
            f_alta=f_alta,
            orden_pasa_banda=int(orden),
        )

        return participante, condicion, config, cargar, usar_normalizacion


def _mostrar_metricas_canal(titulo: str, stats) -> None:
    """Muestra el conjunto completo de estadísticos de un canal como
    métricas de Streamlit, organizadas en una cuadrícula ordenada."""
    st.markdown(f"**{titulo}**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Media ({UNIDAD_SENAL})", f"{stats.media:.3f}")
    c2.metric(f"Mediana ({UNIDAD_SENAL})", f"{stats.mediana:.3f}")
    c3.metric(f"Desv. estándar ({UNIDAD_SENAL})", f"{stats.desviacion_estandar:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"Máximo ({UNIDAD_SENAL})", f"{stats.maximo:.3f}")
    c5.metric(f"Mínimo ({UNIDAD_SENAL})", f"{stats.minimo:.3f}")
    c6.metric(f"RMS ({UNIDAD_SENAL})", f"{stats.rms:.3f}")

    c7, c8, c9 = st.columns(3)
    c7.metric(f"MAV ({UNIDAD_SENAL})", f"{stats.mav:.3f}")
    c8.metric(f"Varianza ({UNIDAD_SENAL}²)", f"{stats.varianza:.4f}")
    c9.metric(f"Energía ({UNIDAD_SENAL}²)", f"{stats.energia:.2e}")


def _cargar_referencia_basal(participante: str, config: ConfiguracionProcesamiento):
    """Carga la condición Basal del participante, aplica la misma
    configuración de filtros y devuelve el RMS de referencia de cada canal
    (derecho, izquierdo). Devuelve None si el archivo Basal no existe.
    """
    try:
        senal_basal = leer_senal_emg(participante, "Basal", data_dir=DATA_DIR)
    except ArchivoNoEncontradoError:
        return None

    procesada_basal = procesar_senal(senal_basal, config)
    stats_d = calcular_estadisticas(procesada_basal.canal_derecho)
    stats_i = calcular_estadisticas(procesada_basal.canal_izquierdo)
    return stats_d.rms, stats_i.rms


def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit."""
    _inyectar_estilos()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    participante, condicion, config, cargar, usar_normalizacion = _barra_lateral()

    # st.session_state persiste la señal cargada entre interacciones (por
    # ejemplo, al cambiar checkboxes de filtros) sin recargar el archivo.
    if cargar:
        try:
            senal = leer_senal_emg(participante, condicion, data_dir=DATA_DIR)
            st.session_state["senal_emg"] = senal
            st.success(f"Señal cargada correctamente: {senal.nombre_archivo}")
        except ArchivoNoEncontradoError as error:
            st.session_state.pop("senal_emg", None)
            st.error(str(error))
            disponibles = listar_archivos_disponibles(DATA_DIR)
            if disponibles:
                with st.expander("Archivos disponibles en Datos/"):
                    for nombre in disponibles:
                        st.code(nombre)
            else:
                st.info(
                    f"La carpeta '{DATA_DIR}' no contiene archivos .txt. "
                    "Coloca allí los registros exportados de OpenSignals."
                )

    if "senal_emg" not in st.session_state:
        st.info(
            "Selecciona un participante y una condición en el panel lateral, "
            "luego presiona 'Cargar señal' para iniciar el análisis."
        )
        return

    senal = st.session_state["senal_emg"]

    # -----------------------------------------------------------------
    # Encabezado con información general del registro (siempre visible)
    # -----------------------------------------------------------------
    st.divider()
    st.markdown(f'<span class="etiqueta-condicion">Condición actual: {senal.condicion}</span>', unsafe_allow_html=True)
    st.write("")
    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
    info_col1.metric("Participante", senal.participante)
    info_col2.metric("Condición", senal.condicion)
    info_col3.metric("Duración total", f"{senal.duracion_segundos:.2f} s")
    info_col4.metric("N° muestras", f"{senal.n_muestras:,}")
    info_col5.metric("Unidad de la señal", UNIDAD_SENAL)

    # -----------------------------------------------------------------
    # Selección interactiva de intervalo temporal de análisis
    # -----------------------------------------------------------------
    st.markdown("#### Intervalo de análisis")
    t_inicio, t_fin = st.slider(
        "Ventana temporal utilizada en todo el análisis (s)",
        min_value=0.0,
        max_value=float(senal.duracion_segundos),
        value=(0.0, float(senal.duracion_segundos)),
        step=max(senal.duracion_segundos / 500, 0.01),
    )
    st.caption(
        f"Intervalo seleccionado: {t_inicio:.2f} s - {t_fin:.2f} s "
        f"({t_fin - t_inicio:.2f} s de duración). Todas las gráficas, "
        "estadísticas y el análisis espectral se calculan únicamente "
        "sobre este intervalo."
    )

    # Aplicar filtros configurados y luego recortar al intervalo elegido.
    procesada_completa = procesar_senal(senal, config)
    procesada = recortar_por_intervalo(procesada_completa, t_inicio, t_fin)

    unidad_actual = UNIDAD_SENAL
    if usar_normalizacion:
        referencia = _cargar_referencia_basal(participante, config)
        if referencia is not None:
            rms_ref_d, rms_ref_i = referencia
            procesada = normalizar_respecto_a_referencia(procesada, rms_ref_d, rms_ref_i)
            unidad_actual = "% RMS basal"
            st.caption(
                f"Normalización activa: señal expresada como porcentaje del "
                f"RMS basal de referencia (Derecho: {rms_ref_d:.3f} {UNIDAD_SENAL}, "
                f"Izquierdo: {rms_ref_i:.3f} {UNIDAD_SENAL})."
            )
        else:
            st.warning(
                "No se encontró el archivo Basal de este participante; "
                "no fue posible normalizar. Se muestra la señal en su "
                "unidad original."
            )

    # -----------------------------------------------------------------
    # Estadísticos base (usados en la sección de asimetría y en el resto
    # de la aplicación)
    # -----------------------------------------------------------------
    stats_derecho = calcular_estadisticas(procesada.canal_derecho)
    stats_izquierdo = calcular_estadisticas(procesada.canal_izquierdo)
    resultado_asimetria = calcular_indice_asimetria(stats_derecho.rms, stats_izquierdo.rms)

    # -----------------------------------------------------------------
    # SECCIÓN PRINCIPAL: Asimetría Bilateral (máxima relevancia visual)
    # -----------------------------------------------------------------
    st.divider()
    st.markdown("## Asimetría Bilateral")
    st.markdown('<div class="bloque-asimetria">', unsafe_allow_html=True)

    col_gauge, col_datos = st.columns([1.1, 1])
    with col_gauge:
        st.plotly_chart(
            figura_indicador_asimetria(resultado_asimetria.indice_porcentaje),
            use_container_width=True,
        )
    with col_datos:
        st.markdown(f"### {resultado_asimetria.clasificacion}")
        st.metric(f"RMS Trapecio Derecho ({unidad_actual})", f"{stats_derecho.rms:.3f}")
        st.metric(f"RMS Trapecio Izquierdo ({unidad_actual})", f"{stats_izquierdo.rms:.3f}")
        st.metric("Índice de Asimetría Bilateral", f"{resultado_asimetria.indice_porcentaje:.2f} %")
        st.caption(
            "Índice calculado como |RMS_derecho - RMS_izquierdo| / "
            "promedio(RMS_derecho, RMS_izquierdo) x 100."
        )
        with st.expander("Rangos de clasificación utilizados"):
            st.write("Simetría adecuada: hasta 10 %")
            st.write("Asimetría leve: 10 % a 20 %")
            st.write("Asimetría moderada: 20 % a 40 %")
            st.write("Asimetría severa: mayor a 40 %")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Pestañas secundarias
    # -----------------------------------------------------------------
    st.divider()
    tab_senales, tab_comparacion, tab_estadisticas, tab_espectral, tab_exportar = st.tabs(
        [
            "Señales",
            "Comparación Sincronizada",
            "Estadísticas",
            "Análisis Espectral",
            "Exportar",
        ]
    )

    # --- Pestaña de señales en el dominio del tiempo -----------------------
    with tab_senales:
        st.plotly_chart(
            figura_trapecio_derecho(procesada.tiempo, procesada.canal_derecho, unidad=unidad_actual),
            use_container_width=True,
        )
        st.plotly_chart(
            figura_trapecio_izquierdo(procesada.tiempo, procesada.canal_izquierdo, unidad=unidad_actual),
            use_container_width=True,
        )
        st.plotly_chart(
            figura_barras_rms(stats_derecho.rms, stats_izquierdo.rms, unidad=unidad_actual),
            use_container_width=True,
        )

    # --- Pestaña de comparación sincronizada --------------------------------
    with tab_comparacion:
        st.caption(
            "El zoom o desplazamiento aplicado sobre un canal se refleja "
            "automáticamente en el otro (eje temporal compartido)."
        )
        st.plotly_chart(
            figura_comparativa_sincronizada(
                procesada.tiempo, procesada.canal_derecho, procesada.canal_izquierdo,
                unidad=unidad_actual,
            ),
            use_container_width=True,
        )

    # --- Pestaña de estadísticas ---------------------------------------------
    with tab_estadisticas:
        col_der, col_izq = st.columns(2)
        with col_der:
            with st.container(border=True):
                _mostrar_metricas_canal("Trapecio Derecho (A1)", stats_derecho)
        with col_izq:
            with st.container(border=True):
                _mostrar_metricas_canal("Trapecio Izquierdo (A2)", stats_izquierdo)

    # --- Pestaña de análisis espectral ---------------------------------------
    with tab_espectral:
        sub_fft, sub_psd, sub_espectrograma = st.tabs(["FFT", "PSD (Welch)", "Espectrograma"])

        with sub_fft:
            fft_d = calcular_fft(procesada.canal_derecho, procesada.fs)
            fft_i = calcular_fft(procesada.canal_izquierdo, procesada.fs)
            st.plotly_chart(
                figura_fft(fft_d.frecuencias, fft_d.magnitudes, "Trapecio Derecho (A1)", COLOR_DERECHO),
                use_container_width=True,
            )
            st.plotly_chart(
                figura_fft(fft_i.frecuencias, fft_i.magnitudes, "Trapecio Izquierdo (A2)", COLOR_IZQUIERDO),
                use_container_width=True,
            )

        with sub_psd:
            psd_d = calcular_psd_welch(procesada.canal_derecho, procesada.fs)
            psd_i = calcular_psd_welch(procesada.canal_izquierdo, procesada.fs)
            st.plotly_chart(
                figura_psd(psd_d.frecuencias, psd_d.potencia, "Trapecio Derecho (A1)", COLOR_DERECHO),
                use_container_width=True,
            )
            st.plotly_chart(
                figura_psd(psd_i.frecuencias, psd_i.potencia, "Trapecio Izquierdo (A2)", COLOR_IZQUIERDO),
                use_container_width=True,
            )

        with sub_espectrograma:
            espec_d = calcular_espectrograma(procesada.canal_derecho, procesada.fs)
            espec_i = calcular_espectrograma(procesada.canal_izquierdo, procesada.fs)
            st.plotly_chart(
                figura_espectrograma(
                    espec_d.frecuencias, espec_d.tiempos, espec_d.magnitud, "Trapecio Derecho (A1)"
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                figura_espectrograma(
                    espec_i.frecuencias, espec_i.tiempos, espec_i.magnitud, "Trapecio Izquierdo (A2)"
                ),
                use_container_width=True,
            )

    # --- Pestaña de exportación -----------------------------------------------
    with tab_exportar:
        st.write(
            "Descarga un resumen en CSV con los parámetros clave calculados "
            "para el intervalo y configuración actualmente analizados."
        )

        fila_resumen = pd.DataFrame(
            [
                {
                    "Participante": senal.participante,
                    "Condicion": senal.condicion,
                    "Intervalo_inicio_s": t_inicio,
                    "Intervalo_fin_s": t_fin,
                    "Unidad": unidad_actual,
                    "RMS_Derecho": stats_derecho.rms,
                    "RMS_Izquierdo": stats_izquierdo.rms,
                    "MAV_Derecho": stats_derecho.mav,
                    "MAV_Izquierdo": stats_izquierdo.mav,
                    "Indice_Asimetria_%": resultado_asimetria.indice_porcentaje,
                    "Clasificacion_Asimetria": resultado_asimetria.clasificacion,
                }
            ]
        )

        st.dataframe(fila_resumen, use_container_width=True, hide_index=True)

        buffer = io.StringIO()
        fila_resumen.to_csv(buffer, index=False)

        st.download_button(
            label="Descargar CSV",
            data=buffer.getvalue(),
            file_name=f"resumen_{senal.participante}_{senal.condicion}.csv".lower().replace(" ", "_"),
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
