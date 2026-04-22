import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp

# Fondo blanco unificado (Plotly por defecto suele usar la plantilla "plotly", más oscura).
pio.templates.default = "plotly_white"

# Para merge en ``update_layout`` / exportación (también importable desde otros scripts).
PLOTLY_WHITE_LAYOUT_KWARGS = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
)

# =============================================================================
# DATETIME & I/O PREPROCESSING
# =============================================================================


def safe_read_csv(path):

    if not os.path.exists(path):
        print(f"⚠️ Archivo no encontrado: {path}")
        return pd.DataFrame()

    try:
        if os.path.getsize(path) == 0:
            print(f"⚠️ Archivo vacío: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"⚠️ Archivo sin columnas válidas: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error al leer {path}: {e}")
        return pd.DataFrame()


def add_datetime_column(df):
    df.rename(columns={'day_of_month': 'day'}, inplace=True)
    df['year'] = 2006
    df.loc[df['month'] < 5, 'year'] = 2007
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df


def add_datetime_column_v2(df):
    df.rename(columns={'day_of_month': 'day'}, inplace=True)
    df['year'] = 1991
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df


def filer_interval(df, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df[(df['datetime'] >= start) & (df['datetime'] <= end)]


def resample(df):
    output = df.resample('1h', on='datetime').mean()
    output.reset_index(inplace=True)
    return output


# =============================================================================
# FIGURE EXPORT (high quality PNG + HTML)
# =============================================================================


def save_figure(fig, path_stem, width=1200, height=700, scale=2):
    """
    Guarda la figura en PNG (alta calidad) y HTML.
    path_stem: Path o str sin extensión (ej. output_dir / 'nombre').
    ``width``/``height`` se aplican también a ``layout`` para que el HTML y el PNG
    (Kaleido) compartan la misma geometría y no diverjan márgenes o ejes.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=width, height=height, **PLOTLY_WHITE_LAYOUT_KWARGS)
    try:
        fig.write_image(
            str(path_stem.with_suffix('.png')),
            width=width,
            height=height,
            scale=scale,
        )
    except Exception as e:
        print(f"⚠️ No se pudo exportar PNG ({path_stem.name}): {e}")
    fig.write_html(str(path_stem.with_suffix('.html')))


# =============================================================================
# SUMMARY / AGGREGATION
# =============================================================================


def mean_variable(df, variable):
    return df[variable].mean()


def compute_crf_daily_stats(df, crf_col='crf', datetime_col='datetime'):
    """
    A partir de la serie crf (0/1 o continua) y datetime, calcula por día:
    - Número de transiciones encendido/apagado (cualquier cambio).
    - Número de encendidos estrictos (transición 0 -> 1).
    Devuelve (mean_transitions_per_day, std_transitions, mean_ons_per_day, std_ons) o None.
    """
    if crf_col not in df.columns or datetime_col not in df.columns:
        return None
    d = df[[datetime_col, crf_col]].copy()
    d[datetime_col] = pd.to_datetime(d[datetime_col])
    d = d.sort_values(datetime_col).reset_index(drop=True)
    d['_date'] = d[datetime_col].dt.date
    on = (d[crf_col] > 0.5).astype(int)

    rows = []
    for _date, grp in on.groupby(d['_date']):
        ser = grp.diff().fillna(0)
        trans = (ser.abs() > 0.5).astype(int).sum()
        ons = (ser == 1).astype(int).sum()
        rows.append({'transitions': trans, 'ons': ons})
    by_day = pd.DataFrame(rows)
    if by_day.empty or len(by_day) < 2:
        return None
    mean_trans = float(by_day['transitions'].mean().item())
    std_trans = float(by_day['transitions'].std().item()) if len(by_day) > 0 else 0.0
    if not math.isfinite(std_trans):
        std_trans = 0.0
    mean_ons = float(by_day['ons'].mean().item())
    std_ons = float(by_day['ons'].std().item()) if len(by_day) > 0 else 0.0
    if not math.isfinite(std_ons):
        std_ons = 0.0
    return (mean_trans, std_trans, mean_ons, std_ons)


# =============================================================================
# PLOTTING — LINES & PROGRESS
# =============================================================================

# Eje X para series temporales (datetime): solo mes abreviado (Jan, Feb, …)
_DATETIME_X_AXIS_FORMAT = dict(dtick='M1', tickformat='%b', ticklabelmode='period')

# Patrones de línea para distinguir series en B/N (Plotly: solid, dash, dot, …)
_LINE_DASH_CYCLE = ('solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot')


def plot_dfs_line(df_dict, variable_name, colors=None, line_styles=None):
    """
    Genera un gráfico de líneas de progreso.

    Parámetros:
    - dfs: lista de DataFrames que contienen los datos de progreso
    - names: lista de nombres para cada línea
    - variable_name: nombre de la columna en los DataFrames que se desea graficar en el eje y
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    - line_styles: lista opcional de dash de Plotly por traza; si es None, se cicla _LINE_DASH_CYCLE
    """
    n = len(df_dict)

    # Colores predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[:n]
    if line_styles is None:
        line_styles = [_LINE_DASH_CYCLE[i % len(_LINE_DASH_CYCLE)] for i in range(n)]
    elif len(line_styles) < n:
        ls = list(line_styles)
        line_styles = (ls * ((n + len(ls) - 1) // len(ls)))[:n]

    # Crear figura
    fig = go.Figure()

    # Añadir líneas para cada DataFrame
    for (name, df), color, dash in zip(df_dict.items(), colors, line_styles):
        line_kw = dict(color=color, width=2)
        if dash is not None and str(dash).lower() != 'solid':
            line_kw['dash'] = dash
        fig.add_trace(
            go.Scatter(
                x=df['episode_num'],
                y=df[variable_name],
                mode='lines',
                line=line_kw,
                name=name,
            )
        )

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title='Datetime',
        yaxis_title=None,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        height=600,
        width=1000,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


def plot_training_reward_terms_progression(
    df,
    episode_col='episode_num',
    comfort_col='mean_reward_comfort_term',
    energy_col='mean_reward_energy_term',
    std_comfort_col='std_reward_comfort_term',
    std_energy_col='std_reward_energy_term',
    title=None,
    color_comfort='#3498DB',
    color_energy='#E74C3C',
    show_std_band=True,
):
    """
    Gráfico de líneas: evolución del término de comfort y del término de energía
    a lo largo de los episodios de entrenamiento (progress.csv). Ambos términos
    superpuestos para comparar su equilibrio durante el entrenamiento.

    Parámetros:
    - df: DataFrame con al menos episode_col, comfort_col, energy_col
    - std_comfort_col, std_energy_col: columnas opcionales para dibujar banda ±1 std
    - title: título de la figura
    - color_comfort, color_energy: colores de cada término
    - show_std_band: si True y existen columnas std, dibuja banda sombreada
    """
    if episode_col not in df.columns or comfort_col not in df.columns or energy_col not in df.columns:
        raise ValueError(
            f"El DataFrame debe contener las columnas '{episode_col}', '{comfort_col}' y '{energy_col}'."
        )
    x = df[episode_col]
    y_comfort = pd.to_numeric(df[comfort_col], errors='coerce')
    y_energy = pd.to_numeric(df[energy_col], errors='coerce')

    fig = go.Figure()

    has_std_comfort = show_std_band and std_comfort_col in df.columns
    has_std_energy = show_std_band and std_energy_col in df.columns

    if has_std_comfort:
        s_c = np.asarray(pd.to_numeric(df[std_comfort_col], errors='coerce').fillna(0))
        y_c = np.asarray(y_comfort)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_c + s_c,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_c - s_c,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.15)',
                showlegend=False,
                hoverinfo='skip',
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_comfort,
            mode='lines',
            name='Comfort term',
            line=dict(color=color_comfort, width=2),
        )
    )

    if has_std_energy:
        s_e = np.asarray(pd.to_numeric(df[std_energy_col], errors='coerce').fillna(0))
        y_e = np.asarray(y_energy)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_e + s_e,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_e - s_e,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.15)',
                showlegend=False,
                hoverinfo='skip',
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_energy,
            mode='lines',
            name='Energy term',
            line=dict(color=color_energy, width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title='Episode',
        yaxis_title='Reward term (mean per episode)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=500,
        width=1000,
    )
    return fig


def plot_episode_reward_terms_timestep(
    df,
    datetime_col='datetime',
    comfort_col='comfort_term',
    energy_col='energy_term',
    title=None,
    color_comfort='#3498DB',
    color_energy='#E74C3C',
    smooth_window=None,
):
    """
    Gráfico de líneas: evolución de los términos comfort y energy en cada timestep
    a lo largo de un episodio (datos de infos.csv, una fila por timestep). Ambos
    términos superpuestos para ver el equilibrio a nivel granular.

    Parámetros:
    - df: DataFrame con columnas datetime_col, comfort_col, energy_col (una fila por timestep).
    - smooth_window: si se indica, se aplica media móvil a las series para suavizar (ej. 12).
    """
    if datetime_col not in df.columns or comfort_col not in df.columns or energy_col not in df.columns:
        raise ValueError(
            f"El DataFrame debe contener las columnas '{datetime_col}', '{comfort_col}' y '{energy_col}'."
        )
    x = pd.to_datetime(df[datetime_col])
    y_comfort = pd.to_numeric(df[comfort_col], errors='coerce')
    y_energy = pd.to_numeric(df[energy_col], errors='coerce')
    if smooth_window:
        y_comfort = y_comfort.rolling(window=int(smooth_window), min_periods=1).mean()
        y_energy = y_energy.rolling(window=int(smooth_window), min_periods=1).mean()

    # Asignación serie <-> leyenda: en algunos exports la columna 'comfort_term' del CSV
    # corresponde al término de energía y 'energy_term' al de confort. Se dibuja de forma que
    # la leyenda coincida con la serie correcta (Comfort = penalización temperatura, Energy = consumo).
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_energy,
            mode='lines',
            name='Comfort term',
            line=dict(color=color_comfort, width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_comfort,
            mode='lines',
            name='Energy term',
            line=dict(color=color_energy, width=1.5),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Reward term (per timestep)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis=dict(**_DATETIME_X_AXIS_FORMAT),
        height=500,
        width=1000,
    )
    return fig


# =============================================================================
# PLOTTING — CONTROL (temperature vs flow)
# =============================================================================


def plot_control(
    df,
    temperature_variables,
    flow_variables,
    names,
    colors=None,
    outdoor_temp_var='outdoor_temperature',
):
    """
    Gráfico de líneas: temperaturas y caudales en ejes secundarios.
    usando ejes separados para cada tipo (temperatura a la izquierda, caudal a la derecha).
    Opcionalmente incluye la temperatura exterior en el mismo eje Y que las temperaturas internas.

    Parámetros:
    - df: DataFrame con columna 'datetime' y los datos a graficar.
    - temperature_variables: lista de columnas con temperaturas.
    - flow_variables: lista de columnas con caudales.
    - names: lista de nombres a mostrar en la leyenda (orden: temps + flows).
    - colors: lista opcional de colores para cada línea.
    - outdoor_temp_var: nombre de la columna de temperatura exterior (misma escala Y que temps).
        Si existe en df se dibuja en negro oscuro. None para no mostrarla.
    """

    # Validación de colores
    total_vars = len(temperature_variables) + len(flow_variables)
    if colors is None:
        colors = px.colors.qualitative.Plotly[:total_vars]
    elif len(colors) < total_vars:
        raise ValueError("No hay suficientes colores para todas las variables.")

    # Asegurar que datetime esté parseado
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Crear figura vacía
    fig = go.Figure()

    # Temperatura exterior (misma escala Y que las internas), en negro oscuro
    if outdoor_temp_var is not None and outdoor_temp_var in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[outdoor_temp_var],
                mode='lines',
                name='Outdoor temperature',
                line=dict(color='#1a1a1a', width=2),
                yaxis='y',
            )
        )

    # Añadir trazos de temperatura
    for i, var in enumerate(temperature_variables):
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[var],
                mode='lines',
                name=names[i],
                line=dict(color=colors[i], width=2),
                opacity=0.5,
                yaxis='y',
            )
        )

    # Añadir trazos de caudal (todos en eje y2)
    for j, var in enumerate(flow_variables):
        idx = len(temperature_variables) + j
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[var],
                mode='lines',
                name=names[idx],
                line=dict(color=colors[idx], dash='dot', width=2),
                opacity=0.7,
                yaxis='y2',
            )
        )

    # Configurar ejes: X con datetime explícito
    fig.update_layout(
        xaxis=dict(
            title='Datetime',
            type='date',
            **_DATETIME_X_AXIS_FORMAT,
        ),
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Flow rate', overlaying='y', side='right', showgrid=False),
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=14),
        ),
        width=1000,
        height=600,
        title=None,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    return fig


# =============================================================================
# PLOTTING — SMOOTHED SIGNALS / HEAT WORK
# =============================================================================


def plot_smoothed_signal(
    df,
    variable,
    datetime_col='datetime',
    window=16,
    color='#2980B9',
    title=None,
    yaxis_title=None,
):
    """Plot raw and rolling-mean versions of one signal."""
    if datetime_col not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{datetime_col}'.")
    if variable not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{variable}'.")

    x = pd.to_datetime(df[datetime_col])
    y_raw = pd.to_numeric(df[variable], errors='coerce')
    y_smooth = y_raw.rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_raw,
            mode='lines',
            name='raw',
            line=dict(color=color, width=1),
            opacity=0.25,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_smooth,
            mode='lines',
            name=f'rolling mean ({window})',
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title=yaxis_title if yaxis_title is not None else variable,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )
    fig.update_xaxes(**_DATETIME_X_AXIS_FORMAT)
    return fig


def plot_heat_work(
    df,
    requested_var,
    outlet_var,
    datetime_col='datetime',
    requested_name='Water temperature setpoint',
    outlet_name='Heat source outlet temperature',
    title=None,
):
    """Plot requested water temperature vs real outlet temperature with shaded gap."""
    if datetime_col not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{datetime_col}'.")
    if requested_var not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{requested_var}'.")
    if outlet_var not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{outlet_var}'.")

    x = pd.to_datetime(df[datetime_col])
    y_request = pd.to_numeric(df[requested_var], errors='coerce')
    y_outlet = pd.to_numeric(df[outlet_var], errors='coerce')

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_request,
            mode='lines',
            name=requested_name,
            line=dict(color='#2980B9', width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_outlet,
            mode='lines',
            name=outlet_name,
            line=dict(color='#7F8C8D', width=2),
            fill='tonexty',
            fillcolor='rgba(127, 140, 141, 0.20)',
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title='Temperature (°C)',
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )
    fig.update_xaxes(**_DATETIME_X_AXIS_FORMAT)
    return fig


# =============================================================================
# PLOTTING — TEMPERATURES
# =============================================================================


def plot_temperatures_v2(df, variables, names, upper_limit, lower_limit, colors=None):
    """
    Genera un gráfico de líneas para varias columnas de temperatura en un DataFrame.

    Parámetros:
    - df: DataFrame que contiene los datos de temperatura
    - variables: lista de nombres de las columnas en el DataFrame que se desean graficar en el eje y
    - names: lista de nombres que se utilizarán en la leyenda para cada línea
    - upper_limit: valor de límite superior para los límites
    - lower_limit: valor de límite inferior para los límites
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    """
    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(variables)]

    # Crear una copia para no modificar el DataFrame original
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])

    # Agregar minutos incrementales para timestamps duplicados
    # Esto resuelve el problema de múltiples puntos con el mismo timestamp
    # que causa que write_image() no renderice las líneas correctamente
    df_copy['minutes_offset'] = df_copy.groupby('datetime').cumcount() * 10
    df_copy['datetime'] = df_copy['datetime'] + pd.to_timedelta(
        df_copy['minutes_offset'], unit='m'
    )
    df_copy.drop('minutes_offset', axis=1, inplace=True)

    fig = go.Figure()

    for i, var in enumerate(variables):
        # Verificar que la variable existe en el DataFrame
        if var not in df_copy.columns:
            print(f"Advertencia: La variable '{var}' no existe en el DataFrame")
            continue

        fig.add_trace(
            go.Scatter(
                x=df_copy['datetime'],
                y=df_copy[var],
                mode='lines',
                name=names[i],
                line=dict(color=colors[i]),
                opacity=0.5,
            )
        )

    fig.add_hline(y=upper_limit, line_dash="dot", line_color="blue")
    fig.add_hline(y=lower_limit, line_dash="dot", line_color="red")

    fig.update_layout(
        title=None,
        xaxis_title='',
        yaxis_title='Temperature (°C)',
        xaxis=dict(type='date', **_DATETIME_X_AXIS_FORMAT),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=20),
        ),
        width=1000,
        height=600,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    return fig


def plot_temperatures(
    df,
    variables,
    names,
    threshold_variable='htg_setpoint_living',
    threshold_limit=1,
    colors=None,
):
    """
    Genera un gráfico de líneas para varias columnas de temperatura en un DataFrame.

    Parámetros:
    - df: DataFrame que contiene los datos de temperatura
    - variables: lista de nombres de las columnas en el DataFrame que se desean graficar en el eje y
    - names: lista de nombres que se utilizarán en la leyenda para cada línea
    - threshold_variable: nombre de la columna en el DataFrame que contiene el valor de referencia para los límites
    - threshold_limit: valor de diferencia para los límites superior e inferior
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    """

    # Colores predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(variables)]

    # Trabajar sobre una copia para no mutar el DataFrame original
    df_copy = df.copy()

    # Validación básica
    if 'datetime' not in df_copy.columns:
        raise ValueError("El DataFrame debe contener una columna 'datetime'.")
    if threshold_variable not in df_copy.columns:
        raise ValueError(
            f"La columna umbral '{threshold_variable}' no existe en el DataFrame."
        )

    # Asegurar datetime y evitar timestamps duplicados (kaleido/write_image puede fallar)
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    df_copy['minutes_offset'] = df_copy.groupby('datetime').cumcount() * 10
    df_copy['datetime'] = df_copy['datetime'] + pd.to_timedelta(
        df_copy['minutes_offset'], unit='m'
    )
    df_copy.drop('minutes_offset', axis=1, inplace=True)

    # Calcular límites superior e inferior dinámicos
    df_copy['down_limit'] = df_copy[threshold_variable] - threshold_limit
    df_copy['up_limit'] = df_copy[threshold_variable] + threshold_limit

    # Crear figura base
    fig = go.Figure()

    # Línea inferior
    fig.add_trace(
        go.Scatter(
            x=df_copy['datetime'],
            y=df_copy['down_limit'],
            mode="lines",
            line=dict(color='blue', dash="dot"),
            showlegend=False,
            name="Límite inferior",
            hoverinfo='skip',
        )
    )

    # Línea superior + sombreado
    fig.add_trace(
        go.Scatter(
            x=df_copy['datetime'],
            y=df_copy['up_limit'],
            mode="lines",
            line=dict(color='red', dash="dot"),
            fill='tonexty',  # Sombrea entre inferior y superior
            fillcolor='rgba(200, 200, 200, 0.4)',
            showlegend=False,
            name="Límite superior",
            hoverinfo='skip',
        )
    )

    # Añadir las líneas de temperatura
    for i, var in enumerate(variables):
        fig.add_trace(
            go.Scatter(
                x=df_copy['datetime'],
                y=df_copy[var],
                mode='lines',
                name=names[i],
                line=dict(color=colors[i]),
                opacity=0.5,
            )
        )

    fig.update_xaxes(**_DATETIME_X_AXIS_FORMAT)

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title='',
        yaxis_title='Temperature (°C)',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=20),
        ),
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        width=1000,
        height=600,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


def _ensure_datetime_unique(df):
    """Asegura datetimes únicos para evitar fallos en write_image (kaleido)."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['minutes_offset'] = df.groupby('datetime').cumcount() * 10
    df['datetime'] = df['datetime'] + pd.to_timedelta(df['minutes_offset'], unit='m')
    df.drop('minutes_offset', axis=1, inplace=True)
    return df


def plot_temperature_one_zone(
    df,
    temp_var,
    setpoint_var,
    zone_name,
    threshold=1.0,
    temp_color=None,
    outdoor_temp_var='outdoor_temperature',
):
    """
    Una gráfica por zona térmica para un DataFrame: temperatura, setpoint variable en el tiempo
    y banda setpoint ± threshold (igual que plot_temperatures_subplots pero con temp/setpoint
    especificados uno a uno). Pensada para llamarse en bucle por zona. Cada serie usa color y
    patrón de línea distintos (sólido, guiones, puntos…) para legibilidad en blanco y negro.

    Parámetros:
    - df: DataFrame con 'datetime', temp_var y setpoint_var
    - temp_var: columna de temperatura de esa zona
    - setpoint_var: columna de setpoint de esa zona (varía con datetime)
    - zone_name: nombre de la zona para título/leyenda
    - threshold: margen en °C para la banda (setpoint ± threshold)
    - temp_color: color opcional para la línea de temperatura
    - outdoor_temp_var: columna de temperatura exterior a dibujar en gris. Por defecto
      'outdoor_temperature'; si existe en df se añade una línea en eje Y derecho propio
      (escala independiente del interior/setpoint). None para no mostrarla.
    """
    if 'datetime' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'datetime'.")
    if setpoint_var not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{setpoint_var}'.")
    if temp_var not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{temp_var}'.")

    df_copy = _ensure_datetime_unique(df.copy())

    fig = go.Figure()

    # Línea de setpoint (varía con datetime)
    fig.add_trace(
        go.Scatter(
            x=df_copy['datetime'],
            y=df_copy[setpoint_var],
            mode='lines',
            name='Setpoint',
            line=dict(color='black', width=1.5, dash='dash'),
        )
    )
    # Línea de temperatura
    line_color = (
        temp_color if temp_color is not None else px.colors.qualitative.Plotly[0]
    )
    fig.add_trace(
        go.Scatter(
            x=df_copy['datetime'],
            y=df_copy[temp_var],
            mode='lines',
            name=zone_name,
            line=dict(color=line_color, dash='solid', width=2),
            opacity=0.9,
        )
    )
    has_outdoor = (
        outdoor_temp_var is not None and outdoor_temp_var in df_copy.columns
    )
    # Línea de temperatura exterior: eje Y2 a la derecha, escala independiente
    if has_outdoor:
        fig.add_trace(
            go.Scatter(
                x=df_copy['datetime'],
                y=df_copy[outdoor_temp_var],
                mode='lines',
                name='Outdoor temperature',
                yaxis='y2',
                line=dict(color='gray', width=1.2, dash='dashdot'),
                opacity=0.8,
            )
        )

    fig.update_xaxes(**_DATETIME_X_AXIS_FORMAT)
    layout = dict(
        **PLOTLY_WHITE_LAYOUT_KWARGS,
        title=None,
        xaxis_title='',
        yaxis=dict(
            title='Temperature (°C)',
            side='left',
            showgrid=True,
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=14),
        ),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        width=1000,
        height=500,
    )
    if has_outdoor:
        layout['yaxis2'] = dict(
            title=dict(text='Outdoor temperature (°C)', font=dict(color='gray')),
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(color='gray'),
        )
    fig.update_layout(**layout)
    return fig


def plot_temperatures_subplots(
    df,
    thermal_zones,
    setpoint_band=1.0,
    colors=None,
):
    """
    Genera una sola figura con un subplot por zona térmica (temperatura + setpoint ± band).

    Parámetros:
    - df: DataFrame con 'datetime', columnas de temperatura (claves de thermal_zones)
          y columnas heating_setpoint_* correspondientes.
    - thermal_zones: dict {columna_temperatura: nombre_leyenda}.
    - setpoint_band: margen en ºC alrededor del setpoint (setpoint ± setpoint_band).
    - colors: lista opcional de colores por zona; si no, se usan Plotly por defecto.

    Devuelve: go.Figure con make_subplots (una fila/columna por zona).
    """
    df_copy = df.copy()
    if "datetime" not in df_copy.columns:
        raise ValueError("El DataFrame debe contener la columna 'datetime'.")
    df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])
    df_copy["minutes_offset"] = df_copy.groupby("datetime").cumcount() * 10
    df_copy["datetime"] = df_copy["datetime"] + pd.to_timedelta(
        df_copy["minutes_offset"], unit="m"
    )
    df_copy.drop("minutes_offset", axis=1, inplace=True)

    valid = [
        (zcol, zname)
        for zcol, zname in thermal_zones.items()
        if zcol in df_copy.columns
        and (setpoint_col := zcol.replace("air_temperature_", "heating_setpoint_"))
        in df_copy.columns
    ]
    if not valid:
        raise ValueError("No hay columnas de zona/setpoint válidas en el DataFrame.")

    n = len(valid)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    subplot_titles = [zname for _, zname in valid]
    fig = sp.make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    if colors is None:
        colors = px.colors.qualitative.Plotly[:n]

    for idx, (zone_col, zone_name) in enumerate(valid):
        setpoint_col = zone_col.replace("air_temperature_", "heating_setpoint_")
        row, col = idx // ncols + 1, idx % ncols + 1
        df_copy["_down"] = df_copy[setpoint_col] - setpoint_band
        df_copy["_up"] = df_copy[setpoint_col] + setpoint_band

        fig.add_trace(
            go.Scatter(
                x=df_copy["datetime"],
                y=df_copy["_down"],
                mode="lines",
                line=dict(color="blue", dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=df_copy["datetime"],
                y=df_copy["_up"],
                mode="lines",
                line=dict(color="red", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(200, 200, 200, 0.4)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=df_copy["datetime"],
                y=df_copy[zone_col],
                mode="lines",
                name=zone_name,
                line=dict(color=colors[idx]),
                opacity=0.8,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(**_DATETIME_X_AXIS_FORMAT)
    fig.update_layout(
        title=None,
        height=250 * nrows,
        width=1000,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )
    fig.update_yaxes(title_text="Temperature (°C)", row="all", col=1)
    return fig


def plot_dfs_line_grouped_by_month(df_dict, variable, colors=None, line_styles=None):
    """
    Gráfico de líneas agrupado por meses de la variable deseada.
    df_dict: dict nombre -> DataFrame (ej. unified).
    variable: nombre de la columna a graficar.
    """
    names = list(df_dict.keys())
    dfs = list(df_dict.values())

    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(dfs)]
    if line_styles is None:
        line_styles = [None for _ in range(len(dfs))]

    # Crear figura
    fig = go.Figure()

    # Para cada DataFrame, calcular la media mensual de variable
    all_dates = []  # Lista para almacenar todas las fechas (meses) únicas
    for df, name, color, line_style in zip(dfs, names, colors, line_styles):

        # Calcular la media mensual
        df['year_month'] = df['datetime'].dt.to_period(
            'M'
        )  # Extraer año y mes como periodo
        monthly_avg = (
            df.groupby('year_month')[variable].mean().reset_index()
        )  # Calcular la media
        # Convertir de nuevo a datetime
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()

        # Añadir las fechas únicas al conjunto de fechas para el tick en el eje
        # X
        all_dates.extend(monthly_avg['year_month'].tolist())

        # Añadir la línea de la media mensual al gráfico
        fig.add_trace(
            go.Scatter(
                x=monthly_avg['year_month'],
                y=monthly_avg[variable],
                mode='lines',
                name=name,
                line=dict(color=color, dash=line_style),
            )
        )

    # Filtrar las fechas únicas y seleccionar 5 fechas espaciadas uniformemente
    unique_dates = sorted(list(set(all_dates)))
    tickvals = [
        unique_dates[i]
        for i in range(0, len(unique_dates), max(1, len(unique_dates) // 5))
    ]

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title='Power demand (W)',
        xaxis=dict(
            type='date',
            **_DATETIME_X_AXIS_FORMAT,
            tickvals=tickvals,
            tickangle=45,
        ),
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        height=600,
        width=1000,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


def plot_dfs_bar_grouped_by_month(df_dict, variable, colors=None):
    """
    Gráfico de barras agrupado por meses de la variable deseada.
    df_dict: dict nombre -> DataFrame (ej. unified).
    variable: nombre de la columna a graficar.
    """
    names = list(df_dict.keys())
    dfs = list(df_dict.values())

    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(dfs)]

    # Crear figura
    fig = go.Figure()

    # Para cada DataFrame, calcular la media mensual de variable
    all_dates = []  # Lista para almacenar todas las fechas (meses) únicas
    for df, name, color in zip(dfs, names, colors):
        # Calcular la media mensual
        monthly_df = df.copy()
        monthly_df['year_month'] = pd.to_datetime(monthly_df['datetime']).dt.to_period('M')
        monthly_avg = monthly_df.groupby('year_month')[variable].mean().reset_index()
        # Convertir de nuevo a datetime
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()

        # Añadir las fechas únicas al conjunto de fechas para el tick en el eje X
        all_dates.extend(monthly_avg['year_month'].tolist())

        # Añadir barras de la media mensual al gráfico
        fig.add_trace(
            go.Bar(
                x=monthly_avg['year_month'],
                y=monthly_avg[variable],
                name=name,
                marker_color=color,
            )
        )

    # Filtrar las fechas únicas y seleccionar 5 fechas espaciadas uniformemente
    unique_dates = sorted(list(set(all_dates)))
    tickvals = [
        unique_dates[i]
        for i in range(0, len(unique_dates), max(1, len(unique_dates) // 5))
    ]

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title='Power demand (W)',
        xaxis=dict(
            type='date',
            **_DATETIME_X_AXIS_FORMAT,
            tickvals=tickvals,
            tickangle=45,
        ),
        barmode='group',
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        height=600,
        width=1000,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


# =============================================================================
# PLOTTING — BARS
# =============================================================================


def plot_bar(dict_data, bar_colors=None):
    keys = list(dict_data.keys())
    values = list(dict_data.values())

    if bar_colors is None:
        bar_colors = ["royalblue"] * len(keys)

    if len(bar_colors) != len(keys):
        raise ValueError(
            "La cantidad de colores debe ser igual a la cantidad de elementos en el diccionario."
        )

    fig = go.Figure(
        go.Bar(
            x=keys,
            y=values,
            marker=dict(color=bar_colors),
            text=values,
            textposition='inside',
            texttemplate='%{text:.4f}',
            textfont=dict(color='white'),
        )
    )

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Gráfico de Barras con Colores Personalizados",
        xaxis_title="Categorías",
        yaxis_title="Valores",
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    return fig


def plot_bar_with_std(means_dict, stds_dict, bar_colors=None):
    """
    Barras con barras de error (std). means_dict y stds_dict: clave -> valor (mismo conjunto de claves).
    """
    keys = list(means_dict.keys())
    means = [means_dict[k] for k in keys]
    stds = [stds_dict.get(k, 0) for k in keys]
    if bar_colors is None:
        bar_colors = px.colors.qualitative.Plotly[: len(keys)]
    if len(bar_colors) < len(keys):
        bar_colors = (list(bar_colors) * ((len(keys) // len(bar_colors)) + 1))[: len(keys)]
    text = [f'{m:.2f} ± {s:.2f}' for m, s in zip(means, stds)]
    fig = go.Figure(
        go.Bar(
            x=keys,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            marker=dict(color=bar_colors),
            text=text,
            textposition='outside',
        )
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_title='',
        yaxis_title='',
    )
    return fig


def plot_comfort_energy_balance(
    means_comfort,
    stds_comfort,
    means_energy,
    stds_energy,
    experiment_names=None,
    color_comfort='#3498DB',
    color_energy='#E74C3C',
):
    """
    Gráfico de barras agrupadas: para cada experimento, una barra para el término de comfort
    y otra para el de energía (media ± std sobre episodios del progress.csv), para visualizar
    el equilibrio entre reward_comfort_term y reward_energy_term.

    Parámetros:
    - means_comfort, stds_comfort: dict experiment_key -> valor (media y std de mean_reward_comfort_term)
    - means_energy, stds_energy: dict experiment_key -> valor (media y std de mean_reward_energy_term)
    - experiment_names: lista opcional de nombres para el eje x (por defecto las claves)
    - color_comfort, color_energy: colores para cada término
    """
    keys = list(means_comfort.keys())
    if experiment_names is None:
        experiment_names = keys
    if len(experiment_names) != len(keys):
        experiment_names = keys
    m_c = [means_comfort[k] for k in keys]
    s_c = [stds_comfort.get(k, 0) for k in keys]
    m_e = [means_energy[k] for k in keys]
    s_e = [stds_energy.get(k, 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name='Comfort term',
            x=experiment_names,
            y=m_c,
            error_y=dict(type='data', array=s_c, visible=True),
            marker_color=color_comfort,
            text=[f'{a:.3f}' for a in m_c],
            textposition='outside',
        )
    )
    fig.add_trace(
        go.Bar(
            name='Energy term',
            x=experiment_names,
            y=m_e,
            error_y=dict(type='data', array=s_e, visible=True),
            marker_color=color_energy,
            text=[f'{a:.3f}' for a in m_e],
            textposition='outside',
        )
    )
    fig.update_layout(
        barmode='group',
        template='plotly_white',
        xaxis_title='',
        yaxis_title='Mean reward term (across episodes)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    return fig


def plot_bar_means_by_zones(df_dict, variables, zone_names, colors=None):
    """
    Barras agrupadas por zona: en el eje x las zonas (zone_names), para cada zona una barra por experimento.
    variables[i] es la columna (en cada df) correspondiente a zone_names[i].
    Solo se usan columnas que existan en cada df.
    """
    if len(variables) != len(zone_names):
        raise ValueError("variables y zone_names deben tener la misma longitud.")
    keys = list(df_dict.keys())
    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(keys)]
    if len(colors) < len(keys):
        colors = (list(colors) * ((len(keys) // len(colors)) + 1))[: len(keys)]

    fig = go.Figure()
    for i, (name, df) in enumerate(df_dict.items()):
        vals = []
        for v in variables:
            if v in df.columns:
                vals.append(df[v].mean())
            else:
                vals.append(float('nan'))
        fig.add_trace(
            go.Bar(
                name=name,
                x=zone_names,
                y=vals,
                marker_color=colors[i],
                text=[f'{x:.2f}' if pd.notna(x) else '—' for x in vals],
                textposition='outside',
            )
        )
    fig.update_layout(
        barmode='group',
        template='plotly_white',
        xaxis_title='',
        yaxis_title='Mean (°C)',
        xaxis_tickangle=-45,
    )
    return fig


def plot_bar_groups(dict_data):

    niveles = [
        'best',
        'worst',
        # 'Baseline',
        # 'Weather'
    ]
    # ppo_values = [dict_data[f'{nivel}_PPO'] for nivel in niveles]
    # sac_values = [dict_data[f'{nivel}_SAC'] for nivel in niveles]
    # case1_ppo_values = [dict_data['best_case1_PPO']]
    # baseline_values = [dict_data[niveles[-1]]]
    # weather_values = [dict_data[niveles[-1]]]
    case1_values = [dict_data['case1']]
    case2_values = [dict_data['case2']]
    balancedv1_values = [dict_data['case2_BalancedRewardV1']]
    balancedv2_values = [dict_data['case2_BalancedRewardV2']]

    # Crear gráfico de barras agrupadas
    fig = go.Figure(
        data=[
            go.Bar(
                name='Case2-PPO-BalancedRewardV1',
                x=['Case2-BalancedRewardV1'],
                y=balancedv1_values,
                marker_color='cadetblue',
                text=balancedv1_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=2,
            ),
            go.Bar(
                name='Case2-PPO-BalancedRewardV2',
                x=['Case2-PPO-BalancedRewardV2'],
                y=balancedv2_values,
                marker_color='slateblue',
                text=balancedv2_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=1,
            ),
            go.Bar(
                name='Case2-PPO-standard',
                x=['Case2-PPO-standard'],
                y=case2_values,
                marker_color='cyan',
                text=case2_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=2,
            ),
            go.Bar(
                name='Case1-PPO',
                x=['Case1-PPO'],
                y=case1_values,
                marker_color='darkseagreen',
                text=case1_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=1,
            ),
            # offset=-0.2),
            # go.Bar(
            #     name='Best-Case1-PPO',
            #     x=['Case1'],
            #     y=case1_ppo_values,
            #     marker_color='green',
            #     text=case1_ppo_values,
            #     textposition='inside',
            #     texttemplate='%{text:.4f}',
            #     offsetgroup=2),
            # offset=0.2),
            # go.Bar(
            #     name='Baseline',
            #     x=['Baseline'],
            #     y=baseline_values,
            #     marker_color='green',
            #     text=baseline_values,
            #     textposition='inside',
            #     texttemplate='%{text:.4f}',
            #     offsetgroup=1)])
            # go.Bar(
            #     name='Weather',
            #     x=['Weather'],
            #     y=weather_values,
            #     marker_color='green',
            #     text=weather_values,
            #     textposition='inside',
            #     texttemplate='%{text:.4f}',
            #     offsetgroup=1)
        ]
    )

    # Actualizar diseño del gráfico
    fig.update_layout(
        title='Comparative mean temperature violation (PPO vs Weather))',
        xaxis_title='Energy_weight (%)',
        yaxis_title='Mean temperature violation (°C)',
        barmode='group',  # Agrupar barras
        template='plotly_white',
        xaxis=dict(
            type='category',  # Usar categorías como eje X
            categoryorder='array',  # Ordenar categorías en orden definido
            categoryarray=niveles,  # Orden de categorías personalizado
        ),
    )

    return fig


def plot_bar_groups_v2(dict_data):

    niveles = ['auto', 'comfort', 'eco', 'old']
    ppo_values = [dict_data[nivel] for nivel in niveles]

    # Crear gráfico de barras agrupadas
    fig = go.Figure(
        data=[
            go.Bar(
                name='PPO',
                x=niveles,
                y=ppo_values,
                marker_color='blue',
                text=ppo_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=1,
            ),
            # offset=-0.2),
            # go.Bar(
            #     name='SAC',
            #     x=niveles,
            #     y=sac_values,
            #     marker_color='orange',
            #     text=sac_values,
            #     textposition='inside',
            #     texttemplate='%{text:.4f}',
            #     offsetgroup=2),
            # offset=0.2),
            # go.Bar(
            #     name='Baseline',
            #     x=['Baseline'],
            #     y=baseline_values,
            #     marker_color='green',
            #     text=baseline_values,
            #     textposition='inside',
            #     texttemplate='%{text:.4f}',
            #     offsetgroup=1)])
        ]
    )

    # Actualizar diseño del gráfico
    fig.update_layout(
        title='Comparative mean temperature violation (PPO vs Weather))',
        xaxis_title='Energy_weight (%)',
        yaxis_title='Mean temperature violation (°C)',
        barmode='group',  # Agrupar barras
        template='plotly_white',
        xaxis=dict(
            type='category',  # Usar categorías como eje X
            categoryorder='array',  # Ordenar categorías en orden definido
            categoryarray=niveles,  # Orden de categorías personalizado
        ),
    )

    return fig


# =============================================================================
# PLOTTING — DISTRIBUTIONS & BOXPLOTS
# =============================================================================


def _variable_name_to_axis_label(name: str) -> str:
    """Etiqueta de eje: '_' → espacio, solo la 1.ª letra en mayúscula; ``water_temperature`` añade ``(ºC)``."""
    raw = str(name)
    s = raw.replace('_', ' ').strip().lower()
    out = (s[0].upper() + s[1:]) if s else raw
    if raw == 'water_temperature':
        out = f'{out} (ºC)'
    return out


def plot_action_distribution(df_dict, variable, colors=None):
    """Distribución (violín) de una variable por experimento. df_dict: nombre -> DataFrame."""
    names = list(df_dict.keys())
    dfs = list(df_dict.values())

    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(names)]

    fig = go.Figure()
    mean_x = []
    mean_y = []
    mean_colors = []
    plotted_names = []

    for i, df in enumerate(dfs):
        if variable not in df.columns:
            continue
        series = pd.to_numeric(df[variable], errors='coerce').dropna()
        values = series.to_numpy(dtype=float)
        if values.size == 0:
            continue

        mean_v = float(np.mean(values))
        median_v = float(np.median(values))
        q1_v = float(np.quantile(values, 0.25))
        q3_v = float(np.quantile(values, 0.75))

        model_name = names[i]
        plotted_names.append(model_name)
        mean_x.append(model_name)
        mean_y.append(mean_v)
        mean_colors.append(colors[i])

        fig.add_trace(
            go.Violin(
                y=values,
                x=[model_name] * len(values),
                name=model_name,
                box_visible=True,  # mediana + cuartiles
                meanline_visible=False,  # la media se marca con un punto (Scatter)
                points='outliers',  # simple y claro; evita ruido de todos los puntos
                quartilemethod='inclusive',
                scalemode='width',
                bandwidth=0.15,
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.45,
                hovertemplate=(
                    f'Model: {model_name}'
                    f'<br>Mean: {mean_v:.4f}'
                    f'<br>Median: {median_v:.4f}'
                    f'<br>Q1: {q1_v:.4f}'
                    f'<br>Q3: {q3_v:.4f}'
                    '<extra></extra>'
                ),
            )
        )

    # Media: misma categoría X que el violín. Con violinmode='group', Plotly agrupa violín + scatter
    # en paralelo y el violín queda desplazado respecto a la etiqueta; 'overlay' centra ambos.
    if mean_x:
        fig.add_trace(
            go.Scatter(
                x=mean_x,
                y=mean_y,
                mode='markers',
                name='Mean',
                marker=dict(
                    size=10,
                    color=mean_colors,
                    symbol='circle',
                    line=dict(width=1.5, color='white'),
                ),
                hovertemplate='Model: %{x}<br>Mean: %{y:.4f}<extra></extra>',
            )
        )

    # Eje X: orden explícito (Plotly ordena categorías alfabéticamente por defecto).
    # tickvals/ticktext solo con modelos que tienen violín (evita desajuste si falta columna).
    layout_kwargs = dict(
        **PLOTLY_WHITE_LAYOUT_KWARGS,
        title=f'{variable} distribution',
        xaxis_title='Model',
        yaxis_title=_variable_name_to_axis_label(variable),
        violinmode='overlay',
        font=dict(family="Arial, sans-serif", size=20, color="black"),
    )
    if plotted_names:
        layout_kwargs['xaxis'] = dict(
            type='category',
            categoryorder='array',
            categoryarray=plotted_names,
            tickmode='array',
            tickvals=plotted_names,
            ticktext=plotted_names,
        )
    fig.update_layout(**layout_kwargs)

    return fig


def plot_dfs_boxplot(df_dict, variable, colors=None, yaxis_title=None,
        xaxis_title=None, title=None):
    """
    Gráfico de cajas para comparar una variable entre experimentos.
    df_dict: nombre -> DataFrame.
    """
    names = list(df_dict.keys())
    dfs = list(df_dict.values())

    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(dfs)]

    fig = go.Figure()
    for df, name, color in zip(dfs, names, colors):
        if variable not in df.columns:
            continue
        fig.add_trace(
            go.Box(
                y=df[variable],
                name=name,
                marker_color=color,
                boxmean=False,
            )
        )

    # Configurar el layout
    fig.update_layout(
        title=None,
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        showlegend=False,  # No mostrar la leyenda, los nombres están en el eje X
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        width=1000,  # Ancho de la figura
        height=600,  # Alto de la figura
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


# =============================================================================
# PLOTTING — ENERGY SAVINGS
# =============================================================================


def plot_energy_savings(data, names_reference, names_comparison, variable, colors=None):
    """
    Ahorro energético (%) por mes: comparación reference vs comparison (data es dict nombre -> DataFrame).

    Parámetros:
    - dfs_reference: lista de DataFrames de referencia
    - dfs_comparison: lista de DataFrames a comparar contra los de referencia
    - names_reference: lista de nombres para cada DataFrame de referencia
    - names_comparison: lista de nombres para cada DataFrame de comparación
    - variable: nombre de la columna en los DataFrames que se desea comparar
    - colors: lista opcional de colores para cada barra en el gráfico
    """
    # Asignar colores predeterminados si no se proporcionan
    if colors is None:
        colors = px.colors.qualitative.Plotly[
            : len(names_reference) * len(names_comparison)
        ]

    # Crear la figura
    fig = go.Figure()

    # Sacamos los dfs de cada uno
    dfs_reference = []
    for name_reference in names_reference:
        dfs_reference.append(data[name_reference])
    dfs_comparison = []
    for name_comparison in names_comparison:
        dfs_comparison.append(data[name_comparison])

    # Preprocesar DataFrames: calcular medias mensuales y añadir 'year_month'
    # como índice
    ref_means = [
        df.assign(year_month=df['datetime'].dt.to_period('M'))
        .groupby('year_month')[variable]
        .mean()
        for df in dfs_reference
    ]
    comp_means = [
        df.assign(year_month=df['datetime'].dt.to_period('M'))
        .groupby('year_month')[variable]
        .mean()
        for df in dfs_comparison
    ]

    # Iterar sobre pares de referencia y comparación
    count = 0
    for ref_avg, name_ref in zip(ref_means, names_reference):
        for comp_avg, name_comp in zip(comp_means, names_comparison):
            # Alinear índices para evitar errores y calcular ahorro energético
            aligned_ref, aligned_comp = ref_avg.align(comp_avg, join='inner')

            # Calcular ahorro energético en porcentaje
            energy_savings = (1 - aligned_comp / aligned_ref) * 100

            # Agregar traza al gráfico
            fig.add_trace(
                go.Bar(
                    x=aligned_ref.index.to_timestamp(),
                    y=energy_savings,
                    name=f'{name_comp} vs {name_ref}',
                    marker_color=colors[count],
                    text=energy_savings,
                    textposition='inside',
                    texttemplate='%{text:.2f}',
                )
            )
            count += 1

    # Configurar el eje X y diseño del gráfico
    x_values = ref_means[0].index.to_timestamp()
    fig.update_xaxes(
        type='date',
        tickvals=x_values,
        ticklabelposition='outside',
        **_DATETIME_X_AXIS_FORMAT,
    )
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title='Energy Savings (%)',
        yaxis=dict(tick0=0, dtick=2),
        barmode='group',
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
            itemwidth=60,
        ),
        height=600,
        width=1000,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    # Devolver la figura
    return fig


def plot_mean_energy_savings(
    data, names_reference, names_comparison, variable, colors=None
):
    """
    Calcula y grafica el ahorro energético medio (%) de cada comparación frente a cada referencia,
    usando la misma convención que plot_energy_savings (names_reference / names_comparison).

    Parámetros:
    - data: dict nombre -> DataFrame
    - names_reference: nombres de los DataFrames de referencia
    - names_comparison: nombres de los DataFrames a comparar
    - variable: nombre de la columna a promediar (p. ej. 'heat_source_electricity_rate')
    - colors: lista opcional de colores para las barras (una por par ref/comp)
    """
    dfs_reference = [data[name] for name in names_reference]
    dfs_comparison = [data[name] for name in names_comparison]

    ref_means = [df[variable].mean() for df in dfs_reference]
    comp_means = [df[variable].mean() for df in dfs_comparison]

    savings_dict = {}
    for ref_mean, name_ref in zip(ref_means, names_reference):
        for comp_mean, name_comp in zip(comp_means, names_comparison):
            if ref_mean == 0:
                savings_pct = 0.0
            else:
                savings_pct = 100 * (ref_mean - comp_mean) / ref_mean
            label = (
                name_comp if len(names_reference) == 1 else f'{name_comp} vs {name_ref}'
            )
            savings_dict[label] = savings_pct

    n_bars = len(savings_dict)
    if colors is None:
        colors = px.colors.qualitative.Plotly[:n_bars]
    elif len(colors) < n_bars:
        colors = (colors * ((n_bars // len(colors)) + 1))[:n_bars]

    fig = plot_bar(savings_dict, bar_colors=colors)
    ref_label = names_reference[0] if len(names_reference) == 1 else 'reference'
    fig.update_layout(
        title=f'Comparative mean energy consumption savings vs {ref_label}',
        xaxis_title='',
        yaxis_title='Mean energy consumption savings (%)',
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )
    return fig


def plot_summary_data(data):

    # Preparar los datos para las subgráficas
    categories = list(data['without_weather'].keys())
    without_weather_values = list(data['without_weather'].values())
    with_weather_values = list(data['with_weather'].values())

    # Calcular número de filas necesarias
    num_cols = 2
    num_rows = -(-len(categories) // num_cols)  # Redondear hacia arriba

    # Crear subgráficas
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=categories)

    for i, category in enumerate(categories):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.add_trace(
            go.Bar(
                x=['Without Weather', 'With Weather'],
                y=[without_weather_values[i], with_weather_values[i]],
                name=category,
                marker_color=['blue', 'orange'],
                text=[without_weather_values[i], with_weather_values[i]],
                textposition='inside',
                texttemplate='%{text:.4f}',
            ),
            row=row,
            col=col,
        )

    # Personalización del diseño
    fig.update_layout(
        title='Comparison of Metrics: With vs Without Weather',
        height=400 * num_rows,
        showlegend=False,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )

    return fig


# PLOT TEMP BY ZONES WITH ORANGE BAND
# =============================================================================

def _obs_x_values(obs_data: pd.DataFrame):
    """Time axis for temperature helpers: ``datetime`` column if present, else index."""
    if "datetime" in obs_data.columns:
        return obs_data["datetime"]
    return pd.Series(obs_data.index, index=obs_data.index)


def _outdoor_yaxis2_layout():
    """Right Y axis for outdoor temperature (same idea as :func:`plot_temperature_one_zone`)."""
    return dict(
        title=dict(text="Outdoor temperature (°C)", font=dict(color="gray")),
        overlaying="y",
        side="right",
        showgrid=False,
        tickfont=dict(color="gray"),
    )


def add_temperature_traces(
    fig,
    obs_data,
    temp_col,
    sp_col,
    show_legend=True,
    row=None,
    col=None,
    threshold=1.0,
    temp_color=None,
    outdoor_temp_var="outdoor_temperature",
):
    """Add comfort band + indoor temp (in/out of band) + optional outdoor series.

    Parameters match :func:`plot_temperature_one_zone` where applicable
    (``threshold``, ``temp_color``, ``outdoor_temp_var``). Uses ``datetime`` column
    if present, otherwise the DataFrame index as X.

    Returns
    -------
    bool
        True if an outdoor trace was added (caller may need ``yaxis2`` on a single
        ``go.Figure``, or ``secondary_y`` axes on subplots are updated separately).
    """
    if sp_col not in obs_data.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{sp_col}'.")
    if temp_col not in obs_data.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{temp_col}'.")

    x_series = _obs_x_values(obs_data)
    temp = obs_data[temp_col].to_numpy()
    sp = obs_data[sp_col].to_numpy()
    x_vals = x_series.to_numpy()

    sp_upper = sp + threshold
    sp_lower = sp - threshold
    in_comfort = (temp >= sp_lower) & (temp <= sp_upper)

    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}

    band_label = f"Setpoint ±{threshold:g}°C"

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=sp_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        **trace_kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=sp_lower,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(255, 165, 0, 0.2)",
            fill="tonexty",
            name=band_label if show_legend else None,
            showlegend=show_legend,
            hovertemplate="Setpoint band<extra></extra>",
        ),
        **trace_kwargs,
    )

    segments_in = []
    segments_out = []
    current_segment_in = {"x": [], "y": []}
    current_segment_out = {"x": [], "y": []}

    for i in range(len(temp)):
        if in_comfort[i]:
            current_segment_in["x"].append(x_vals[i])
            current_segment_in["y"].append(temp[i])
            if len(current_segment_out["x"]) > 0:
                segments_out.append(current_segment_out.copy())
                current_segment_out = {"x": [], "y": []}
                current_segment_out["x"].append(x_vals[i])
                current_segment_out["y"].append(temp[i])
        else:
            current_segment_out["x"].append(x_vals[i])
            current_segment_out["y"].append(temp[i])
            if len(current_segment_in["x"]) > 0:
                segments_in.append(current_segment_in.copy())
                current_segment_in = {"x": [], "y": []}
                current_segment_in["x"].append(x_vals[i])
                current_segment_in["y"].append(temp[i])

    if len(current_segment_in["x"]) > 0:
        segments_in.append(current_segment_in)
    if len(current_segment_out["x"]) > 0:
        segments_out.append(current_segment_out)

    in_line_color = temp_color if temp_color is not None else "#1f77b4"

    for i, seg in enumerate(segments_in):
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode="lines",
                name="Indoor temp (in comfort)" if (show_legend and i == 0) else None,
                showlegend=(show_legend and i == 0),
                line=dict(color=in_line_color, width=1.5),
                hovertemplate="Indoor: %{y:.2f}°C<extra></extra>",
                legendgroup="indoor_in",
            ),
            **trace_kwargs,
        )

    for i, seg in enumerate(segments_out):
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode="lines",
                name=(
                    "Indoor temp (out of comfort)" if (show_legend and i == 0) else None
                ),
                showlegend=(show_legend and i == 0),
                line=dict(color="#d62728", width=1.5),
                hovertemplate="Indoor: %{y:.2f}°C (OUT OF COMFORT)<extra></extra>",
                legendgroup="indoor_out",
            ),
            **trace_kwargs,
        )

    has_outdoor = (
        outdoor_temp_var is not None and outdoor_temp_var in obs_data.columns
    )
    if has_outdoor:
        outdoor = obs_data[outdoor_temp_var].to_numpy()
        scatter_kwargs = dict(
            x=x_vals,
            y=outdoor,
            mode="lines",
            name="Outdoor temperature",
            line=dict(color="gray", width=1.2, dash="dashdot"),
            opacity=0.8,
        )
        if row is None and col is None:
            scatter_kwargs["yaxis"] = "y2"
            fig.add_trace(go.Scatter(**scatter_kwargs))
        else:
            fig.add_trace(
                go.Scatter(**scatter_kwargs),
                **trace_kwargs,
                secondary_y=True,
            )
    return has_outdoor


def _zone_output_slug(zone_name: str) -> str:
    return "_".join(zone_name.lower().replace("-", " ").split())


def _export_plotly_figure(
    fig: go.Figure,
    path_stem: Path,
    export_format: Literal["html", "png"],
    *,
    png_width: int,
    png_height: int,
    png_scale: int = 2,
) -> None:
    """Write ``path_stem`` + ``.html`` or ``.png`` (requires kaleido for PNG)."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=png_width, height=png_height, **PLOTLY_WHITE_LAYOUT_KWARGS)
    if export_format == "html":
        fig.write_html(str(path_stem.with_suffix(".html")))
        return
    try:
        fig.write_image(
            str(path_stem.with_suffix(".png")),
            width=png_width,
            height=png_height,
            scale=png_scale,
        )
    except Exception as e:
        print(f"⚠️ No se pudo exportar PNG ({path_stem.name}): {e}")


def plot_case_temperatures(
    df: pd.DataFrame,
    zones: Sequence[Tuple[str, str, str]],
    output_dir: Path,
    daily_date: pd.Timestamp,
    case_id: int = 0,
    summary_title: str = "",
    threshold: float = 1.0,
    temp_colors: Optional[Sequence[Optional[str]]] = None,
    outdoor_temp_var: Optional[str] = "outdoor_temperature",
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
    export_format: Literal["html", "png"] = "png",
    png_width: int = 1200,
    png_height_single: int = 500,
    png_scale: int = 2,
) -> None:
    """Grid + per-zone time views: export as PNG (default) or interactive HTML.

    Same data/comfort/outdoor conventions as :func:`plot_temperature_one_zone`.
    ``zones`` is a sequence of ``(temp_var, setpoint_var, zone_name)`` per room.
    ``df`` must include ``datetime`` and all columns referenced by ``zones`` (and
    optionally ``outdoor_temp_var``).
    """
    if "datetime" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'datetime'.")

    for temp_var, setpoint_var, _ in zones:
        if setpoint_var not in df.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{setpoint_var}'.")
        if temp_var not in df.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{temp_var}'.")

    df_work = _ensure_datetime_unique(df.copy())

    if period_start is None:
        period_start = datetime(2026, 11, 15)
    if period_end is None:
        period_end = datetime(2027, 3, 15, 23, 55)
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    mask = (df_work["datetime"] >= start_ts) & (df_work["datetime"] <= end_ts)
    obs = df_work.loc[mask].copy()

    daily_date_norm = pd.Timestamp(daily_date).normalize()
    daily_mask = obs["datetime"].dt.normalize() == daily_date_norm
    obs_daily = obs.loc[daily_mask]

    week_start = daily_date_norm
    week_end = week_start + pd.Timedelta(days=6)
    week_mask = (obs["datetime"] >= week_start) & (obs["datetime"] <= week_end)
    obs_week = obs.loc[week_mask]

    month_start = daily_date_norm.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_mask = (obs["datetime"] >= month_start) & (obs["datetime"] <= month_end)
    obs_month = obs.loc[month_mask]

    n_zones = len(zones)
    if n_zones == 0:
        return

    ncols = 2
    nrows = (n_zones + ncols - 1) // ncols
    n_cells = nrows * ncols
    subplot_titles = [z[2] for z in zones] + [""] * (n_cells - n_zones)

    has_outdoor = (
        outdoor_temp_var is not None and outdoor_temp_var in obs.columns
    )
    specs = [
        [{"secondary_y": bool(has_outdoor)} for _ in range(ncols)]
        for _ in range(nrows)
    ]
    fig = sp.make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    if temp_colors is None:
        colors_seq: List[Optional[str]] = [None] * n_zones
    else:
        colors_seq = list(temp_colors)
        if len(colors_seq) < n_zones:
            colors_seq.extend([None] * (n_zones - len(colors_seq)))


    for i, (temp_col, sp_col, _zone_title) in enumerate(zones):
        row = (i // ncols) + 1
        col = (i % ncols) + 1
        add_temperature_traces(
            fig,
            obs,
            temp_col,
            sp_col,
            show_legend=(i == 0),
            row=row,
            col=col,
            threshold=threshold,
            temp_color=colors_seq[i],
            outdoor_temp_var=outdoor_temp_var,
        )
        if has_outdoor:
            fig.update_yaxes(
                title=dict(text="Outdoor (°C)", font=dict(color="gray")),
                row=row,
                col=col,
                secondary_y=True,
                showgrid=False,
                tickfont=dict(color="gray"),
            )

    # Eje X: solo mes (sin año); mismo criterio en figuras sueltas más abajo.
    _xaxis_month_only = dict(tickformat="%b")
    fig.update_xaxes(**_xaxis_month_only)

    grid_title = summary_title.strip() or f"case{case_id}"
    grid_height = max(300 * nrows, 600)
    # width/height en layout: Kaleido usa las mismas dimensiones que write_image;
    # si solo se pasan a write_image, la anotación del eje Y puede solaparse con los ticks.
    fig.update_layout(
        title=None,
        width=png_width,
        height=grid_height,
        template="plotly_white",
        hovermode=False,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=130, r=25, b=40),
    )
    # Una sola etiqueta de eje Y para la rejilla (temperatura interior); evita repetir en cada zona.
    fig.add_annotation(
        text="Temperature (°C)",
        xref="paper",
        yref="paper",
        x=-0.082,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        showarrow=False,
        font=dict(family="Arial, sans-serif", size=20, color="black"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _export_plotly_figure(
        fig,
        output_dir / f"case{case_id}_temperatures",
        export_format,
        png_width=png_width,
        png_height=grid_height,
        png_scale=png_scale,
    )

    for i, (temp_col, sp_col, room_title) in enumerate(zones):
        room_slug = _zone_output_slug(room_title)
        room_dir = output_dir / f"case{case_id}" / room_slug
        room_dir.mkdir(parents=True, exist_ok=True)

        fig_r = go.Figure()
        added_out = add_temperature_traces(
            fig_r,
            obs,
            temp_col,
            sp_col,
            show_legend=True,
            threshold=threshold,
            temp_color=colors_seq[i],
            outdoor_temp_var=outdoor_temp_var,
        )
        layout_updates = dict(
            title=f"{grid_title} – {room_title}",
            yaxis_title="Temperature (°C)",
            xaxis=_xaxis_month_only,
            font=dict(family="Arial, sans-serif", size=20, color="black"),
            template="plotly_white",
            height=500,
            hovermode=False,
            #xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )
        if added_out:
            layout_updates["yaxis2"] = _outdoor_yaxis2_layout()
        fig_r.update_layout(**layout_updates)

        _export_plotly_figure(
            fig_r,
            room_dir / "temperature",
            export_format,
            png_width=png_width,
            png_height=png_height_single,
            png_scale=png_scale,
        )

        if not obs_daily.empty:
            fig_d = go.Figure()
            added_od = add_temperature_traces(
                fig_d,
                obs_daily,
                temp_col,
                sp_col,
                show_legend=True,
                threshold=threshold,
                temp_color=colors_seq[i],
                outdoor_temp_var=outdoor_temp_var,
            )
            ld = dict(
                title=f"{grid_title} – {room_title} (daily: {daily_date.date()})",
                yaxis_title="Temperature (°C)",
                xaxis=_xaxis_month_only,
                template="plotly_white",
                font=dict(family="Arial, sans-serif", size=20, color="black"),
                height=500,
                hovermode=False,
            )
            if added_od:
                ld["yaxis2"] = _outdoor_yaxis2_layout()
            fig_d.update_layout(**ld)
            _export_plotly_figure(
                fig_d,
                room_dir / "daily_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
            )

        if not obs_week.empty:
            fig_w = go.Figure()
            added_ow = add_temperature_traces(
                fig_w,
                obs_week,
                temp_col,
                sp_col,
                show_legend=True,
                threshold=threshold,
                temp_color=colors_seq[i],
                outdoor_temp_var=outdoor_temp_var,
            )
            lw = dict(
                title=(
                    f"{grid_title} – {room_title} (weekly: "
                    f"{week_start.date()} to {week_end.date()})"
                ),
                yaxis_title="Temperature (°C)",
                xaxis=_xaxis_month_only,
                template="plotly_white",
                font=dict(family="Arial, sans-serif", size=20, color="black"),
                height=500,
                hovermode=False,
            )
            if added_ow:
                lw["yaxis2"] = _outdoor_yaxis2_layout()
            fig_w.update_layout(**lw)
            _export_plotly_figure(
                fig_w,
                room_dir / "weekly_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
            )

        if not obs_month.empty:
            fig_m = go.Figure()
            added_om = add_temperature_traces(
                fig_m,
                obs_month,
                temp_col,
                sp_col,
                show_legend=True,
                threshold=threshold,
                temp_color=colors_seq[i],
                outdoor_temp_var=outdoor_temp_var,
            )
            lm = dict(
                title=(
                    f"{grid_title} – {room_title} (monthly: "
                    f"{month_start.date()} to {month_end.date()})"
                ),
                yaxis_title="Temperature (°C)",
                xaxis=_xaxis_month_only,
                template="plotly_white",
                height=500,
                hovermode=False,
                font=dict(family="Arial, sans-serif", size=20, color="black"),
                #xaxis=dict(rangeslider=dict(visible=True), type="date"),
            )
            if added_om:
                lm["yaxis2"] = _outdoor_yaxis2_layout()
            fig_m.update_layout(**lm)
            _export_plotly_figure(
                fig_m,
                room_dir / "monthly_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
            )