import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
try:
    from PIL import Image, ImageChops
except Exception:
    Image = None
    ImageChops = None

# Fondo blanco unificado (Plotly por defecto suele usar la plantilla "plotly", más oscura).
pio.templates.default = "plotly_white"

# Para merge en ``update_layout`` / exportación (también importable desde otros scripts).
PLOTLY_WHITE_LAYOUT_KWARGS = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
)

# --- Estilo paper (equiv. sample_plots: seaborn-v0_8-whitegrid + serif rcParams) ---
_PLOTLY_SERIF = (
    'DejaVu Serif, Bitstream Vera Serif, Georgia, Times New Roman, serif'
)
# Casi negro para impresión / PDF de artículo
_PLOTLY_TEXT = '#0a0a0a'
_PLOTLY_GRID = '#cccccc'
PLOTLY_TRACE_LINE_WIDTH = 1.8

# Banda setpoint ± umbral (temperaturas): verde muy suave; líneas = mismo RGB que el tono del relleno.
_COMFORT_BAND_RGB = (120, 185, 120)
_COMFORT_BAND_FILLCOLOR = (
    f'rgba({_COMFORT_BAND_RGB[0]}, {_COMFORT_BAND_RGB[1]}, {_COMFORT_BAND_RGB[2]}, 0.2)'
)
_COMFORT_BAND_LINE_COLOR = (
    f'rgb({_COMFORT_BAND_RGB[0]}, {_COMFORT_BAND_RGB[1]}, {_COMFORT_BAND_RGB[2]})'
)

# Tamaños coherentes en todas las facetas (leyenda, ejes, títulos)
PLOTLY_PAPER_FONT_SIZE = 15
PLOTLY_PAPER_TICK_SIZE = 14
PLOTLY_PAPER_AXIS_TITLE_SIZE = 15
PLOTLY_PAPER_TITLE_SIZE = 17
PLOTLY_PAPER_LEGEND_SIZE = 14

# make_subplots: menos hueco entre paneles (Plotly no tiene tight_layout; dominios normalizados 0–1).
PLOTLY_SUBPLOT_VERTICAL_SPACING = 0.06
# Rejilla ``plot_case_temperatures`` (una columna): paneles más pegados (cf. sample_plots/evolution.py).
PLOTLY_ZONE_TEMP_GRID_VERTICAL_SPACING = 0.02
PLOTLY_SUBPLOT_HORIZONTAL_SPACING = 0.055

PLOTLY_PAPER_STYLE_AXIS = dict(
    showgrid=True,
    gridcolor=_PLOTLY_GRID,
    gridwidth=1,
    griddash='solid',
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor=_PLOTLY_GRID,
    mirror=False,
    ticks='outside',
    ticklen=0,
    tickcolor=_PLOTLY_TEXT,
    tickfont=dict(family=_PLOTLY_SERIF, size=PLOTLY_PAPER_TICK_SIZE, color=_PLOTLY_TEXT),
    title=dict(
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_AXIS_TITLE_SIZE,
            color=_PLOTLY_TEXT,
        )
    ),
)

PLOTLY_PAPER_STYLE_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family=_PLOTLY_SERIF, size=PLOTLY_PAPER_FONT_SIZE, color=_PLOTLY_TEXT),
    title=dict(
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_TITLE_SIZE,
            color=_PLOTLY_TEXT,
        )
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        borderwidth=0,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_LEGEND_SIZE,
            color=_PLOTLY_TEXT,
        ),
    ),
)


def apply_plotly_paper_style(
    fig: go.Figure,
    *,
    line_width: Optional[float] = None,
) -> go.Figure:
    """Misma estética que ``sample_plots/boxplots.py`` (matplotlib) aplicada a Plotly."""
    lw = PLOTLY_TRACE_LINE_WIDTH if line_width is None else line_width
    fig.update_layout(**PLOTLY_PAPER_STYLE_LAYOUT)
    fig.update_xaxes(**PLOTLY_PAPER_STYLE_AXIS)
    fig.update_yaxes(**PLOTLY_PAPER_STYLE_AXIS)
    try:
        fig.update_yaxes(
            showgrid=False,
            linewidth=1,
            linecolor=_PLOTLY_GRID,
            tickcolor='gray',
            tickfont=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_TICK_SIZE,
                color='gray',
            ),
            title=dict(
                font=dict(
                    family=_PLOTLY_SERIF,
                    size=PLOTLY_PAPER_AXIS_TITLE_SIZE,
                    color='gray',
                )
            ),
            secondary_y=True,
        )
    except Exception:
        pass
    fig.update_traces(
        line=dict(width=lw),
        selector=dict(type='scatter'),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """``#RRGGBB`` → ``rgba(r,g,b,alpha)`` para rellenos alineados con el color de línea."""
    h = str(hex_color).lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


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


def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
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


def filer_interval(df: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    # Los stubs de pandas suelen devolver DataFrame | Series; aquí es siempre DataFrame.
    return cast(pd.DataFrame, df.loc[mask])


def resample(df):
    output = df.resample('1h', on='datetime').mean()
    output.reset_index(inplace=True)
    return output


# =============================================================================
# FIGURE EXPORT (high quality PNG + HTML)
# =============================================================================


def _autocrop_png_whitespace(
    png_path: Path,
    *,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
    pad_px: int = 0,
) -> bool:
    """Recorta bordes blancos de un PNG exportado (si Pillow está disponible)."""
    if Image is None or ImageChops is None:
        return False
    path = Path(png_path)
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im_rgb = im.convert('RGB')
            bg = Image.new('RGB', im_rgb.size, background_rgb)
            diff = ImageChops.difference(im_rgb, bg)
            bbox = diff.getbbox()
            if bbox is None:
                return False
            l, t, r, b = bbox
            if pad_px > 0:
                l = max(0, l - pad_px)
                t = max(0, t - pad_px)
                r = min(im.width, r + pad_px)
                b = min(im.height, b + pad_px)
            if (l, t, r, b) == (0, 0, im.width, im.height):
                return False
            im.crop((l, t, r, b)).save(path)
        return True
    except Exception:
        return False


def save_figure(
    fig,
    path_stem,
    width=1200,
    height=700,
    scale=2,
    paper_style=True,
    autocrop_png=True,
):
    """
    Guarda la figura en PNG (alta calidad) y HTML.
    path_stem: Path o str sin extensión (ej. output_dir / 'nombre').
    ``width``/``height`` se aplican también a ``layout`` para que el HTML y el PNG
    (Kaleido) compartan la misma geometría y no diverjan márgenes o ejes.
    ``paper_style``: serif + rejilla gris claro (equiv. sample_plots matplotlib).
    ``autocrop_png``: recorta bordes blancos del PNG (equiv. aproximado a tight bbox).
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=width, height=height, **PLOTLY_WHITE_LAYOUT_KWARGS)
    if paper_style:
        apply_plotly_paper_style(fig)
    png_path = path_stem.with_suffix('.png')
    try:
        fig.write_image(
            str(png_path),
            width=width,
            height=height,
            scale=scale,
        )
        if autocrop_png:
            _autocrop_png_whitespace(png_path)
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
_MARKER_SYMBOL_CYCLE = (
    'circle',
    'square',
    'diamond',
    'triangle-up',
    'triangle-down',
    'x',
    'cross',
    'pentagon',
    'star',
)


def plot_dfs_line(
    df_dict,
    variable_name,
    colors=None,
    line_styles=None,
    marker_symbols=None,
    marker_every=None,
):
    """
    Genera un gráfico de líneas de progreso.

    Parámetros:
    - dfs: lista de DataFrames que contienen los datos de progreso
    - names: lista de nombres para cada línea
    - variable_name: nombre de la columna en los DataFrames que se desea graficar en el eje y
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    - line_styles: lista opcional de dash de Plotly por traza; si es None, se cicla _LINE_DASH_CYCLE.
    - marker_symbols: lista opcional de símbolos de marcador por traza.
    - marker_every: separación aproximada entre marcadores (en puntos). Si None, se auto-ajusta.
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
    if marker_symbols is None:
        marker_symbols = [_MARKER_SYMBOL_CYCLE[i % len(_MARKER_SYMBOL_CYCLE)] for i in range(n)]
    elif len(marker_symbols) < n:
        ms = list(marker_symbols)
        marker_symbols = (ms * ((n + len(ms) - 1) // len(ms)))[:n]

    # Crear figura
    fig = go.Figure()

    # Añadir líneas para cada DataFrame
    for (name, df), color, dash, symbol in zip(
        df_dict.items(), colors, line_styles, marker_symbols
    ):
        x_vals = pd.to_numeric(df['episode_num'], errors='coerce')
        y_vals = pd.to_numeric(df[variable_name], errors='coerce')
        valid = x_vals.notna() & y_vals.notna()
        x_vals = x_vals.loc[valid]
        y_vals = y_vals.loc[valid]
        if x_vals.empty:
            continue

        line_kw = dict(color=color, width=2)
        if dash is not None and str(dash).lower() != 'solid':
            line_kw['dash'] = dash
        if marker_every is None:
            step = max(1, int(math.ceil(len(x_vals) / 40)))
        else:
            step = max(1, int(marker_every))
        maxdisplayed = max(1, int(math.ceil(len(x_vals) / step)))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=line_kw,
                marker=dict(
                    symbol=symbol,
                    size=6,
                    maxdisplayed=maxdisplayed,
                    color=color,
                    line=dict(width=0),
                ),
                name=name,
                legendgroup=name,
                hovertemplate=(
                    f'{name}'
                    '<br>Episode: %{x}'
                    '<br>Value: %{y:.4f}'
                    '<extra></extra>'
                ),
            )
        )

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title='Datetime',
        yaxis_title=None,
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
                fillcolor=_hex_to_rgba(color_comfort, 0.15),
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
                fillcolor=_hex_to_rgba(color_energy, 0.15),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
        xaxis_title='',
        yaxis_title='Reward term (per timestep)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis=dict(**_DATETIME_X_AXIS_FORMAT),
        height=500,
        width=1000,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_FONT_SIZE,
            color=_PLOTLY_TEXT,
        ),
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

    # Eje Y temperatura: rango a partir de lecturas plausibles (evita picos sensor / datos erróneos)
    _temp_y_parts: list[pd.Series] = []
    for var in temperature_variables:
        if var not in df.columns:
            continue
        t_ok = _plausible_indoor_air_c(df[var])
        _temp_y_parts.append(t_ok if not t_ok.empty else df[var].astype(float).dropna())
    if outdoor_temp_var is not None and outdoor_temp_var in df.columns:
        o_ok = _plausible_indoor_air_c(df[outdoor_temp_var])
        _temp_y_parts.append(
            o_ok if not o_ok.empty else df[outdoor_temp_var].astype(float).dropna()
        )
    yaxis_primary: dict = dict(title='Temperature (°C)', side='left')
    if _temp_y_parts:
        merged_t = pd.concat(_temp_y_parts, ignore_index=True)
        arr_t = merged_t.to_numpy(dtype=float)
        arr_t = arr_t[np.isfinite(arr_t)]
        if arr_t.size > 0:
            _pad = 0.5
            _lo = float(np.nanmin(arr_t)) - _pad
            _hi = float(np.nanmax(arr_t)) + _pad
            if math.isfinite(_lo) and math.isfinite(_hi) and _hi > _lo:
                yaxis_primary['range'] = [_lo, _hi]

    # Configurar ejes: X con datetime explícito
    fig.update_layout(
        xaxis=dict(
            title='Datetime',
            type='date',
            **_DATETIME_X_AXIS_FORMAT,
        ),
        yaxis=yaxis_primary,
        yaxis2=dict(title='Flow rate', overlaying='y', side='right', showgrid=False),
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
    show_legend=True,
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
    _layout = dict(
        title=title,
        xaxis_title='',
        yaxis_title=yaxis_title if yaxis_title is not None else variable,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_FONT_SIZE,
            color=_PLOTLY_TEXT,
        ),
        showlegend=show_legend,
        **PLOTLY_WHITE_LAYOUT_KWARGS,
    )
    if show_legend:
        _layout['legend'] = dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        )
    fig.update_layout(**_layout)
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
        ),
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
        ),
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_FONT_SIZE,
            color=_PLOTLY_TEXT,
        ),
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
        vertical_spacing=PLOTLY_SUBPLOT_VERTICAL_SPACING,
        horizontal_spacing=PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
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
        height=max(220 * nrows, 400),
        width=1000,
        showlegend=False,
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
    """Etiqueta de eje: '_' → espacio, solo la 1.ª letra en mayúscula; ``flow_rate*`` → Flow rate (m3/h); ``water_temperature`` añade ``(ºC)``."""
    raw = str(name)
    if raw.startswith('flow_rate'):
        return 'Flow rate (m³/h)'
    if raw == 'water_temperature':
        return 'Temperature (ºC)'


def _violin_fill_rgba(line_color: str, alpha: float = 0.75) -> str:
    """Relleno tipo ``sample_plots/violins.py`` (alpha ~0.75 sobre tab)."""
    h = str(line_color).lstrip('#')
    if len(h) != 6:
        return _hex_to_rgba('#1f77b4', alpha)
    return _hex_to_rgba(f'#{h}', alpha)


def _violin_y_range_from_arrays(arrays: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Rango del eje Y acotado a los datos + margen (evita escalas desacopladas del KDE)."""
    parts = [
        np.asarray(a, dtype=float)
        for a in arrays
        if a is not None and getattr(a, 'size', 0) > 0
    ]
    if not parts:
        return 0.0, 1.0
    stack = np.concatenate(parts)
    stack = stack[np.isfinite(stack)]
    if stack.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(stack)), float(np.max(stack))
    span = hi - lo
    if span <= 0:
        delta = max(abs(lo) * 0.05, 1e-9)
        return lo - delta, hi + delta
    pad = span * 0.06
    return lo - pad, hi + pad


def plot_action_distribution(df_dict, variable, colors=None):
    """Distribución (violín) por experimento. Estética alineada con ``sample_plots/violins.py``."""
    names = list(df_dict.keys())
    dfs = list(df_dict.values())

    if colors is None:
        colors = px.colors.qualitative.Plotly[: len(names)]

    fig = go.Figure()
    plotted_names: List[str] = []
    value_arrays: List[np.ndarray] = []

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
        value_arrays.append(values)

        fill_rgba = _violin_fill_rgba(colors[i], 0.75)

        fig.add_trace(
            go.Violin(
                y=values,
                x=[model_name] * len(values),
                name=model_name,
                scalegroup='action_distribution',
                # Equiv. matplotlib: violinplot(showmeans=True, showextrema=False): sin caja, media negra.
                box_visible=False,
                points=False,
                quartilemethod='inclusive',
                scalemode='width',
                side='both',
                meanline_visible=True,
                meanline=dict(color=_PLOTLY_TEXT, width=2),
                line=dict(color=_PLOTLY_TEXT, width=1),
                fillcolor=fill_rgba,
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

    layout_kwargs: dict = dict(
        **PLOTLY_WHITE_LAYOUT_KWARGS,
        title=None,
        xaxis_title='',
        yaxis_title=_variable_name_to_axis_label(variable),
        violinmode='overlay',
        showlegend=False,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_FONT_SIZE,
            color=_PLOTLY_TEXT,
        ),
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
    if value_arrays:
        y0, y1 = _violin_y_range_from_arrays(value_arrays)
        yaxis_cfg: dict = dict(range=[y0, y1], autorange=False)
        if variable == 'water_temperature':
            yaxis_cfg['dtick'] = 2.5
        layout_kwargs['yaxis'] = yaxis_cfg

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
                fillcolor=color,
                marker=dict(color=color),
                boxmean=False,
                line=dict(color='black', width=1),
            )
        )

    # Configurar el layout
    fig.update_layout(
        title=None,
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        showlegend=False,  # No mostrar la leyenda, los nombres están en el eje X
        font=dict(family=_PLOTLY_SERIF, size=PLOTLY_PAPER_FONT_SIZE, color=_PLOTLY_TEXT),
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
        font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
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
    fig = sp.make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=categories,
        vertical_spacing=PLOTLY_SUBPLOT_VERTICAL_SPACING,
        horizontal_spacing=PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
    )

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
        height=max(320 * num_rows, 360),
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


def _plausible_indoor_air_c(series: pd.Series, *, lo_c: float = 5.0, hi_c: float = 45.0) -> pd.Series:
    """Filtra lecturas de temperatura/setpoint interior claramente fuera de rango (spikes)."""
    s = series.astype(float)
    return s[(s >= lo_c) & (s <= hi_c)]


def _indoor_temperature_y_range(
    obs_data: pd.DataFrame,
    temp_col: str,
    sp_col: str,
    threshold: float,
    *,
    pad_c: float = 0.5,
) -> tuple[float, float]:
    """Rango del eje Y (°C) a partir de temperatura interior, setpoint y banda.

    Ignora valores fuera de un rango físico razonable para aire interior (p. ej.
    picos erróneos del sensor o unidades mal interpretadas) al calcular min/max,
    para no aplastar la escala del eje.
    """
    if obs_data.empty or temp_col not in obs_data.columns or sp_col not in obs_data.columns:
        return (15.0, 28.0)
    t = obs_data[temp_col].astype(float)
    s = obs_data[sp_col].astype(float)
    t_ok = _plausible_indoor_air_c(t)
    s_ok = _plausible_indoor_air_c(s)
    if t_ok.empty:
        t_ok = t.dropna()
    if s_ok.empty:
        s_ok = s.dropna()
    vals = pd.concat(
        [t_ok, s_ok, s_ok + threshold, s_ok - threshold],
        ignore_index=True,
    )
    arr = vals.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (15.0, 28.0)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return (15.0, 28.0)
    lo -= pad_c
    hi += pad_c
    if hi <= lo:
        lo -= 1.0
        hi += 1.0
    return (lo, hi)


def _combined_indoor_temperature_y_range(
    obs_data: pd.DataFrame,
    zones: Sequence[Tuple[str, str, str]],
    threshold: float,
    *,
    pad_c: float = 0.5,
) -> tuple[float, float]:
    """Unión del rango Y interior de todas las zonas (misma escala en subplots apilados)."""
    los: list[float] = []
    his: list[float] = []
    for temp_col, sp_col, _ in zones:
        lo, hi = _indoor_temperature_y_range(
            obs_data, temp_col, sp_col, threshold, pad_c=pad_c
        )
        los.append(lo)
        his.append(hi)
    return min(los), max(his)


def _nice_temperature_dtick(y_lo: float, y_hi: float) -> float:
    """Espaciado uniforme de ticks (°C) según el span del eje."""
    span = y_hi - y_lo
    if span <= 10:
        return 1.0
    if span <= 22:
        return 2.0
    return 5.0


def _outdoor_temperature_y_range(series: pd.Series, *, pad_c: float = 2.0) -> tuple[float, float]:
    """Rango Y para temperatura exterior (panel inferior)."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return (-5.0, 35.0)
    arr = s.to_numpy(dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return (-5.0, 35.0)
    lo -= pad_c
    hi += pad_c
    if hi <= lo:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def _subplot_domain_refs(row_1based: int) -> tuple[str, str]:
    """``xref`` / ``yref`` para anotaciones ancladas al dominio (una columna)."""
    if row_1based <= 1:
        return 'x domain', 'y domain'
    return f'x{row_1based} domain', f'y{row_1based} domain'


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
            line=dict(color=_COMFORT_BAND_LINE_COLOR, width=1),
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
            line=dict(color=_COMFORT_BAND_LINE_COLOR, width=1),
            fillcolor=_COMFORT_BAND_FILLCOLOR,
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
            showlegend=show_legend,
            legendgroup="outdoor",
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


def _xaxis_layout_for_datetime_span(dt: pd.Series) -> dict:
    """Plotly ``xaxis`` / ``update_xaxes`` kwargs from the span of a datetime column.

    Chooses ``tickformat`` (and optional ``dtick``) so short windows
    do not repeat coarse labels (e.g. month abbreviation only on a single-day plot).
    """
    out: dict = {"type": "date"}
    if dt is None or getattr(dt, "empty", True):
        out["tickformat"] = "%Y-%m-%d"
        return out
    s = pd.to_datetime(dt, errors="coerce").dropna()
    if s.empty:
        out["tickformat"] = "%Y-%m-%d"
        return out

    t0 = pd.Timestamp(s.min())
    t1 = pd.Timestamp(s.max())
    span = t1 - t0
    if span.value <= 0:
        span = pd.Timedelta(hours=1)

    days = span.total_seconds() / 86400.0
    cross_year = t0.year != t1.year

    if days <= 1.5:
        out["tickformat"] = "%H:%M"
        out["dtick"] = 2 * 3600 * 1000
    elif days <= 14:
        # Corto: día + mes (+ año 2 dígitos si cruza año); sin inclinación.
        out["tickformat"] = "%a %d %b %y" if cross_year else "%a %d %b"
    elif days <= 120:
        out["tickformat"] = "%d %b %y" if cross_year else "%d %b"
    else:
        # Span largo: día + mes + año corto (único por tick sin repetir solo mes/año).
        out["tickformat"] = "%d %b %y"

    return out


def _export_plotly_figure(
    fig: go.Figure,
    path_stem: Path,
    export_format: Literal["html", "png"],
    *,
    png_width: int,
    png_height: int,
    png_scale: int = 2,
    paper_style: bool = False,
) -> None:
    """Write ``path_stem`` + ``.html`` or ``.png`` (requires kaleido for PNG)."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=png_width, height=png_height, **PLOTLY_WHITE_LAYOUT_KWARGS)
    if paper_style:
        apply_plotly_paper_style(fig)
    if export_format == "html":
        fig.write_html(str(path_stem.with_suffix(".html")))
        return
    out_png = path_stem.with_suffix(".png")
    try:
        fig.write_image(
            str(out_png),
            width=png_width,
            height=png_height,
            scale=png_scale,
        )
        _autocrop_png_whitespace(out_png)
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
    paper_style: bool = False,
    export_zone_subfolders: bool = True,
) -> None:
    """Grid + per-zone time views: export as PNG (default) or interactive HTML.

    La rejilla resumen es **una columna** (subplots apilados): mismos límites y
    espaciado de ticks en el eje Y para todas las zonas; etiquetas de tiempo solo
    en el panel inferior; nombre de cada zona alineado a la **izquierda** en su
    panel. Si hay datos de exterior (columna ``outdoor_temperature`` o la indicada
    en ``outdoor_temp_var``), se añade un último panel solo con la temperatura
    exterior (sin eje Y secundario en las zonas). El eje X del panel inferior
    usa los mismos ticks que los paneles interiores (``matches='x'`` + mismo
    ``xa_period``).

    Same data/comfort conventions as :func:`plot_temperature_one_zone`.
    ``zones`` is a sequence of ``(temp_var, setpoint_var, zone_name)`` per room.
    ``df`` must include ``datetime`` and all columns referenced by ``zones``.
    ``paper_style``: al exportar, aplica :func:`apply_plotly_paper_style`.
    ``export_zone_subfolders``: si es True (por defecto), exporta también las
    figuras por zona bajo ``case{N}/{slug_zona}/`` (periodo completo, día, semana, mes).
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

    xa_period = _xaxis_layout_for_datetime_span(obs["datetime"])
    xa_daily = (
        _xaxis_layout_for_datetime_span(obs_daily["datetime"])
        if not obs_daily.empty
        else xa_period
    )
    xa_week = (
        _xaxis_layout_for_datetime_span(obs_week["datetime"])
        if not obs_week.empty
        else xa_period
    )
    xa_month = (
        _xaxis_layout_for_datetime_span(obs_month["datetime"])
        if not obs_month.empty
        else xa_period
    )

    n_zones = len(zones)
    if n_zones == 0:
        return

    outdoor_col: Optional[str] = outdoor_temp_var
    if outdoor_col is None and 'outdoor_temperature' in obs.columns:
        outdoor_col = 'outdoor_temperature'
    include_outdoor_panel = (
        outdoor_col is not None
        and outdoor_col in obs.columns
        and obs[outdoor_col].notna().any()
    )

    n_rows = n_zones + (1 if include_outdoor_panel else 0)
    fig = sp.make_subplots(
        rows=n_rows,
        cols=1,
        vertical_spacing=PLOTLY_ZONE_TEMP_GRID_VERTICAL_SPACING,
        horizontal_spacing=PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
    )

    if temp_colors is None:
        colors_seq: List[Optional[str]] = [None] * n_zones
    else:
        colors_seq = list(temp_colors)
        if len(colors_seq) < n_zones:
            colors_seq.extend([None] * (n_zones - len(colors_seq)))

    y_lo, y_hi = _combined_indoor_temperature_y_range(obs, zones, threshold)
    y_dtick = _nice_temperature_dtick(y_lo, y_hi)

    for i, (temp_col, sp_col, _zone_title) in enumerate(zones):
        row = i + 1
        add_temperature_traces(
            fig,
            obs,
            temp_col,
            sp_col,
            show_legend=(i == 0),
            row=row,
            col=1,
            threshold=threshold,
            temp_color=colors_seq[i],
            outdoor_temp_var='',
        )
        fig.update_yaxes(
            title_text='',
            range=[y_lo, y_hi],
            dtick=y_dtick,
            row=row,
            col=1,
        )

    if include_outdoor_panel:
        x_series = _obs_x_values(obs)
        x_vals = x_series.to_numpy()
        outdoor_arr = pd.to_numeric(obs[outdoor_col], errors='coerce').to_numpy()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=outdoor_arr,
                mode='lines',
                name='Outdoor temperature',
                line=dict(color=_PLOTLY_TEXT, width=1.5),
                showlegend=True,
                hovertemplate='Outdoor: %{y:.2f}°C<extra></extra>',
            ),
            row=n_rows,
            col=1,
        )
        o_lo, o_hi = _outdoor_temperature_y_range(obs[outdoor_col])
        fig.update_yaxes(
            range=[o_lo, o_hi],
            row=n_rows,
            col=1,
        )

    ann_font = dict(
        family=_PLOTLY_SERIF,
        size=PLOTLY_PAPER_AXIS_TITLE_SIZE,
        color=_PLOTLY_TEXT,
    )
    # Títulos alineados a la izquierda, fuera del área de datos: ancla inferior en el borde
    # superior del dominio Y y ligero desplazamiento en px hacia arriba (evita solapar la serie).
    for i, (_, _, room_title) in enumerate(zones):
        row = i + 1
        xref, yref = _subplot_domain_refs(row)
        fig.add_annotation(
            xref=xref,
            yref=yref,
            x=0,
            y=1,
            xanchor='left',
            yanchor='bottom',
            yshift=6,
            text=room_title,
            showarrow=False,
            font=ann_font,
        )
    if include_outdoor_panel:
        xref, yref = _subplot_domain_refs(n_rows)
        fig.add_annotation(
            xref=xref,
            yref=yref,
            x=0,
            y=1,
            xanchor='left',
            yanchor='bottom',
            yshift=6,
            text='Outdoor temperature',
            showarrow=False,
            font=ann_font,
        )

    fig.update_xaxes(**xa_period)
    for r in range(2, n_rows + 1):
        fig.update_xaxes(matches='x', row=r, col=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=n_rows, col=1)

    grid_title = summary_title.strip() or f"case{case_id}"
    grid_height = max(220 * n_rows, 480)
    # width/height en layout: Kaleido usa las mismas dimensiones que write_image;
    # si solo se pasan a write_image, la anotación del eje Y puede solaparse con los ticks.
    fig.update_layout(
        title=None,
        width=png_width,
        height=grid_height,
        template="plotly_white",
        hovermode=False,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_FONT_SIZE,
            color=_PLOTLY_TEXT,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_LEGEND_SIZE,
                color=_PLOTLY_TEXT,
            ),
        ),
        margin=dict(l=120, r=20, b=30, t=52),
    )
    # Una sola etiqueta de eje Y para la rejilla (temperatura interior); evita repetir en cada zona.
    fig.add_annotation(
        text="Temperature (°C)",
        xref="paper",
        yref="paper",
        x=-0.072,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        showarrow=False,
        font=dict(
            family=_PLOTLY_SERIF,
            size=PLOTLY_PAPER_AXIS_TITLE_SIZE + 2,
            color=_PLOTLY_TEXT,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _export_plotly_figure(
        fig,
        output_dir / f"case{case_id}_temperatures",
        export_format,
        png_width=png_width,
        png_height=grid_height,
        png_scale=png_scale,
        paper_style=paper_style,
    )

    if not export_zone_subfolders:
        return

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
            xaxis=xa_period,
            font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
            template="plotly_white",
            height=500,
            hovermode=False,
            #xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )
        if added_out:
            layout_updates["yaxis2"] = _outdoor_yaxis2_layout()
        fig_r.update_layout(**layout_updates)
        _yr0, _yr1 = _indoor_temperature_y_range(obs, temp_col, sp_col, threshold)
        # go.Figure() sin make_subplots: no usar update_yaxes(..., secondary_y=...).
        fig_r.update_layout(yaxis=dict(range=[_yr0, _yr1]))

        _export_plotly_figure(
            fig_r,
            room_dir / "temperature",
            export_format,
            png_width=png_width,
            png_height=png_height_single,
            png_scale=png_scale,
            paper_style=paper_style,
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
                xaxis=xa_daily,
                template="plotly_white",
                font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
                height=500,
                hovermode=False,
            )
            if added_od:
                ld["yaxis2"] = _outdoor_yaxis2_layout()
            fig_d.update_layout(**ld)
            _yd0, _yd1 = _indoor_temperature_y_range(
                obs_daily, temp_col, sp_col, threshold
            )
            fig_d.update_layout(yaxis=dict(range=[_yd0, _yd1]))
            _export_plotly_figure(
                fig_d,
                room_dir / "daily_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
                paper_style=paper_style,
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
                xaxis=xa_week,
                template="plotly_white",
                font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
                height=500,
                hovermode=False,
            )
            if added_ow:
                lw["yaxis2"] = _outdoor_yaxis2_layout()
            fig_w.update_layout(**lw)
            _yw0, _yw1 = _indoor_temperature_y_range(
                obs_week, temp_col, sp_col, threshold
            )
            fig_w.update_layout(yaxis=dict(range=[_yw0, _yw1]))
            _export_plotly_figure(
                fig_w,
                room_dir / "weekly_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
                paper_style=paper_style,
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
                xaxis=xa_month,
                template="plotly_white",
                height=500,
                hovermode=False,
                font=dict(
                family=_PLOTLY_SERIF,
                size=PLOTLY_PAPER_FONT_SIZE,
                color=_PLOTLY_TEXT,
            ),
                #xaxis=dict(rangeslider=dict(visible=True), type="date"),
            )
            if added_om:
                lm["yaxis2"] = _outdoor_yaxis2_layout()
            fig_m.update_layout(**lm)
            _ym0, _ym1 = _indoor_temperature_y_range(
                obs_month, temp_col, sp_col, threshold
            )
            fig_m.update_layout(yaxis=dict(range=[_ym0, _ym1]))
            _export_plotly_figure(
                fig_m,
                room_dir / "monthly_temperature",
                export_format,
                png_width=png_width,
                png_height=png_height_single,
                png_scale=png_scale,
                paper_style=paper_style,
            )