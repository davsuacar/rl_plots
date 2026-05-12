import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp

from utils.plot_functions.plot_functions import (
    PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
    PLOTLY_SUBPLOT_VERTICAL_SPACING,
)

pio.templates.default = "plotly_white"

# -------------------------- Datetime preprocessing -------------------------- #


def add_datetime_column(df):
    df.rename(columns={'day_of_month': 'day'}, inplace=True)
    df['year'] = 2022
    df.loc[df['month'] < 5, 'year'] = 2023
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour', 'minutes']])
    return df


def filer_interval(df, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df[(df['datetime'] >= start) & (df['datetime'] <= end)]


def resample(df):
    output = df.resample('1h', on='datetime').mean()
    output.reset_index(inplace=True)
    return output

# ------------------------------- Summary data ------------------------------- #


def mean_variable(df, variable):
    return df[variable].mean()

# ---------------------------- Plotting functions ---------------------------- #


def plot_dfs_line(df_dict, variable_name, colors=None):
    """
    Genera un gráfico de líneas de progreso.

    Parámetros:
    - dfs: lista de DataFrames que contienen los datos de progreso
    - names: lista de nombres para cada línea
    - variable_name: nombre de la columna en los DataFrames que se desea graficar en el eje y
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    """

    # Colores predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[:len(df_dict)]

    # Crear figura
    fig = go.Figure()

    # Añadir líneas para cada DataFrame
    for (name, df), color in zip(df_dict.items(), colors):
        fig.add_trace(go.Scatter(
            x=df['episode_num'],
            y=df[variable_name],
            mode='lines',
            line=dict(color=color),
            name=name
        ))

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title='Datetime',
        yaxis_title=None,
        font=dict(
            family="Arial, sans-serif",
            size=20,
            color="black"
        ),
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
    )

    # Devolver la figura
    return fig


def plot_control(
        df,
        temperature_variables,
        flow_variables,
        names,
        colors=None):
    """
    Genera un gráfico de líneas comparando temperaturas con caudales,
    usando ejes separados para cada tipo (temperatura a la izquierda, caudal a la derecha).

    Parámetros:
    - df: DataFrame con columna 'datetime' y los datos a graficar.
    - temperature_variables: lista de columnas con temperaturas.
    - flow_variables: lista de columnas con caudales.
    - names: lista de nombres a mostrar en la leyenda (orden: temps + flows).
    - colors: lista opcional de colores para cada línea.
    """

    # Validación de colores
    total_vars = len(temperature_variables) + len(flow_variables)
    if colors is None:
        colors = px.colors.qualitative.Plotly[:total_vars]
    elif len(colors) < total_vars:
        raise ValueError(
            "No hay suficientes colores para todas las variables.")

    # Asegurar que datetime esté parseado
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Crear figura vacía
    fig = go.Figure()

    # Añadir trazos de temperatura
    for i, var in enumerate(temperature_variables):
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df[var],
            mode='lines',
            name=names[i],
            line=dict(color=colors[i], width=2),
            opacity=0.5,
            yaxis='y'
        ))

    # Añadir trazos de caudal (todos en eje y2)
    for j, var in enumerate(flow_variables):
        idx = len(temperature_variables) + j
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df[var],
            mode='lines',
            name=names[idx],
            line=dict(color=colors[idx], dash='dot', width=2),
            opacity=0.7,
            yaxis='y2'
        ))

    # Configurar ejes
    fig.update_layout(
        xaxis=dict(
            title='',
            dtick="M1",
            tickformat="%b\n%Y",
            ticklabelmode="period"
        ),
        yaxis=dict(
            title='Temperature (°C)',
            side='left'
        ),
        yaxis2=dict(
            title='Flow rate',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        font=dict(
            family="Arial, sans-serif",
            size=16,
            color="black"
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=14)
        ),
        width=1000,
        height=600,
        title=None
    )

    return fig


def plot_temperatures(df, variables, names, colors=None):
    """
    Genera un gráfico de líneas para varias columnas de temperatura en un DataFrame.

    Parámetros:
    - df: DataFrame que contiene los datos de temperatura
    - variables: lista de nombres de las columnas en el DataFrame que se desean graficar en el eje y
    - names: lista de nombres que se utilizarán en la leyenda para cada línea
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    """

    # Colores predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[:len(variables)]

    # Crear figura con las columnas especificadas
    fig = px.line(df, x='datetime', y=variables)

    # Agregar líneas horizontales
    fig.add_hline(y=19.5, line_dash="dot", line_color="blue")
    fig.add_hline(y=22.0, line_dash="dot", line_color="red")

    # Agregar líneas horizontales
    # Convertir fechas clave
    dec_1 = pd.to_datetime("2022-12-01")
    feb_1 = pd.to_datetime("2023-02-01")

    # Crear lista de líneas horizontales por tramo
    tramos = [{'start': df['datetime'].min(),
               'end': dec_1,
               'y0': 19.5,
               'y1': 21.5,
               'color0': 'blue',
               'color1': 'red'},
              {'start': dec_1,
               'end': feb_1,
               'y0': 19.0,
               'y1': 21.0,
               'color0': 'blue',
               'color1': 'red'},
              {'start': feb_1,
               'end': df['datetime'].max(),
               'y0': 20.0,
               'y1': 22.0,
               'color0': 'blue',
               'color1': 'red'},
              ]

    # Configurar nombres, colores y opacidad para cada línea
    for i, trace in enumerate(fig.data[:len(variables)]):
        trace.name = names[i]
        trace.line.color = colors[i]
        trace.opacity = 0.5

    # Configurar formato del eje X para mostrar el mes abreviado
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        ticklabelmode="period"
    )

    # Configurar el eje Y con valores específicos y rango
    # fig.update_yaxes(dtick=5, tickvals=[val for val in range(
    #     19, 23, 1) if val < 19.5 or val > 22] + [19.5, 22], range=[19, 24])

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
            font=dict(size=20)
        ),
        font=dict(
            family="Arial, sans-serif",
            size=16,
            color="black"
        ),
        width=1000,
        height=600
    )

    # Devolver la figura
    return fig


def plot_dfs_line_grouped_by_month(
        dfs,
        names,
        variable,
        colors=None,
        line_styles=None):
    """
    Genera un gráfico de líneas agrupado por meses de la variable deseada.

    Parámetros:
    - dfs: lista de DataFrames que contienen los datos a comparar
    - names: lista de nombres para cada línea (aparecerán en la leyenda)
    - variable: nombre de la columna en los DataFrames que se desea graficar.
    - colors: lista opcional de colores para cada línea; si no se proporciona, se asignan colores predeterminados
    - line_styles: lista opcional de estilos de línea para cada gráfico; si no se proporciona, se asignan estilos predeterminados
    """

    # Colores y estilos de línea predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[:len(dfs)]
    if line_styles is None:
        line_styles = [None for _ in range(len(dfs))]

    # Crear figura
    fig = go.Figure()

    # Para cada DataFrame, calcular la media mensual de variable
    all_dates = []  # Lista para almacenar todas las fechas (meses) únicas
    for df, name, color, line_style in zip(dfs, names, colors, line_styles):

        # Calcular la media mensual
        df['year_month'] = df['datetime'].dt.to_period(
            'M')  # Extraer año y mes como periodo
        monthly_avg = df.groupby('year_month')[
            variable].mean().reset_index()  # Calcular la media
        # Convertir de nuevo a datetime
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()

        # Añadir las fechas únicas al conjunto de fechas para el tick en el eje
        # X
        all_dates.extend(monthly_avg['year_month'].tolist())

        # Añadir la línea de la media mensual al gráfico
        fig.add_trace(go.Scatter(
            x=monthly_avg['year_month'],
            y=monthly_avg[variable],
            mode='lines',
            name=name,
            line=dict(color=color, dash=line_style)
        ))

    # Filtrar las fechas únicas y seleccionar 5 fechas espaciadas uniformemente
    unique_dates = sorted(list(set(all_dates)))
    tickvals = [
        unique_dates[i] for i in range(
            0, len(unique_dates), max(
                1, len(unique_dates) // 5))]

    # Configurar el layout
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title='Power demand (W)',
        xaxis=dict(
            tickformat='%b %Y',  # Formato de mes y año
            tickvals=tickvals,   # Fechas seleccionadas como ticks
            tickangle=45         # Ángulo para que las etiquetas no se superpongan
        ),
        showlegend=True,
        font=dict(
            family="Arial, sans-serif",
            size=20,
            color="black"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        height=600,
        width=1000
    )

    # Devolver la figura
    return fig


def plot_bar(dict_data, bar_colors=None):

    keys = list(dict_data.keys())
    values = list(dict_data.values())

    if bar_colors is None:
        bar_colors = ["royalblue"] * len(keys)

    if len(bar_colors) != len(keys):
        raise ValueError(
            "La cantidad de colores debe ser igual a la cantidad de elementos en el diccionario.")

    fig = go.Figure(go.Bar(
        x=keys,
        y=values,
        marker=dict(color=bar_colors),
        text=values,
        textposition='inside',
        texttemplate='%{text:.4f}'  # Lista de colores
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Gráfico de Barras con Colores Personalizados",
        xaxis_title="Categorías",
        yaxis_title="Valores",
        template="plotly_white"
    )

    return fig


def plot_bar_groups(dict_data):

    niveles = ['best',
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
                offsetgroup=2),
            go.Bar(
                name='Case2-PPO-BalancedRewardV2',
                x=['Case2-PPO-BalancedRewardV2'],
                y=balancedv2_values,
                marker_color='slateblue',
                text=balancedv2_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=1),
            go.Bar(
                name='Case2-PPO-standard',
                x=['Case2-PPO-standard'],
                y=case2_values,
                marker_color='cyan',
                text=case2_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=2),
            go.Bar(
                name='Case1-PPO',
                x=['Case1-PPO'],
                y=case1_values,
                marker_color='darkseagreen',
                text=case1_values,
                textposition='inside',
                texttemplate='%{text:.4f}',
                offsetgroup=1),
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
        ])

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
                offsetgroup=1),
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
        ])

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


def plot_action_distribution(dfs, names, variable, colors=None):

    if colors is None:
        colors = px.colors.qualitative.Plotly[:len(names)]

    # Create figure
    fig = go.Figure()

    # Add violin distribution for each flow rate
    for i, df in enumerate(dfs):
        fig.add_trace(go.Violin(
            y=df[variable],
            x=[names[i]] * len(df[variable]),  # Asigna el nombre en el eje X
            name=names[i],  # Nombre para la leyenda
            box_visible=False,
            line_color=colors[i]
        ))

    # Layout update
    fig.update_layout(
        title='Water supply temperature distribution',
        xaxis_title='Model',
        yaxis_title='Temperature ºC',
        xaxis=dict(
            tickmode='array',
            tickvals=names,
            ticktext=names
        )
    )

    # Show figure
    return fig


def plot_dfs_boxplot(dfs, variable, names, colors=None):
    """
    Genera un gráfico de cajas para la comparación de una variable en múltiples DataFrames.

    Parámetros:
    - dfs: lista de DataFrames que contienen los datos a comparar
    - variable: nombre de la columna en los DataFrames que se desea graficar
    - names: lista de nombres para cada grupo (aparecerán en el eje X)
    - colors: lista opcional de colores para cada caja; si no se proporciona, se asignan colores predeterminados
    """

    # Colores predeterminados si no se especifican
    if colors is None:
        colors = px.colors.qualitative.Plotly[:len(dfs)]

    # Crear figura
    fig = go.Figure()

    # Añadir cajas para cada DataFrame
    for df, name, color in zip(dfs, names, colors):
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
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,  # No mostrar la leyenda, los nombres están en el eje X
        font=dict(
            family="Arial, sans-serif",
            size=20,
            color="black"
        ),
        width=1000,  # Ancho de la figura
        height=600   # Alto de la figura
    )

    # Devolver la figura
    return fig


def plot_energy_savings(
        data,
        names_reference,
        names_comparison,
        variable,
        colors=None):
    """
    Calcula y grafica el porcentaje de ahorro energético al comparar cada DataFrame en dfs_comparison
    contra cada DataFrame en dfs_reference.

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
        colors = px.colors.qualitative.Plotly[:len(
            names_reference) * len(names_comparison)]

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
            fig.add_trace(go.Bar(
                x=aligned_ref.index.to_timestamp(),
                y=energy_savings,
                name=f'{name_comp} vs {name_ref}',
                marker_color=colors[count],
                text=energy_savings,
                textposition='inside',
                texttemplate='%{text:.2f}'
            ))
            count += 1

    # Configurar el eje X y diseño del gráfico
    x_values = ref_means[0].index.to_timestamp()
    fig.update_xaxes(
        tickvals=x_values,
        ticktext=x_values.strftime('%b'),
        ticklabelposition='outside'
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
            itemwidth=60
        ),
        height=600,
        width=1000
    )

    # Devolver la figura
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
                texttemplate='%{text:.4f}'
            ),
            row=row, col=col
        )

    # Personalización del diseño
    fig.update_layout(
        title='Comparison of Metrics: With vs Without Weather',
        height=max(320 * num_rows, 360),
        showlegend=False,
        template='plotly_white'
    )

    return fig
