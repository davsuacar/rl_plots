import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

# Read the CSV files
case1_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso/Eval-DRL-Baseline-2026-cases/caso1/Eval-DRL-Baseline-2026-case-1_2025-12-17_10:25-res1/progress.csv')
case2_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso/Eval-DRL-Baseline-2026-cases/caso2/Eval-DRL-Baseline-2026-case-2_2025-12-17_10:31-res1/progress.csv')
case3_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso/Eval-DRL-Baseline-2026-cases/caso3/Eval-DRL-Baseline-2026-case-3_2025-12-17_10:35-res1/progress.csv')

# Create the plotly figure
fig = go.Figure()

# Add traces for each case
fig.add_trace(go.Scatter(
    x=case1_df['episode_num'],
    y=case1_df['mean_reward'],
    mode='lines',
    name='Case 1 (PPO)',
    line=dict(color='#4c72b0', width=3),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=case2_df['episode_num'],
    y=case2_df['mean_reward'],
    mode='lines',
    name='Case 2 (TQC)',
    line=dict(color='#55a868', width=3),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=case3_df['episode_num'],
    y=case3_df['mean_reward'],
    mode='lines',
    name='Case 3 (TQC)',
    line=dict(color='#c44e52', width=3),
    showlegend=True
))

# Update layout
fig.update_layout(
    title='Mean Reward Comparison',
    xaxis_title='Episode Number',
    yaxis_title='Mean Reward',
    hovermode='x unified',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ),
    width=1000,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        gridcolor='lightgray',
        showgrid=True
    ),
    yaxis=dict(
        gridcolor='lightgray',
        showgrid=True
    )
)

# Save as HTML
fig.write_html('./data/paper/plots/pilot_study/training_and_evaluation/rewards_plot.html')
print("Gráfico guardado en rewards_plot.html")
