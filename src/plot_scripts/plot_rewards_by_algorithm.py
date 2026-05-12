import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

# Read the CSV files
PPO_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/PPO/training/progress_ppo_case3.csv')
TQC_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/TQC/training/progress_tqc_case3.csv')
SAC_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/SAC/training/progress_sac_case3.csv')
RPO_df = pd.read_csv(
    '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/RPPO/training/progress_rpo_case3.csv')


# Create the plotly figure
fig = go.Figure()

# Add traces for each case
fig.add_trace(go.Scatter(
    x=PPO_df['episode_num'],
    y=PPO_df['mean_reward'],
    mode='lines',
    name='PPO',
    line=dict(color='#1ABC9C', width=3),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=TQC_df['episode_num'],
    y=TQC_df['mean_reward'],
    mode='lines',
    name='TQC',
    line=dict(color='#3498DB', width=3),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=SAC_df['episode_num'],
    y=SAC_df['mean_reward'],
    mode='lines',
    name='SAC',
    line=dict(color='#9B59B6', width=3),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=RPO_df['episode_num'],
    y=RPO_df['mean_reward'],
    mode='lines',
    name='RecPPO',
    line=dict(color='#E74C3C', width=3),
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
fig.write_html('./data/paper/plots/pilot_study/training_and_evaluation/caso_1/rewards_plot.html')
print("Gráfico guardado en rewards_plot.html")
