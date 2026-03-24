# Plot Container

Entorno de análisis y visualización de datos para experimentos de aprendizaje por refuerzo (RL), integrado con [Weights & Biases](https://wandb.ai). Incluye Jupyter Lab en Docker, utilidades para descargar historiales de runs y notebooks para comparar simulaciones, consumo energético, caudal y episodios.

## Estructura del proyecto

```
plot_container/
├── src/
│   └── utils/
│       ├── download_data.py   # Descarga el history de un run de W&B a CSV
│       └── uponor_plots.py   # Utilidades de visualización Uponor
├── notebooks/
│   ├── sim_vs_reality/       # Comparación simulación vs realidad
│   ├── comparative_analysis/ # Análisis comparativo (Uponor, setpoints)
│   ├── energy_analysis/      # Consumo energético
│   ├── flow_rate/            # Análisis de caudal
│   ├── episode_analysis/     # Observaciones, recompensas, acciones, progreso
│   └── occupancy_analysis/   # Análisis de ocupación
├── data/                     # CSVs descargados (no versionado)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .devcontainer/            # Dev Container para VS Code / Cursor
```

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) y Docker Compose
- (Opcional) [Weights & Biases](https://wandb.ai) y API key para descargar runs

## Uso

### Descargar datos de un run (W&B)

1. Configura las variables en `src/utils/download_data.py`:

   ```python
   ENTITY = "tu-entity"
   PROJECT = "tu-project"
   RUN_ID = "id-del-run"
   ```

2. Ejecuta el script (con `wandb` configurado y login hecho):

   ```bash
   python src/utils/download_data.py
   ```

   El CSV se guarda en `data/{ENTITY}_{PROJECT}_{RUN_ID}_history.csv`.

### Levantar Jupyter Lab con Docker

```bash
docker compose up -d
```

Jupyter Lab queda disponible en **http://localhost:8888**. El directorio del proyecto (`.`) se monta en `/home/jovyan/work`, así que cualquier cambio en notebooks, `src/` u otros archivos queda guardado en tu máquina y persiste al cerrar el contenedor. El `PYTHONPATH` incluye `src` para imports desde los notebooks.

### Usar el Dev Container (VS Code / Cursor)

1. Abre la carpeta del proyecto en VS Code o Cursor.
2. Ejecuta **“Reopen in Container”** (o “Dev Containers: Reopen in Container”).
3. Se construye el contenedor y el workspace queda en `/home/jovyan/work` con Jupyter y extensiones ya configuradas.

## Dependencias principales

- **Python 3.12** (en el contenedor)
- `pandas`, `numpy`, `scikit-learn`, `scipy`
- `matplotlib`, `seaborn`, `plotly`, `bokeh`
- `wandb`

Ver `requirements.txt` para la lista completa.

## Notas

- La carpeta `data/` está en `.gitignore`; los CSVs generados no se versionan.
- En `docker-compose.yml` hay un volumen que monta una ruta local de datos (`LAB_EVALUATION`); puedes ajustarla o comentarla si no la usas.
- Para producción, configura un token o contraseña de Jupyter (variables en `docker-compose.yml` o en el `CMD` del Dockerfile).
