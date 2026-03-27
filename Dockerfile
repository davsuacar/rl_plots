# ---------------------------------------------------------------------------- #
#  Jupyter Lab + Data Science stack (sin imagen oficial Jupyter desactualizada) #
# ---------------------------------------------------------------------------- #

ARG PYTHON_VERSION=3.12
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim-bookworm

# ---------------------------------------------------------------------------- #
#                      CONTAINER ARGUMENTS AND ENV CONFIG                      #
# ---------------------------------------------------------------------------- #

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Usuario no root (compatible con entornos Jupyter)
ARG NB_UID=1000
ARG NB_GID=100
ENV NB_UID=${NB_UID}
ENV NB_GID=${NB_GID}

WORKDIR /home/jovyan/work

# Incluir src en el path para imports en notebooks y scripts
ENV PYTHONPATH=/home/jovyan/work/src

# ---------------------------------------------------------------------------- #
#                        SYSTEM PACKAGES & USER                                #
# ---------------------------------------------------------------------------- #

# Dependencias gráficas (Plotly Kaleido / Chromium headless, Playwright, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libasound2 \
    libatk-bridge2.0-0 \
    libcairo2 \
    libcups2 \
    libexpat1 \
    libgbm1 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Usuario jovyan (estándar en imágenes Jupyter)
RUN groupadd -g ${NB_GID} -o jovyan \
    && useradd -m -u ${NB_UID} -g jovyan -o -s /bin/bash jovyan \
    && chown -R jovyan:jovyan /home/jovyan

# ---------------------------------------------------------------------------- #
#                        PYTHON & JUPYTER STACK                               #
# ---------------------------------------------------------------------------- #

# Jupyter Lab + kernel + dependencias base de datos/visualización
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
    jupyterlab \
    jupyter \
    ipykernel \
    ipywidgets \
    notebook

# Registrar el kernel para que Cursor/VS Code y Jupyter lo detecten
RUN python -m ipykernel install --name python3 --display-name "Python 3.12"

# Copiar e instalar dependencias del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto
COPY --chown=jovyan:jovyan . .

# ---------------------------------------------------------------------------- #
#                              RUNTIME                                        #
# ---------------------------------------------------------------------------- #

USER jovyan
EXPOSE 8888

# Token por defecto deshabilitado; en producción usar --identity-provider o token
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=''", "--ServerApp.password=''"]
