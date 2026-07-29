# Image de production pour src/dashboard/ -- STRICTEMENT lecture seule sur
# des artefacts DEJA generes (data/processed/, models/, reports/, img/) :
# aucun entrainement ni scraping ne tourne dans ce conteneur, cf.
# src/dashboard/data_loader.py. Utilisee par docker-compose.yml (local/
# self-hosting) et render.yaml (Render.com).
#
# PENSER A REGENERER reports/notebooks_pdf/ (page Telechargements) avant un
# build si un notebook a change : python scripts/generate_notebook_pdfs.py
# -- hors-ligne uniquement, jupyter/nbconvert/playwright ne sont PAS dans
# requirements-deploy.txt (image de production volontairement allegee).

FROM python:3.12-slim

WORKDIR /app

# Dependances d'abord, dans leur propre couche -- un rebuild qui ne change
# que le code (src/) ne reinstalle jamais les packages Python.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Code + artefacts necessaires au dashboard en lecture seule. data/raw/ et
# notebooks/ ne sont jamais lus par le dashboard -- exclus par
# .dockerignore, jamais copies dans l'image.
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY img/ img/

ENV PYTHONUNBUFFERED=1
# Port par defaut pour un lancement local (docker-compose) -- Render
# fournit sa PROPRE valeur de PORT au conteneur au demarrage, qui prend le
# pas sur cette valeur par defaut (cf. app.py, jamais code en dur).
ENV PORT=8050

EXPOSE 8050

# /healthz (pas "/") : instantane, sans lecture de donnees -- cf. app.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8050') + '/healthz', timeout=3)" || exit 1

# Forme shell (pas exec) -- necessaire pour que $PORT soit developpe par
# le shell avant d'etre passe a gunicorn. 1 worker/4 threads : profil
# memoire modeste, adapte a un niveau gratuit (Render free -- 512 Mo) ;
# a augmenter (--workers) si deploye avec plus de ressources/trafic.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 src.dashboard.app:server
