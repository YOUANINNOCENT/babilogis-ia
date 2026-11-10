# Service IA BabiLogis (FastAPI) : modèle Random Forest du Bloc 2 + agent « Babi » (DeepSeek).
# Déploiement Railway : service « ia », dossier racine = babilog/ia_service, réseau privé seulement.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY donnees ./donnees
ENV BABILOGIS_PRODUCTION=1
# « :: » : écoute en IPv4 et IPv6 (réseau privé Railway : ia.railway.internal)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0  --port ${PORT:-8001}"]
