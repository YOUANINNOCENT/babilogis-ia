"""Service IA BabiLogis (FastAPI) — Blocs 4 et 5.

Lancer :  uvicorn app.main:app --port 8001 --reload
Docs   :  http://127.0.0.1:8001/docs
"""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import Assistant
from .modele import COMMUNES, ModeleLoyer
from .outils import Outils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
etat: dict = {}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    modele = ModeleLoyer.charger()
    outils = Outils(modele)
    etat.update(modele=modele, outils=outils, assistant=Assistant(outils))
    yield
    outils.http.close()


# Sécurité (mise en ligne) :
#  - IA_CLE_INTERNE : secret partagé avec le serveur Dart. S'il est défini, toute requête (sauf /sante)
#    doit porter l'en-tête X-Cle-Interne : personne d'autre ne peut consommer l'agent (LLM payant).
#  - BABILOGIS_PRODUCTION=1 : documentation /docs désactivée.
#  - CORS_ORIGINES : origines autorisées depuis un navigateur (vide = aucune ; le site passe par le serveur Dart).
CLE_INTERNE = os.environ.get("IA_CLE_INTERNE", "")
PRODUCTION = os.environ.get("BABILOGIS_PRODUCTION") == "1"
ORIGINES = [o.strip() for o in os.environ.get("CORS_ORIGINES", "").split(",") if o.strip()]

app = FastAPI(title="BabiLogis — Service IA", version="1.0.0", lifespan=cycle_de_vie,
              description="Estimation des loyers (modèle du Bloc 2) et agent « Assistant logement ».",
              docs_url=None if PRODUCTION else "/docs", redoc_url=None,
              openapi_url=None if PRODUCTION else "/openapi.json")
if ORIGINES:
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINES, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Cle-Interne"])


@app.middleware("http")
async def verifier_cle_interne(requete: Request, suivant):
    if CLE_INTERNE and requete.url.path != "/sante":
        recue = requete.headers.get("x-cle-interne", "")
        if not hmac.compare_digest(recue.encode(), CLE_INTERNE.encode()):
            return JSONResponse({"detail": "Accès réservé au serveur BabiLogis."}, status_code=401)
    return await suivant(requete)


class DemandeEstimation(BaseModel):
    commune: str = Field(examples=["Cocody"])
    type: str = Field("Appartement", examples=["Appartement"], description="Studio, Appartement, Villa ou Chambre")
    pieces: int = Field(ge=1, le=20, examples=[3])
    meuble: bool = False
    equipements: list[str] = Field(default_factory=list, examples=[["Parking", "Gardien"]])
    quartier: str | None = Field(None, examples=["Riviera Palmeraie"])
    loyer: int | None = Field(None, ge=1, description="Loyer demandé : ajoute le verdict")


class Message(BaseModel):
    role: Literal["utilisateur", "assistant"]
    texte: str = Field(max_length=2000)


class DemandeAgent(BaseModel):
    message: str = Field(min_length=1, max_length=1000, examples=["Je cherche un 3 pièces à Cocody sous 300 000"])
    historique: list[Message] = Field(default_factory=list, max_length=20)


@app.get("/sante")
def sante():
    m: ModeleLoyer = etat["modele"]
    return {"statut": "ok", "service": "BabiLogis IA (FastAPI)", "modele": m.meta.get("modele"),
            "nb_annonces_entrainement": int(len(m.annonces)), "agent": etat["assistant"].moteur,
            "validation_croisee": m.meta.get("validation_croisee")}


def _estimer(d: DemandeEstimation) -> dict:
    try:
        return etat["modele"].estimer(d.commune, d.type, d.pieces, d.meuble, d.equipements, d.quartier, d.loyer)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/estimer", summary="Estimation du loyer (appelée par le serveur Dart via IA_SERVICE_URL)")
def estimer(d: DemandeEstimation):
    return _estimer(d)


@app.post("/verdict", summary="Verdict sur un loyer demandé")
def verdict(d: DemandeEstimation):
    if not d.loyer:
        raise HTTPException(422, "Le champ « loyer » est obligatoire pour un verdict.")
    return _estimer(d)


@app.get("/marche/{commune}", summary="Statistiques de marché d'une commune")
def marche(commune: str, type: str | None = None):
    try:
        return etat["modele"].statistiques(commune, type)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/communes")
def communes():
    m: ModeleLoyer = etat["modele"]
    return [{"nom": c, "annonces": int((m.annonces.commune == c).sum()),
             "donnees_suffisantes": bool((m.annonces.commune == c).sum() >= 30)} for c in COMMUNES]


@app.post("/agent/discuter", summary="Agent IA « Assistant logement »")
def discuter(d: DemandeAgent):
    return etat["assistant"].discuter(d.message.strip(), [h.model_dump() for h in d.historique])
