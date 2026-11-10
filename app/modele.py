"""Modèle d'estimation du loyer (Bloc 2) : chargement, réentraînement si besoin, prédiction et verdict."""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from . import config

log = logging.getLogger("babilogis.modele")

COMMUNES = ["Abobo", "Adjamé", "Anyama", "Attécoubé", "Bingerville", "Cocody", "Koumassi",
            "Marcory", "Plateau", "Port-Bouët", "Songon", "Treichville", "Yopougon"]
# Communes absentes des données collectées -> commune voisine au marché comparable (fiabilité faible)
COMMUNE_PROCHE = {"Adjamé": "Abobo", "Anyama": "Abobo", "Attécoubé": "Yopougon", "Treichville": "Marcory"}
EQUIPEMENTS = ["piscine", "jardin", "parking", "gardien", "climatisation", "groupe_electrogene",
               "forage", "ascenseur", "terrasse", "haut_standing"]
CAT = ["commune", "type_logement"]
NUM = ["pieces", "meuble", "nb_equipements"] + [f"eq_{e}" for e in EQUIPEMENTS]
COLONNES = CAT + NUM + ["quartier_cle"]
SEUIL_BAS, SEUIL_HAUT = 0.85, 1.15
LIBELLES = {"bonne_affaire": "Bonne affaire", "juste": "Prix juste", "au_dessus": "Au-dessus du marché"}


def sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower().strip()


_COMMUNES_NORM = {sans_accents(c).replace("-", " "): c for c in COMMUNES}


def normaliser_commune(nom: str | None) -> str | None:
    if not nom:
        return None
    return _COMMUNES_NORM.get(sans_accents(nom).replace("-", " "))


def normaliser_type(t: str | None, pieces: int | None = None) -> str:
    t = sans_accents(t or "")
    if "studio" in t:
        return "studio"
    if "chambre" in t:
        return "chambre"
    if any(m in t for m in ("villa", "maison", "duplex", "triplex")):
        return "villa"
    if t.startswith("appart") or t in ("f", "apt"):
        return "appartement"
    return "studio" if pieces == 1 else "appartement"


def normaliser_equipement(e: str) -> str | None:
    e = sans_accents(e).replace("é", "e")
    table = {"clim": "climatisation", "groupe": "groupe_electrogene", "garage": "parking", "parking": "parking",
             "gardien": "gardien", "securite": "gardien", "forage": "forage", "piscine": "piscine",
             "jardin": "jardin", "ascenseur": "ascenseur", "terrasse": "terrasse", "balcon": "terrasse",
             "standing": "haut_standing"}
    for cle, val in table.items():
        if cle in e:
            return val
    return None   # ex. « Cuisine équipée » : non utilisé par le modèle


def verdict(loyer: float, estimation: float) -> str:
    r = loyer / estimation
    return "bonne_affaire" if r < SEUIL_BAS else ("au_dessus" if r > SEUIL_HAUT else "juste")


def arrondir(v: float, pas: int = 5000) -> int:
    return int(round(v / pas) * pas)


def construire_pipeline() -> TransformedTargetRegressor:
    """Même modèle que le notebook du Bloc 2 (Random Forest sur log(loyer))."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
        ("quartier", TargetEncoder(target_type="continuous", random_state=2026), ["quartier_cle"]),
    ])
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5, random_state=2026, n_jobs=-1)
    return TransformedTargetRegressor(regressor=Pipeline([("prep", prep), ("m", rf)]), func=np.log, inverse_func=np.exp)


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eqs = df["equipements"].fillna("").astype(str).str.split(";")
    for e in EQUIPEMENTS:
        df[f"eq_{e}"] = eqs.apply(lambda l, e=e: int(e in l))
    df["meuble"] = df["meuble"].astype(int)
    df["quartier_cle"] = df["quartier"].fillna("inconnu").astype(str).str.lower().str.split().str[0]
    return df


@dataclass
class ModeleLoyer:
    pipeline: TransformedTargetRegressor
    meta: dict
    annonces: pd.DataFrame
    reference: pd.DataFrame
    fourchette: tuple[float, float] = (0.75, 1.33)
    communes_connues: set = field(default_factory=set)

    # ── Chargement ────────────────────────────────────────────────────────
    @classmethod
    def charger(cls) -> "ModeleLoyer":
        annonces = preparer(pd.read_csv(config.FICHIER_ANNONCES))
        reference = pd.read_csv(config.FICHIER_REFERENCE)
        meta = json.loads(config.FICHIER_META.read_text(encoding="utf-8")) if config.FICHIER_META.exists() else {}
        pipeline = None
        if config.FICHIER_MODELE.exists() and meta.get("scikit_learn") == sklearn.__version__:
            try:
                pipeline = joblib.load(config.FICHIER_MODELE)
                log.info("Modèle chargé depuis %s", config.FICHIER_MODELE.name)
            except Exception as e:  # fichier abîmé ou incompatible
                log.warning("Chargement impossible (%s) : réentraînement", e)
        else:
            log.warning("scikit-learn %s ≠ %s (version du modèle) : réentraînement à partir de %s",
                        sklearn.__version__, meta.get("scikit_learn"), config.FICHIER_ANNONCES.name)
        if pipeline is None:
            pipeline = construire_pipeline().fit(annonces[COLONNES], annonces["prix_fcfa"].astype(float))
            try:
                joblib.dump(pipeline, config.FICHIER_MODELE)
                meta["scikit_learn"] = sklearn.__version__
                config.FICHIER_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
        annonces["estimation"] = pipeline.predict(annonces[COLONNES])
        f = meta.get("fourchette_multiplicateurs", [0.75, 1.33])
        return cls(pipeline, meta, annonces, reference, (float(f[0]), float(f[1])), set(annonces["commune"].unique()))

    # ── Estimation ────────────────────────────────────────────────────────
    def estimer(self, commune: str, type_logement: str, pieces: int, meuble: bool = False,
                equipements: list[str] | None = None, quartier: str | None = None,
                loyer: float | None = None) -> dict:
        commune_ok = normaliser_commune(commune)
        if commune_ok is None:
            raise ValueError(f"Commune inconnue : {commune}. Communes possibles : {', '.join(COMMUNES)}.")
        pieces = int(pieces)
        if not 1 <= pieces <= 20:
            raise ValueError("Nombre de pièces invalide (1 à 20).")
        type_m = normaliser_type(type_logement, pieces)
        eqs = sorted({x for x in (normaliser_equipement(e) for e in (equipements or [])) if x})
        utilisee = commune_ok if commune_ok in self.communes_connues else COMMUNE_PROCHE.get(commune_ok, "Yopougon")
        ligne = {"commune": utilisee, "type_logement": type_m, "pieces": pieces, "meuble": int(bool(meuble)),
                 "nb_equipements": len(eqs),
                 "quartier_cle": (str(quartier).lower().split() or ["inconnu"])[0] if quartier else "inconnu"}
        ligne.update({f"eq_{e}": int(e in eqs) for e in EQUIPEMENTS})
        centrale = float(self.pipeline.predict(pd.DataFrame([ligne])[COLONNES])[0])
        bas, haut = self.fourchette
        fiable = utilisee == commune_ok
        if not fiable:                      # commune sans données : fourchette élargie
            bas, haut = bas * 0.9, haut * 1.1
        n_comp = int(((self.annonces.commune == utilisee) & (self.annonces.pieces == pieces)).sum())
        res = {
            "commune": commune_ok, "type_logement": type_m, "pieces": pieces, "meuble": bool(meuble),
            "equipements_retenus": eqs,
            "estimation_centrale": arrondir(centrale),
            # fourchette du « prix juste » = mêmes seuils que le verdict (85 % – 115 %)
            "estimation_min": arrondir(centrale * SEUIL_BAS * (bas / self.fourchette[0])),
            "estimation_max": arrondir(centrale * SEUIL_HAUT * (haut / self.fourchette[1])),
            # intervalle où se trouvent 60 % des loyers observés (incertitude du modèle)
            "intervalle_observe": [arrondir(centrale * bas), arrondir(centrale * haut)],
            "fiabilite": "normale" if fiable and n_comp >= 10 else "faible",
            "nb_comparables": n_comp,
            "source": "modele",
            "modele": self.meta.get("modele", "Random Forest"),
        }
        if not fiable:
            res["avertissement"] = (f"Peu ou pas d'annonces collectées à {commune_ok} : estimation basée sur "
                                    f"{utilisee}, à prendre avec prudence.")
        if loyer:
            v = verdict(float(loyer), centrale)
            res.update({"loyer": int(loyer), "verdict": v, "verdict_libelle": LIBELLES[v],
                        "ecart_pct": round((float(loyer) / centrale - 1) * 100)})
        return res

    # ── Statistiques ──────────────────────────────────────────────────────
    def statistiques(self, commune: str, type_logement: str | None = None) -> dict:
        c = normaliser_commune(commune)
        if c is None:
            raise ValueError(f"Commune inconnue : {commune}.")
        d = self.annonces[self.annonces.commune == c]
        if type_logement:
            d = d[d.type_logement == normaliser_type(type_logement)]
        if d.empty:
            return {"commune": c, "nb_annonces": 0,
                    "message": "Aucune annonce collectée pour cette commune dans nos données."}
        par_pieces = (d.groupby("pieces")["prix_fcfa"].agg(["size", "median"]).reset_index()
                      .rename(columns={"size": "annonces", "median": "loyer_median"}))
        return {
            "commune": c, "type_logement": type_logement, "nb_annonces": int(len(d)),
            "loyer_median": arrondir(d.prix_fcfa.median()),
            "loyer_q1": arrondir(d.prix_fcfa.quantile(.25)), "loyer_q3": arrondir(d.prix_fcfa.quantile(.75)),
            "par_pieces": [{"pieces": int(r.pieces), "annonces": int(r.annonces), "loyer_median": arrondir(r.loyer_median)}
                           for r in par_pieces.itertuples() if r.pieces <= 8],
            "quartiers_frequents": d.quartier.dropna().str.title().value_counts().head(5).index.tolist(),
        }
