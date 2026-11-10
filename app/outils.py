"""Outils de l'agent : chaque fonction renvoie un dict JSON simple (lisible par Gemini et par l'interface)."""
from __future__ import annotations

import logging

import httpx

from . import config
from .modele import COMMUNES, LIBELLES, ModeleLoyer, normaliser_commune, normaliser_type, verdict

log = logging.getLogger("babilogis.outils")
TYPES_DART = {"studio": "Studio", "appartement": "Appartement", "villa": "Villa", "chambre": "Studio"}


class Outils:
    def __init__(self, modele: ModeleLoyer, client_http: httpx.Client | None = None):
        self.modele = modele
        self.http = client_http or httpx.Client(timeout=4.0)

    # ── 1. Recherche d'annonces ───────────────────────────────────────────
    def rechercher_annonces(self, commune: str | None = None, type_logement: str | None = None,
                            pieces: int | None = None, budget_max: int | None = None,
                            meuble: bool | None = None, limite: int = 5) -> dict:
        commune_ok = normaliser_commune(commune) if commune else None
        if commune and commune_ok is None:
            return {"erreur": f"Commune inconnue : {commune}. Communes : {', '.join(COMMUNES)}."}
        limite = max(1, min(int(limite or 5), 10))
        params = {"per_page": 100}
        if commune_ok:
            params["commune"] = commune_ok
        if type_logement:
            params["type"] = TYPES_DART[normaliser_type(type_logement, pieces)]
        if pieces:
            params["pieces"] = int(pieces)
        if budget_max:
            params["budget_max"] = int(budget_max)
        if meuble:
            params["meuble"] = 1

        annonces, source = [], "plateforme"
        try:
            r = self.http.get(f"{config.API_BABILOGIS}/annonces", params=params)
            r.raise_for_status()
            for a in r.json().get("data", []):
                if budget_max and a["loyer"] > int(budget_max):
                    continue          # l'API Dart ignore les budgets ≥ 700 000 : on refiltre
                annonces.append({
                    "reference": a["id"], "titre": a["titre"], "commune": a["commune"], "quartier": a["quartier"],
                    "pieces": a["pieces"], "loyer": a["loyer"], "meuble": a["meuble"],
                    "equipements": a.get("equipements", []), "verdict": a.get("verdict"),
                    "verdict_libelle": a.get("verdict_libelle"),
                    "lien": f"{config.SITE_BABILOGIS}{a.get('url', '/annonces/' + a['id'])}",
                })
        except Exception as e:
            log.warning("API BabiLogis injoignable (%s) : recherche dans les annonces collectées", e)
            source = "annonces_collectees"
            annonces = self._chercher_localement(commune_ok, type_logement, pieces, budget_max, meuble)

        ordre = {"bonne_affaire": 0, "juste": 1, "au_dessus": 2, None: 3}
        annonces.sort(key=lambda a: (ordre.get(a.get("verdict"), 3), a["loyer"]))
        return {"source": source, "nb_resultats": len(annonces), "annonces": annonces[:limite],
                "criteres": {k: v for k, v in dict(commune=commune_ok, type_logement=type_logement, pieces=pieces,
                                                   budget_max=budget_max, meuble=meuble).items() if v}}

    def _chercher_localement(self, commune, type_logement, pieces, budget_max, meuble) -> list[dict]:
        d = self.modele.annonces
        if commune:
            d = d[d.commune == commune]
        if type_logement:
            d = d[d.type_logement == normaliser_type(type_logement, pieces)]
        if pieces:
            d = d[d.pieces == int(pieces)] if int(pieces) < 5 else d[d.pieces >= 5]
        if budget_max:
            d = d[d.prix_fcfa <= int(budget_max)]
        if meuble:
            d = d[d.meuble == 1]
        res = []
        for a in d.itertuples():
            v = verdict(a.prix_fcfa, a.estimation)
            res.append({"reference": "WEB-" + str(a.id).split("-")[-1], "titre": f"{a.type_logement.title()} {int(a.pieces)} pièce(s)",
                        "commune": a.commune, "quartier": a.quartier if isinstance(a.quartier, str) else None,
                        "pieces": int(a.pieces), "loyer": int(a.prix_fcfa), "meuble": bool(a.meuble),
                        "verdict": v, "verdict_libelle": LIBELLES[v], "lien": a.url})
        return res

    # ── 2. Estimation ─────────────────────────────────────────────────────
    def estimer_loyer(self, commune: str, type_logement: str, pieces: int, meuble: bool = False,
                      equipements: list[str] | None = None, quartier: str | None = None) -> dict:
        try:
            return self.modele.estimer(commune, type_logement, pieces, meuble, equipements, quartier)
        except ValueError as e:
            return {"erreur": str(e)}

    # ── 3. Verdict sur un prix ────────────────────────────────────────────
    def evaluer_prix(self, commune: str, type_logement: str, pieces: int, loyer: int, meuble: bool = False,
                     equipements: list[str] | None = None, quartier: str | None = None) -> dict:
        try:
            return self.modele.estimer(commune, type_logement, pieces, meuble, equipements, quartier, loyer=loyer)
        except ValueError as e:
            return {"erreur": str(e)}

    # ── 4. Statistiques de marché ─────────────────────────────────────────
    def statistiques_commune(self, commune: str, type_logement: str | None = None) -> dict:
        try:
            return self.modele.statistiques(commune, type_logement)
        except ValueError as e:
            return {"erreur": str(e)}

    def comparer_communes(self, pieces: int = 3, type_logement: str | None = None) -> dict:
        d = self.modele.annonces[self.modele.annonces.pieces == int(pieces)]
        if type_logement:
            d = d[d.type_logement == normaliser_type(type_logement, pieces)]
        t = d.groupby("commune")["prix_fcfa"].agg(["size", "median"]).sort_values("median")
        return {"pieces": int(pieces), "classement": [
            {"commune": c, "annonces": int(r["size"]), "loyer_median": int(round(r["median"] / 5000) * 5000)}
            for c, r in t.iterrows() if r["size"] >= 3]}

    def appeler(self, nom: str, arguments: dict) -> dict:
        f = getattr(self, nom, None)
        if nom.startswith("_") or nom == "appeler" or not callable(f):
            return {"erreur": f"Outil inconnu : {nom}"}
        try:
            return f(**{k: v for k, v in (arguments or {}).items() if v is not None})
        except TypeError as e:
            return {"erreur": f"Arguments invalides pour {nom} : {e}"}


DECLARATIONS = [
    {"name": "rechercher_annonces",
     "description": "Cherche des logements à louer publiés sur BabiLogis (Abidjan). Renvoie au plus 'limite' annonces "
                    "triées des meilleures affaires aux plus chères, avec leur verdict de prix et un lien.",
     "parameters": {"type": "object", "properties": {
         "commune": {"type": "string", "enum": COMMUNES, "description": "Commune d'Abidjan"},
         "type_logement": {"type": "string", "enum": ["studio", "appartement", "villa"]},
         "pieces": {"type": "integer", "description": "Nombre de pièces (chambres + salon). 5 = 5 et plus."},
         "budget_max": {"type": "integer", "description": "Loyer mensuel maximum en FCFA"},
         "meuble": {"type": "boolean"},
         "limite": {"type": "integer", "description": "Nombre d'annonces à renvoyer (1 à 10, défaut 5)"}}}},
    {"name": "estimer_loyer",
     "description": "Estime le loyer mensuel de marché (FCFA) d'un logement avec le modèle BabiLogis (Random Forest "
                    "entraîné sur les annonces collectées). Renvoie une estimation centrale et une fourchette.",
     "parameters": {"type": "object", "required": ["commune", "type_logement", "pieces"], "properties": {
         "commune": {"type": "string", "enum": COMMUNES},
         "type_logement": {"type": "string", "enum": ["chambre", "studio", "appartement", "villa"]},
         "pieces": {"type": "integer"}, "meuble": {"type": "boolean"},
         "equipements": {"type": "array", "items": {"type": "string"},
                         "description": "ex. piscine, jardin, parking, gardien, climatisation, groupe électrogène, forage"},
         "quartier": {"type": "string", "description": "ex. Riviera, Angré, Zone 4"}}}},
    {"name": "evaluer_prix",
     "description": "Dit si un loyer demandé est une bonne affaire, un prix juste ou au-dessus du marché, "
                    "en le comparant à l'estimation du modèle (seuils 85 % et 115 %).",
     "parameters": {"type": "object", "required": ["commune", "type_logement", "pieces", "loyer"], "properties": {
         "commune": {"type": "string", "enum": COMMUNES},
         "type_logement": {"type": "string", "enum": ["chambre", "studio", "appartement", "villa"]},
         "pieces": {"type": "integer"}, "loyer": {"type": "integer", "description": "Loyer mensuel demandé (FCFA)"},
         "meuble": {"type": "boolean"}, "equipements": {"type": "array", "items": {"type": "string"}},
         "quartier": {"type": "string"}}}},
    {"name": "statistiques_commune",
     "description": "Statistiques du marché locatif d'une commune : loyer médian, écart interquartile, loyer médian "
                    "par nombre de pièces, quartiers les plus présents.",
     "parameters": {"type": "object", "required": ["commune"], "properties": {
         "commune": {"type": "string", "enum": COMMUNES},
         "type_logement": {"type": "string", "enum": ["chambre", "studio", "appartement", "villa"]}}}},
    {"name": "comparer_communes",
     "description": "Classe les communes de la moins chère à la plus chère pour un nombre de pièces donné.",
     "parameters": {"type": "object", "properties": {
         "pieces": {"type": "integer"},
         "type_logement": {"type": "string", "enum": ["chambre", "studio", "appartement", "villa"]}}}},
]
