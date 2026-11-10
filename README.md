# BabiLogis — Service IA (Blocs 4 et 5)

Service Python **FastAPI** qui fournit :

- **`POST /estimer`** : l'estimation du loyer par le modèle du Bloc 2 (Random Forest). Le serveur Dart l'appelle à chaque estimation ou publication d'annonce.
- **`POST /agent/discuter`** : **Babi**, l'agent IA « Assistant logement ».
  - Il utilise **Groq** (ou DeepSeek / Gemini, au choix) avec appel d'outils (*function calling*) : `rechercher_annonces`, `estimer_loyer`, `evaluer_prix`, `statistiques_commune` et `comparer_communes`.
  - Sans clé ou si le LLM est indisponible, il passe automatiquement à un **agent par règles** qui utilise les mêmes outils.
- Les autres adresses sont `GET /sante`, `POST /verdict`, `GET /marche/{commune}` et `GET /communes`.
- La documentation interactive est sur http://127.0.0.1:8001/docs.

## Lancer

```bash
cd ~/Desktop/babilog/ia_service
./lancer.sh
```

Au premier lancement, le script :

- crée `.venv` ;
- installe `requirements.txt` ;
- copie `.env.exemple` en `.env`.

### Activer Groq (LLM de l'agent)

1. Ouvre le fichier `.env` (créé au premier lancement) et colle ta clé : `GROQ_API_KEY=gsk_...` (ou lance `bash configurer_cle.sh`). Ne la partage jamais et ne la mets pas sur GitHub (le fichier `.env` est ignoré par git).
2. Relance `./lancer.sh`. `GET /sante` doit maintenant afficher `"agent": "groq (openai/gpt-oss-120b)"`.

Le modèle se change avec `GROQ_MODEL` (par ex. `openai/gpt-oss-20b`). Pour DeepSeek : `IA_FOURNISSEUR=deepseek` et `DEEPSEEK_API_KEY=sk-...`. Pour Gemini : `IA_FOURNISSEUR=gemini` et `GEMINI_API_KEY=AIza...`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Les 10 tests couvrent :

- l'estimation, la validation et le verdict ;
- la compréhension des phrases ;
- la boucle d'outils de Groq/DeepSeek et de Gemini (simulées) ;
- le passage à l'agent par règles.

## Les 3 services ensemble

| Terminal | Commande | Adresse |
|---|---|---|
| 1. Service IA | `cd ia_service && ./lancer.sh` | http://127.0.0.1:8001 |
| 2. API et site Dart | `cd serveur && dart run bin/serveur.dart` | http://127.0.0.1:8000 |
| 3. Site (développement) | `cd babilogi && flutter run -d chrome` | page **Assistant IA** |

Le serveur Dart cherche le service IA sur `http://127.0.0.1:8001`. Pour le désactiver, utilise `IA_SERVICE_URL=aucun`. S'il ne répond pas, les estimations repassent sur les règles simples.

## Données

Le dossier `donnees/` contient :

- `annonces_propres.csv` et `loyers_reference.csv`, issus du Bloc 1 ;
- `modele_loyer.joblib` et `modele_loyer_meta.json`, issus du Bloc 2.

Si ta version de scikit-learn diffère de celle du modèle, le service **réentraîne** le modèle au démarrage, à partir de `annonces_propres.csv`. Ça prend quelques secondes.
