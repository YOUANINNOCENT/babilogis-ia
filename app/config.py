"""Configuration du service IA BabiLogis (variables d'environnement ou fichier .env)."""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _charger_env(fichier: Path) -> None:
    """Lit un fichier .env simple (CLE=valeur) sans dépendance externe."""
    if not fichier.exists():
        return
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, valeur = ligne.split("=", 1)
        os.environ.setdefault(cle.strip(), valeur.strip().strip('"').strip("'"))


_charger_env(RACINE / ".env")

DONNEES = Path(os.environ.get("BABILOGIS_DONNEES", RACINE / "donnees"))
FICHIER_MODELE = DONNEES / "modele_loyer.joblib"
FICHIER_META = DONNEES / "modele_loyer_meta.json"
FICHIER_ANNONCES = DONNEES / "annonces_propres.csv"
FICHIER_REFERENCE = DONNEES / "loyers_reference.csv"

# API Dart (annonces publiées sur la plateforme)
API_BABILOGIS = os.environ.get("BABILOGIS_API_URL", "http://127.0.0.1:8000/api").rstrip("/")
SITE_BABILOGIS = os.environ.get("BABILOGIS_SITE_URL", "http://127.0.0.1:8000").rstrip("/")

# Gemini (Google AI Studio : https://aistudio.google.com/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Fournisseur du LLM : "groq" (par défaut), "deepseek" ou "gemini"
IA_FOURNISSEUR = os.environ.get("IA_FOURNISSEUR", "groq").strip().lower()

# DeepSeek (https://platform.deepseek.com/api_keys)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Groq (https://console.groq.com/keys)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
