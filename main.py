
"""
=============================================================================
SMART INVEST — Projet KBC (fichier unique, code complet)
=============================================================================
Service qui analyse le compte courant d'un client, prédit ses rentrées et
dépenses (IA : régression saisonnière Ridge + Holt-Winters + saisonnier naïf,
combinés en ensemble validé hors échantillon) et investit automatiquement
chaque fin de mois l'argent qui dort au-delà du matelas de sécurité, dans un ou
plusieurs des trois produits : compte épargne (2 %), placement à terme (3 %),
fonds géré (~5 % espéré, avec risque).
Installation :
pip install numpy pandas scikit-learn statsmodels scipy streamlit plotly
Utilisation :
streamlit run smart_invest_kbc.py -> interface web complète
python smart_invest_kbc.py -> rapport (client démo « stable »)
python smart_invest_kbc.py --demo famille -> autre client de démonstration
python smart_invest_kbc.py --fichier client.json -> votre utilisateur par défaut
Format des données (CSV ou JSON) :
JSON : {"client": {"nom": ..., "solde_initial": ..., "date_ouverture": "AAAA-MM-JJ",
"matelas": ... (optionnel)},
"transactions": [{"date": "AAAA-MM-JJ", "montant": -920.0,
"libelle": "Loyer", "categorie": "Logement"}, ...]}
CSV : date;montant;libelle;categorie (montant négatif = dépense)
Sections du fichier :
Configuration 7. Produits & portefeuille
Modèles métier 8. Moteur de décision
Chargement des données 9. Proposition par défaut
Prédiction (IA) 10. Générateur de clients de démo
Analyse & éligibilité 11. Simulations (rejeu + Monte Carlo)
Profil de risque 12. Interface Streamlit 13. Ligne de commande
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import streamlit as st
import unicodedata
import warnings
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

=============================================================================
1. CONFIGURATION
=============================================================================
Paramètres métier du service « Smart Invest ».
#
Toutes les valeurs chiffrées du service sont centralisées ici pour pouvoir les
ajuster sans toucher à la logique. Les montants sont des Decimal (jamais de
float pour de l'argent).
#
⚠ Les valeurs fiscales (précompte mobilier, exonération livret, TOB) changent
chaque année en Belgique : à vérifier avant toute présentation « officielle ».

CENT = Decimal("0.01")

@dataclass(frozen=True)
class EligibilityConfig:
# Âge minimum du compte (en mois) pour disposer d'un historique suffisant.
min_history_months: int = 24
# Coefficient de variation max des dépenses mensuelles (écart-type / moyenne),
# calculé hors mois exceptionnels. Au-delà : service « en attente ».
max_expense_cv: float = 0.30
# Si plus de X % des mois sont exceptionnels, le compte est jugé instable.
max_exceptional_ratio: float = 0.25
# Capacité d'épargne moyenne minimale (rentrées - dépenses) par mois.
min_mean_monthly_saving: Decimal = Decimal("50")

@dataclass(frozen=True)
class DecisionConfig:
# Quantile de sécurité : 1.645 ≈ 95 % unilatéral (loi normale).
safety_k: float = 1.645
# Marge « extras » minimale, en % des dépenses prévues du mois suivant.
extra_margin_pct: Decimal = Decimal("0.10")
# En dessous de ce montant, on n'investit rien ce mois-là
# (frais et complexité > gain attendu).
min_investment: Decimal = Decimal("50")
# Plafond d'investissement mensuel (garde-fou).
max_investment: Decimal = Decimal("25000")
# Horizon (mois) sur lequel une dépense annoncée est « provisionnée ».
# (la provision tient compte de ce que les mois intermédiaires rapporteront)
planned_expense_lookahead_months: int = 12
# Matelas proposé par défaut = X mois de dépenses moyennes.
default_cushion_months: Decimal = Decimal("1.5")

@dataclass(frozen=True)
class ProductConfig:
# --- Compte épargne (livret) ---
savings_rate: Decimal = Decimal("0.02") # 2 % / an, retrait libre
# --- Placement à terme ---
term_rate: Decimal = Decimal("0.03") # 3 % / an
term_months: int = 12
term_auto_renew: bool = True # reconduction tacite à l'échéance (sinon retour sur le compte épargne)
term_early_break_fee_pct: Decimal = Decimal("0.01") # 1 % du capital + perte des intérêts courus
# --- Fonds géré (argent confié à des gestionnaires) ---
fund_expected_return: Decimal = Decimal("0.05") # rendement annuel espéré
fund_volatility: float = 0.12 # volatilité annuelle
fund_entry_fee_pct: Decimal = Decimal("0.01") # frais d'entrée
fund_annual_fee_pct: Decimal = Decimal("0.012") # frais de gestion annuels
fund_exit_tax_pct: Decimal = Decimal("0.0132") # TOB sur fonds de capitalisation (BE)
fund_settlement_days: int = 3 # délai de règlement (J+3)
fund_daily_liquidity_cap: Decimal = Decimal("10000") # au-delà : vente en plusieurs jours
fund_unannounced_exit_fee_pct: Decimal = Decimal("0.005") # retrait non anticipé (< 30 j)
fund_notice_days: int = 30
# --- Fiscalité (Belgique, à vérifier chaque année) ---
withholding_tax: Decimal = Decimal("0.30") # précompte mobilier (placement à terme)
savings_withholding_tax: Decimal = Decimal("0.15") # compte épargne réglementé, au-delà de l'exonération
savings_tax_exemption: Decimal = Decimal("1050") # intérêts livret exonérés / an / personne

@dataclass(frozen=True)
class ForecastConfig:
horizon_months: int = 12
backtest_months: int = 6 # validation « rolling origin » sur les N derniers mois
ridge_alpha: float = 1.0
outlier_mad_threshold: float = 3.5 # score robuste pour les mois exceptionnels
random_seed: int = 42

@dataclass(frozen=True)
class AppConfig:
eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
decision: DecisionConfig = field(default_factory=DecisionConfig)
products: ProductConfig = field(default_factory=ProductConfig)
forecast: ForecastConfig = field(default_factory=ForecastConfig)

DEFAULT_CONFIG = AppConfig()

def money(value) -> Decimal:
"""Convertit n'importe quelle valeur numérique en Decimal arrondi au cent."""
if isinstance(value, Decimal):
return value.quantize(CENT)
return Decimal(str(round(float(value), 2))).quantize(CENT)

def eur(value, decimals: int = 0) -> str:
"""Format belge : 1 234,56 (espace insécable fine pour les milliers)."""
s = f"{float(value):,.{decimals}f}"
return s.replace(",", "\u202f").replace(".", ",")

=============================================================================
2. MODÈLES MÉTIER
=============================================================================
Objets métier partagés par tous les modules.

class ProductType(str, Enum):
SAVINGS = "epargne"
TERM = "terme"
FUND = "fonds"
@property
def label(self) -> str:
return {
ProductType.SAVINGS: "Compte épargne",
ProductType.TERM: "Placement à terme",
ProductType.FUND: "Fonds géré",
}[self]
@property
def short_description(self) -> str:
return {
ProductType.SAVINGS: "Taux plus bas, argent disponible à tout moment, sans risque.",
ProductType.TERM: "Taux plus élevé, argent bloqué pendant la durée du terme.",
ProductType.FUND: "Argent confié à des gestionnaires : rendement espéré plus élevé, "
"risque de perte, retrait possible mais pas toujours avantageux.",
}[self]

class RiskProfile(str, Enum):
DEFENSIVE = "defensif"
NEUTRAL = "neutre"
DYNAMIC = "dynamique"
@property
def label(self) -> str:
return {"defensif": "Défensif", "neutre": "Neutre", "dynamique": "Dynamique"}[self.value]

class EligibilityStatus(str, Enum):
ELIGIBLE = "eligible"
PENDING = "en_attente" # compte trop instable pour l'instant : on réanalyse plus tard
REFUSED = "refuse" # pas de capacité d'épargne
TOO_YOUNG = "historique_insuffisant"
@property
def label(self) -> str:
return {
"eligible": "Éligible",
"en_attente": "En attente (compte trop instable)",
"refuse": "Non proposé (pas de capacité d'épargne)",
"historique_insuffisant": "Historique insuffisant (< 24 mois)",
}[self.value]

@dataclass
class PlannedExpense:
"""Grosse dépense annoncée à l'avance par le client."""
label: str
amount: Decimal
due: date
def __post_init__(self):
self.amount = Decimal(str(self.amount))
if self.amount <= 0:
raise ValueError("Le montant d'une dépense prévue doit être positif.")

@dataclass
class Allocation:
"""Répartition de l'investissement mensuel entre les 3 produits (somme = 1)."""
weights: dict[ProductType, Decimal] = field(default_factory=dict)
def __post_init__(self):
self.weights = {ProductType(k): Decimal(str(v)) for k, v in self.weights.items() if Decimal(str(v)) > 0}
total = sum(self.weights.values(), Decimal("0"))
if not self.weights:
raise ValueError("Au moins un produit doit être choisi.")
if any(v < 0 for v in self.weights.values()):
raise ValueError("Les pourcentages doivent être positifs.")
if abs(total - Decimal("1")) > Decimal("0.001"):
raise ValueError(f"La répartition doit totaliser 100 % (actuellement {total * 100:.1f} %).")
@classmethod
def single(cls, product: ProductType) -> "Allocation":
return cls({product: Decimal("1")})
def describe(self) -> str:
return " · ".join(f"{p.label} {w * 100:.0f} %" for p, w in self.weights.items())

@dataclass
class ClientSettings:
"""Réglages du client (proposés par défaut par le service, modifiables)."""
cushion: Decimal # matelas de sécurité toujours laissé sur le compte courant
investment_rate: Decimal # part (0-1) du surplus investissable réellement investie
allocation: Allocation
risk_profile: RiskProfile = RiskProfile.NEUTRAL
planned_expenses: list[PlannedExpense] = field(default_factory=list)

=============================================================================
3. CHARGEMENT DES DONNÉES
=============================================================================
Chargement et validation de l'historique de transactions d'un client.
#
Formats acceptés (souples, pour pouvoir brancher votre « utilisateur par défaut ») :
#
* CSV (séparateur « , » ou « ; », décimales « . » ou « , »)
* JSON : soit une liste de transactions, soit un objet
{"client": {...}, "transactions": [...]}
#
Colonnes reconnues (insensible à la casse / aux accents) :
#
* date : date, date_operation, date_valeur, booking_date
* montant : montant, amount, valeur (négatif = dépense, positif = rentrée)
ou bien : debit + credit, ou montant + type ("débit"/"crédit")
* libellé : libelle, description, communication, label (optionnel)
* catégorie : categorie, category (optionnel)
#
Champs client optionnels (JSON) : nom, solde_initial / opening_balance,
solde_actuel / current_balance, date_ouverture / opening_date, matelas / cushion.

class DataValidationError(ValueError):
"""Données client invalides ou inexploitables."""

_ALIASES = {
"date": ["date", "date_operation", "dateoperation", "date_valeur", "booking_date", "jour", "transaction_date"],
"amount": ["montant", "amount", "valeur", "value", "somme"],
"debit": ["debit", "sortie", "depense", "out"],
"credit": ["credit", "entree", "rentree", "in"],
"type": ["type", "sens", "direction"],
"description": ["libelle", "description", "communication", "label", "intitule", "motif"],
"category": ["categorie", "category", "cat"],
}

def _norm(name: str) -> str:
s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
return s.strip().lower().replace(" ", "_").replace("-", "_")

@dataclass
class ClientData:
transactions: pd.DataFrame # date, amount, description, category
name: str = "Client"
opening_balance: Decimal = Decimal("0")
opening_balance_estimated: bool = False
opening_date: date | None = None
cushion: Decimal | None = None # matelas déjà choisi par le client (sinon proposé)
warnings: list[str] = field(default_factory=list)
@property
def current_balance(self) -> Decimal:
return money(self.opening_balance + money(self.transactions["amount"].sum()))
def balance_series(self) -> pd.Series:
"""Solde quotidien reconstitué (fin de journée)."""
daily = self.transactions.groupby(self.transactions["date"].dt.normalize())["amount"].sum()
return float(self.opening_balance) + daily.cumsum()

def _to_number(series: pd.Series) -> pd.Series:
if pd.api.types.is_numeric_dtype(series):
return series.astype(float)
cleaned = (
series.astype(str)
.str.replace(" ", "", regex=False)
.str.replace(" ", "", regex=False)
.str.replace("€", "", regex=False)
.str.replace("EUR", "", regex=False)
)
# "1.234,56" -> "1234.56" ; "1234,56" -> "1234.56"
has_both = cleaned.str.contains(r".") & cleaned.str.contains(",")
cleaned = cleaned.where(~has_both, cleaned.str.replace(".", "", regex=False))
cleaned = cleaned.str.replace(",", ".", regex=False)
return pd.to_numeric(cleaned, errors="coerce")

def _parse_dates(series: pd.Series) -> pd.Series:
"""Dates ISO (AAAA-MM-JJ) lues telles quelles, les autres au format belge (JJ/MM/AAAA)."""
s = series.astype(str).str.strip().str[:19]
iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
if iso.any():
out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
if (~iso).any():
out[~iso] = pd.to_datetime(s[~iso], errors="coerce", dayfirst=True, format="mixed")
return out

def _pick(columns: dict[str, str], key: str) -> str | None:
for alias in _ALIASES[key]:
if alias in columns:
return columns[alias]
return None

def normalize_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
warnings: list[str] = []
if raw is None or raw.empty:
raise DataValidationError("Aucune transaction trouvée.")
columns = {_norm(c): c for c in raw.columns}
date_col = _pick(columns, "date")
if not date_col:
raise DataValidationError(f"Colonne de date introuvable. Colonnes reçues : {list(raw.columns)}")
amount_col = _pick(columns, "amount")
debit_col, credit_col, type_col = _pick(columns, "debit"), _pick(columns, "credit"), _pick(columns, "type")
if debit_col and credit_col:
amount = _to_number(raw[credit_col]).fillna(0).abs() - _to_number(raw[debit_col]).fillna(0).abs()
elif amount_col:
amount = _to_number(raw[amount_col])
if type_col and (amount >= 0).all():
kind = raw[type_col].astype(str).map(_norm)
is_out = kind.str.startswith(("deb", "sort", "dep", "out", "withdraw"))
amount = amount.where(~is_out, -amount)
else:
raise DataValidationError("Colonne de montant introuvable (montant/amount ou debit+credit).")
dates = _parse_dates(raw[date_col])
desc_col, cat_col = _pick(columns, "description"), _pick(columns, "category")
df = pd.DataFrame({
"date": dates,
"amount": amount,
"description": raw[desc_col].astype(str).str.strip() if desc_col else "",
"category": raw[cat_col].astype(str).str.strip() if cat_col else "",
})
bad = df["date"].isna() | df["amount"].isna()
if bad.any():
warnings.append(f"{int(bad.sum())} ligne(s) ignorée(s) : date ou montant illisible.")
df = df[~bad]
zero = df["amount"] == 0
if zero.any():
df = df[~zero]
if df.empty:
raise DataValidationError("Aucune transaction valide après nettoyage.")
if (df["amount"].abs() > 1_000_000).any():
warnings.append("Montant(s) > 1 000 000 € détecté(s) : vérifiez les données.")
dup = df.duplicated(keep="first")
if dup.any():
warnings.append(f"{int(dup.sum())} doublon(s) exact(s) conservé(s) (vérifiez s'il s'agit d'erreurs).")
df["amount"] = df["amount"].round(2)
df = df.sort_values("date", kind="stable").reset_index(drop=True)
return df, warnings

def _estimate_opening_balance(tx: pd.DataFrame) -> Decimal:
"""Si le solde initial est inconnu : le plus petit solde garantissant un compte jamais à découvert."""
running = tx["amount"].cumsum()
lowest = float(running.min()) if len(running) else 0.0
return money(max(0.0, -lowest) + 500.0)

def build_client(transactions: pd.DataFrame, meta: dict | None = None) -> ClientData:
meta = {_norm(k): v for k, v in (meta or {}).items()}
tx, warnings = normalize_transactions(transactions)
def first(keys):
for k in keys:
if k in meta and meta[k] not in (None, ""):
return meta[k]
return None
opening = first("solde_initial", "opening_balance", "solde_depart")
current = first("solde_actuel", "current_balance", "solde")
estimated = False
if opening is not None:
opening_bal = money(opening)
elif current is not None:
opening_bal = money(Decimal(str(current)) - money(tx["amount"].sum()))
else:
opening_bal = _estimate_opening_balance(tx)
estimated = True
warnings.append(f"Solde initial inconnu : estimé à {opening_bal} € (compte jamais à découvert).")
opening_date = first("date_ouverture", "opening_date", "ouverture")
parsed = _parse_dates(pd.Series([opening_date])).iloc[0] if opening_date else pd.NaT
opening_date = parsed.date() if pd.notna(parsed) else tx["date"].min().date()
cushion = first("matelas", "cushion", "matelas_securite")
return ClientData(
transactions=tx,
name=str(first("nom", "name", "client") or "Client"),
opening_balance=opening_bal,
opening_balance_estimated=estimated,
opening_date=opening_date,
cushion=money(cushion) if cushion is not None else None,
warnings=warnings,
)

def load_client(source: str | Path | bytes | io.IOBase, filename: str | None = None) -> ClientData:
"""Charge un client depuis un chemin, des octets (upload) ou un flux."""
if isinstance(source, (str, Path)):
path = Path(source)
filename = filename or path.name
data = path.read_bytes()
elif isinstance(source, bytes):
data = source
else:
data = source.read()
if isinstance(data, str):
data = data.encode()
filename = (filename or "").lower()
text = data.decode("utf-8-sig", errors="replace")
if filename.endswith(".json") or text.lstrip().startswith(("{", "[")):
try:
payload = json.loads(text)
except json.JSONDecodeError as exc:
raise DataValidationError(f"JSON invalide : {exc}") from exc
if isinstance(payload, list):
return build_client(pd.DataFrame(payload))
tx = payload.get("transactions") or payload.get("operations")
if tx is None:
raise DataValidationError("Le JSON doit contenir une clé 'transactions'.")
meta = payload.get("client") or {k: v for k, v in payload.items() if k not in ("transactions", "operations")}
return build_client(pd.DataFrame(tx), meta)
first_line = text.splitlines()[0] if text else ""
sep = ";" if first_line.count(";") > first_line.count(",") else ","
try:
raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
except Exception as exc: # noqa: BLE001
raise DataValidationError(f"CSV illisible : {exc}") from exc
return build_client(raw)

def monthly_flows(tx: pd.DataFrame, only_complete: bool = True) -> pd.DataFrame:
"""Agrégation mensuelle : rentrées, dépenses (positives), net, nb d'opérations."""
df = tx.copy()
df["month"] = df["date"].dt.to_period("M")
g = df.groupby("month")
out = pd.DataFrame({
"income": g["amount"].apply(lambda s: s[s > 0].sum()),
"expenses": g["amount"].apply(lambda s: -s[s < 0].sum()),
"n_tx": g.size(),
})
full = pd.period_range(out.index.min(), out.index.max(), freq="M")
out = out.reindex(full, fill_value=0.0)
out["net"] = out["income"] - out["expenses"]
if only_complete and len(out) > 1:
last_day = df["date"].max()
if last_day.day < last_day.days_in_month - 2: # dernier mois incomplet
out = out.iloc[:-1]
out.index.name = "month"
return out.astype({"income": float, "expenses": float, "net": float, "n_tx": int})

def history_months(client: ClientData) -> int:
"""Âge du compte en mois pleins."""
start = client.opening_date or client.transactions["date"].min().date()
end = client.transactions["date"].max().date() + timedelta(days=1) # borne exclusive
months = (end.year - start.year) * 12 + (end.month - start.month)
if end.day < start.day:
months -= 1
return int(max(0, months))

=============================================================================
4. PRÉDICTION (IA)
=============================================================================
Moteur de prédiction (« IA ») des rentrées et dépenses mensuelles.
#
Trois modèles complémentaires sont entraînés sur l'historique mensuel, puis
combinés en un ensemble pondéré par leur précision mesurée hors échantillon :
#
1. Régression saisonnière (Ridge) : tendance linéaire + effet propre à
chaque mois de l'année (décembre plus cher, pécule en juin, vacances...).
La régularisation Ridge évite de sur-apprendre avec seulement 2-3 ans.
2. Holt-Winters (lissage exponentiel, tendance amortie + saisonnalité
additive sur 12 mois) : suit mieux les changements de niveau récents.
3. Saisonnier naïf ajusté : même mois des années précédentes, recalé sur
le niveau des 12 derniers mois. Référence simple et robuste.
#
Validation : « rolling origin » — on se place à chacun des N derniers mois,
on entraîne sur le passé uniquement, on prédit le mois suivant et on mesure
l'erreur. Poids de l'ensemble ∝ 1 / MAE². L'écart-type des erreurs de
l'ensemble donne l'intervalle de confiance utilisé par le moteur de décision.

MODEL_LABELS = {
"ridge": "Régression saisonnière (Ridge)",
"holt_winters": "Holt-Winters",
"seasonal_naive": "Saisonnier naïf ajusté",
}

---------------------------------------------------------------------------
Modèles individuels : fit(series) -> predict(n_ahead) -> np.ndarray
---------------------------------------------------------------------------
def _features(periods: pd.PeriodIndex, origin: pd.Period) -> np.ndarray:
t = np.array([(p - origin).n for p in periods], dtype=float)
months = np.array([p.month for p in periods])
dummies = np.zeros((len(periods), 12))
dummies[np.arange(len(periods)), months - 1] = 1.0
return np.column_stack([t, dummies])

class SeasonalRidge:
name = "ridge"
def __init__(self, alpha: float = 1.0):
self.alpha = alpha
def fit(self, series: pd.Series) -> "SeasonalRidge":
self.origin = series.index[0]
self.last = series.index[-1]
X = _features(series.index, self.origin)
self.scaler = StandardScaler().fit(X)
# Si moins de 24 mois : on régularise davantage l'effet saisonnier.
alpha = self.alpha * (2.0 if len(series) < 24 else 1.0)
self.model = Ridge(alpha=alpha).fit(self.scaler.transform(X), series.values)
self.fitted_ = self.model.predict(self.scaler.transform(X))
return self
def predict(self, n_ahead: int) -> np.ndarray:
future = pd.period_range(self.last + 1, periods=n_ahead, freq="M")
return self.model.predict(self.scaler.transform(_features(future, self.origin)))
def seasonal_profile(self) -> pd.Series:
"""Effet de chaque mois de l'année (en €) par rapport à la moyenne."""
coefs = self.model.coef_[1:] / self.scaler.scale_[1:]
effect = pd.Series(coefs, index=range(1, 13))
# Ramène chaque coefficient à l'écart en € pour un mois « actif » vs moyenne.
return effect - effect.mean()
def trend_per_month(self) -> float:
return float(self.model.coef_[0] / self.scaler.scale_[0])

class HoltWinters:
name = "holt_winters"
def fit(self, series: pd.Series) -> "HoltWinters":
from statsmodels.tsa.holtwinters import ExponentialSmoothing
self.last = series.index[-1]
y = series.astype(float).values
seasonal = "add" if len(y) >= 24 else None
with warnings.catch_warnings():
warnings.simplefilter("ignore")
model = ExponentialSmoothing(
y, trend="add", damped_trend=True, seasonal=seasonal,
seasonal_periods=12 if seasonal else None, initialization_method="estimated",
)
self.res = model.fit(optimized=True)
return self
def predict(self, n_ahead: int) -> np.ndarray:
return np.asarray(self.res.forecast(n_ahead), dtype=float)

class SeasonalNaive:
name = "seasonal_naive"
def fit(self, series: pd.Series) -> "SeasonalNaive":
self.series = series.astype(float)
self.last = series.index[-1]
recent = self.series.iloc[-12:]
self.level_shift = float(recent.mean() - self.series.mean())
self.by_month = self.series.groupby(self.series.index.month).mean()
return self
def predict(self, n_ahead: int) -> np.ndarray:
future = pd.period_range(self.last + 1, periods=n_ahead, freq="M")
base = float(self.series.iloc[-12:].mean())
out = []
for p in future:
if p.month in self.by_month.index:
out.append(self.by_month[p.month] + self.level_shift)
else:
out.append(base)
return np.array(out)

def _make_models(cfg: ForecastConfig):
return [SeasonalRidge(cfg.ridge_alpha), HoltWinters(), SeasonalNaive()]

---------------------------------------------------------------------------
Ensemble
---------------------------------------------------------------------------
@dataclass
class SeriesForecast:
name: str
history: pd.Series
forecast: pd.Series # moyenne de l'ensemble
sigma: float # écart-type de l'erreur à 1 mois
weights: dict[str, float]
backtest: pd.DataFrame # erreurs par modèle et par origine
metrics: pd.DataFrame # MAE / MAPE / RMSE par modèle + ensemble
seasonal_profile: pd.Series
trend_per_month: float
fitted: pd.Series # ajustement in-sample (Ridge) pour détection d'anomalies
per_model: dict[str, pd.Series] = field(default_factory=dict)
def sigma_at(self, h: int) -> float:
"""Incertitude à h mois : croît doucement avec l'horizon."""
return self.sigma * (1.0 + 0.05 * (h - 1))

def _safe_fit_predict(model, series: pd.Series, n: int) -> np.ndarray | None:
try:
return model.fit(series).predict(n)
except Exception: # noqa: BLE001 - un modèle qui échoue est simplement écarté
return None

def forecast_series(series: pd.Series, name: str, cfg: ForecastConfig | None = None,
horizon: int | None = None) -> SeriesForecast:
cfg = cfg or ForecastConfig()
horizon = horizon or cfg.horizon_months
series = series.astype(float)
n = len(series)
if n < 6:
raise ValueError("Au moins 6 mois d'historique sont nécessaires pour prédire.")
# --- Validation hors échantillon (rolling origin) ---
n_bt = int(min(cfg.backtest_months, max(1, n - 12)))
records = []
for k in range(n - n_bt, n):
train, actual = series.iloc[:k], series.iloc[k]
for model in _make_models(cfg):
pred = _safe_fit_predict(model, train, 1)
if pred is not None and np.isfinite(pred[0]):
records.append({"origin": series.index[k], "model": model.name,
"pred": float(pred[0]), "actual": float(actual)})
bt = pd.DataFrame(records)
if bt.empty:
raise RuntimeError("Aucun modèle n'a pu être entraîné.")
bt["error"] = bt["actual"] - bt["pred"]
mae = bt.groupby("model")["error"].apply(lambda e: float(np.mean(np.abs(e))))
inv = 1.0 / np.maximum(mae, 1e-6) ** 2
weights = (inv / inv.sum()).to_dict()
# Erreurs de l'ensemble (mêmes pondérations) pour calibrer l'incertitude.
piv = bt.pivot_table(index="origin", columns="model", values="pred")
w = pd.Series(weights).reindex(piv.columns).fillna(0)
ens_pred = (piv * w).sum(axis=1) / (piv.notna() * w).sum(axis=1)
actuals = bt.groupby("origin")["actual"].first()
ens_err = actuals - ens_pred
ens_rows = pd.DataFrame({"origin": ens_pred.index, "model": "ensemble", "pred": ens_pred.values,
"actual": actuals.values, "error": ens_err.values})
bt = pd.concat([bt, ens_rows], ignore_index=True)
def _metrics(g):
a = g["actual"].abs().replace(0, np.nan)
return pd.Series({
"MAE (€)": float(np.mean(np.abs(g["error"]))),
"RMSE (€)": float(np.sqrt(np.mean(g["error"] ** 2))),
"MAPE (%)": float(np.nanmean(np.abs(g["error"]) / a) * 100),
})
metrics = bt.groupby("model")[["actual", "error"]].apply(_metrics)
metrics["Poids"] = pd.Series(weights).reindex(metrics.index).fillna(np.nan)
metrics.index = [MODEL_LABELS.get(i, "Ensemble (retenu)") for i in metrics.index]
# Écart-type : RMSE de l'ensemble, avec un plancher (peu de points de validation).
rmse_ens = float(np.sqrt(np.mean(ens_err ** 2)))
floor = 0.03 * float(series.iloc[-12:].mean() or 1.0)
in_sample_resid_std = float(np.std(series.values - SeasonalRidge(cfg.ridge_alpha).fit(series).fitted_, ddof=1))
sigma = max(rmse_ens, floor, in_sample_resid_std)
# --- Modèles finaux sur tout l'historique ---
future_idx = pd.period_range(series.index[-1] + 1, periods=horizon, freq="M")
per_model, preds = {}, []
ridge_final = None
for model in _make_models(cfg):
p = _safe_fit_predict(model, series, horizon)
if p is None or model.name not in weights:
continue
p = np.maximum(p, 0.0) # des rentrées/dépenses ne sont jamais négatives
per_model[model.name] = pd.Series(p, index=future_idx)
preds.append((weights[model.name], p))
if model.name == "ridge":
ridge_final = model
total_w = sum(wt for wt, _ in preds)
ensemble = sum(wt * p for wt, p in preds) / total_w
ridge_final = ridge_final or SeasonalRidge(cfg.ridge_alpha).fit(series)
return SeriesForecast(
name=name, history=series, forecast=pd.Series(ensemble, index=future_idx), sigma=sigma,
weights={k: float(v) for k, v in weights.items()}, backtest=bt, metrics=metrics,
seasonal_profile=ridge_final.seasonal_profile(), trend_per_month=ridge_final.trend_per_month(),
fitted=pd.Series(ridge_final.fitted_, index=series.index), per_model=per_model,
)

@dataclass
class CashflowForecast:
income: SeriesForecast
expenses: SeriesForecast
shock_provision: float # coût mensuel moyen des imprévus (retirés de l'apprentissage)
@property
def horizon_index(self) -> pd.PeriodIndex:
return self.income.forecast.index
def table(self, k: float = 1.645) -> pd.DataFrame:
rows = []
for h, p in enumerate(self.horizon_index, start=1):
inc = float(self.income.forecast[p])
exp = float(self.expenses.forecast[p]) + self.shock_provision
s_net = self.net_sigma(h)
rows.append({
"month": p, "income": inc, "expenses": exp, "net": inc - exp,
"net_low": inc - exp - k * s_net, "net_high": inc - exp + k * s_net,
"expenses_high": exp + k * self.expenses.sigma_at(h),
"income_low": max(0.0, inc - k * self.income.sigma_at(h)),
})
return pd.DataFrame(rows).set_index("month")
def net_sigma(self, h: int = 1) -> float:
return float(np.hypot(self.income.sigma_at(h), self.expenses.sigma_at(h)))

def forecast_cashflows(income: pd.Series, expenses_clean: pd.Series, shock_provision: float,
cfg: ForecastConfig | None = None, horizon: int | None = None) -> CashflowForecast:
return CashflowForecast(
income=forecast_series(income, "Rentrées", cfg, horizon),
expenses=forecast_series(expenses_clean, "Dépenses", cfg, horizon),
shock_provision=float(shock_provision),
)

=============================================================================
5. ANALYSE & ÉLIGIBILITÉ
=============================================================================
Analyse du compte courant : flux mensuels, opérations récurrentes, mois
exceptionnels, indicateurs de stabilité et éligibilité au service.

---------------------------------------------------------------------------
Opérations récurrentes
---------------------------------------------------------------------------
def _key(desc: str) -> str:
s = str(desc).lower()
s = re.sub(r"\d+", "", s) # retire numéros de facture, dates...
s = re.sub(r"[^a-zà-ÿ ]+", " ", s)
return re.sub(r"\s+", " ", s).strip() or "(sans libellé)"

def recurring_operations(tx: pd.DataFrame, n_months: int) -> pd.DataFrame:
"""Détecte les opérations qui reviennent presque chaque mois avec un montant stable."""
df = tx.copy()
df["key"] = df["description"].map(_key)
df["sign"] = np.where(df["amount"] > 0, "Rentrée", "Dépense")
df["month"] = df["date"].dt.to_period("M")
grp = df.groupby(["key", "sign"])
monthly = df.groupby(["key", "sign", "month"])["amount"].sum().abs()
out = pd.DataFrame({
"libellé": grp["description"].agg(lambda s: s.mode().iat[0]),
"mois_présents": monthly.groupby(level=[0, 1]).size(),
"montant_moyen": monthly.groupby(level=[0, 1]).mean(),
"cv_montant": monthly.groupby(level=[0, 1]).agg(lambda s: s.std(ddof=0) / s.mean() if s.mean() else 0),
"total": grp["amount"].sum().abs(),
}).reset_index()
out["fréquence"] = out["mois_présents"] / max(1, n_months)
out["récurrente"] = (out["fréquence"] >= 0.75) & (out["cv_montant"] <= 0.20)
return out.sort_values(["sign", "total"], ascending=[False, False]).reset_index(drop=True)

---------------------------------------------------------------------------
Mois exceptionnels (grosses dépenses imprévues)
---------------------------------------------------------------------------
def detect_exceptional_months(expenses: pd.Series, threshold: float = 3.5, alpha: float = 1.0) -> pd.DataFrame:
"""
Un mois est « exceptionnel » si ses dépenses dépassent nettement ce que la
saisonnalité explique (score robuste basé sur la médiane des écarts absolus).
Les effets saisonniers récurrents (décembre, vacances) ne sont donc pas
considérés comme exceptionnels, seulement les vrais imprévus.
"""
fit = SeasonalRidge(alpha).fit(expenses).fitted_
resid = expenses.values - fit
med = np.median(resid)
mad = np.median(np.abs(resid - med)) or 1.0
score = 0.6745 * (resid - med) / mad
flag = (score > threshold) & (resid > 0.10 * np.median(expenses.values))
excess = np.where(flag, resid - med, 0.0)
return pd.DataFrame({"expenses": expenses.values, "expected": fit, "score": score,
"exceptional": flag, "excess": excess}, index=expenses.index)

def intramonth_drawdowns(client: ClientData, months: pd.PeriodIndex) -> pd.Series:
"""
Creux de trésorerie de chaque mois : de combien le solde descend sous son
niveau du début de mois avant que les rentrées (salaire) n'arrivent.
Le matelas doit tenir compte de ce creux, sinon le compte passerait sous
le matelas au milieu du mois (loyer le 1er, salaire le 25...).
"""
daily = client.balance_series()
out = {}
prev_end = float(client.opening_balance)
for m in months:
start, end = m.to_timestamp(how="start"), m.to_timestamp(how="end")
in_month = daily[(daily.index >= start) & (daily.index <= end)]
if len(in_month):
out[m] = max(0.0, prev_end - float(in_month.min()))
prev_end = float(in_month.iloc[-1])
else:
out[m] = 0.0
return pd.Series(out, dtype=float)

def exceptional_transactions(tx: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
"""Les plus grosses dépenses des mois exceptionnels (pour expliquer au client)."""
if not months:
return tx.iloc[0:0]
df = tx[tx["date"].dt.to_period("M").isin(months) & (tx["amount"] < 0)]
return df.sort_values("amount").groupby(df["date"].dt.to_period("M")).head(2)

---------------------------------------------------------------------------
Rapport d'analyse complet
---------------------------------------------------------------------------
@dataclass
class AccountAnalysis:
client: ClientData
monthly: pd.DataFrame
anomalies: pd.DataFrame
recurring: pd.DataFrame
drawdowns: pd.Series
forecast: CashflowForecast | None
history_months: int
mean_income: float
mean_expenses: float
mean_net: float
std_expenses_clean: float
cv_expenses: float
exceptional_ratio: float
income_regularity: float
savings_rate: float
shock_provision: float
intramonth_need: float # creux de trésorerie typique (p90, 12 derniers mois)
stability_score: int
status: EligibilityStatus
reasons: list[str] = field(default_factory=list)
@property
def eligible(self) -> bool:
return self.status == EligibilityStatus.ELIGIBLE
@property
def current_balance(self) -> Decimal:
return self.client.current_balance
@property
def last_month(self) -> pd.Period:
return self.monthly.index[-1]

def _stability_score(cv: float, regularity: float, exc_ratio: float, savings_rate: float) -> int:
s = 100.0
s -= min(45.0, cv * 150) # variabilité des dépenses
s -= min(25.0, max(0.0, 0.9 - regularity) * 40) # revenus non réguliers
s -= min(15.0, exc_ratio * 60) # imprévus fréquents
s -= 15.0 if savings_rate <= 0 else max(0.0, 10 - savings_rate * 50)
return int(np.clip(round(s), 0, 100))

def analyze_account(client: ClientData, cfg: AppConfig = DEFAULT_CONFIG,
with_forecast: bool = True) -> AccountAnalysis:
ecfg = cfg.eligibility
monthly = monthly_flows(client.transactions)
months_hist = history_months(client)
n = len(monthly)
# Mois exceptionnels + série de dépenses « nettoyée » pour l'apprentissage.
if n >= 6:
anomalies = detect_exceptional_months(monthly["expenses"], cfg.forecast.outlier_mad_threshold,
cfg.forecast.ridge_alpha)
else:
anomalies = pd.DataFrame({"expenses": monthly["expenses"], "expected": monthly["expenses"],
"score": 0.0, "exceptional": False, "excess": 0.0}, index=monthly.index)
clean_exp = monthly["expenses"] - anomalies["excess"]
# Provision mensuelle pour imprévus = coût moyen des imprévus observés.
shock_provision = float(anomalies["excess"].sum() / max(1, n))
drawdowns = intramonth_drawdowns(client, monthly.index)
# Hors mois exceptionnels (déjà couverts par la marge et la provision d'imprévus)
normal_dd = drawdowns[~anomalies["exceptional"].reindex(drawdowns.index, fill_value=False)]
intramonth_need = float(np.percentile(normal_dd.iloc[-12:], 90)) if len(normal_dd) else 0.0
recurring = recurring_operations(client.transactions, n)
inc_total = float(monthly["income"].sum())
rec_income = float(recurring.loc[recurring["récurrente"] & (recurring["sign"] == "Rentrée"), "total"].sum())
income_regularity = rec_income / inc_total if inc_total > 0 else 0.0
mean_income = float(monthly["income"].mean())
mean_exp = float(monthly["expenses"].mean())
mean_net = mean_income - mean_exp
std_clean = float(clean_exp.std(ddof=1)) if n > 1 else 0.0
cv = std_clean / float(clean_exp.mean()) if clean_exp.mean() else 1.0
exc_ratio = float(anomalies["exceptional"].mean()) if n else 0.0
savings_rate = mean_net / mean_income if mean_income > 0 else -1.0
# --- Éligibilité (ordre : historique > capacité d'épargne > stabilité) ---
reasons: list[str] = []
if months_hist < ecfg.min_history_months:
status = EligibilityStatus.TOO_YOUNG
reasons.append(f"Le compte a {months_hist} mois d'historique ; il en faut au moins "
f"{ecfg.min_history_months} pour une analyse fiable. "
f"Proposition possible dans {ecfg.min_history_months - months_hist} mois.")
elif money(mean_net) < ecfg.min_mean_monthly_saving:
status = EligibilityStatus.REFUSED
reasons.append(f"En moyenne, les dépenses ({eur(mean_exp)} €) sont supérieures ou trop proches "
f"des rentrées ({eur(mean_income)} €) : pas de surplus à investir.")
else:
status = EligibilityStatus.ELIGIBLE
if cv > ecfg.max_expense_cv:
status = EligibilityStatus.PENDING
reasons.append(f"Dépenses trop variables : coefficient de variation {cv:.0%} "
f"(max {ecfg.max_expense_cv:.0%}).")
if income_regularity < 0.60:
status = EligibilityStatus.PENDING
reasons.append(f"Rentrées irrégulières : seulement {income_regularity:.0%} des rentrées "
f"proviennent de sources régulières (min 60 %).")
if exc_ratio > ecfg.max_exceptional_ratio:
status = EligibilityStatus.PENDING
reasons.append(f"Trop de mois exceptionnels ({exc_ratio:.0%} des mois).")
if status == EligibilityStatus.PENDING:
reasons.append("Le compte sera réanalysé automatiquement chaque mois ; le service sera "
"proposé dès que le comportement se stabilise.")
else:
reasons.append(f"Historique de {months_hist} mois, capacité d'épargne moyenne de "
f"{eur(mean_net)} €/mois, dépenses stables (CV {cv:.0%}).")
forecast = None
if with_forecast and n >= 12:
forecast = forecast_cashflows(monthly["income"], clean_exp, shock_provision, cfg.forecast)
return AccountAnalysis(
client=client, monthly=monthly.assign(expenses_clean=clean_exp), anomalies=anomalies,
recurring=recurring, forecast=forecast, history_months=months_hist, drawdowns=drawdowns,
mean_income=mean_income, mean_expenses=mean_exp, mean_net=mean_net,
std_expenses_clean=std_clean, cv_expenses=cv, exceptional_ratio=exc_ratio,
income_regularity=income_regularity, savings_rate=savings_rate, shock_provision=shock_provision,
intramonth_need=intramonth_need,
stability_score=_stability_score(cv, income_regularity, exc_ratio, savings_rate),
status=status, reasons=reasons,
)

def category_breakdown(tx: pd.DataFrame) -> pd.DataFrame:
exp = tx[tx["amount"] < 0].copy()
exp["category"] = exp["category"].replace("", "Non catégorisé").fillna("Non catégorisé")
n_months = max(1, exp["date"].dt.to_period("M").nunique())
out = (-exp.groupby("category")["amount"].sum() / n_months).sort_values(ascending=False)
return out.rename("moyenne_mensuelle").to_frame()

=============================================================================
6. PROFIL DE RISQUE
=============================================================================
Questionnaire de profil d'investisseur (version simplifiée d'un test MiFID).
#
En Belgique, une banque doit évaluer les connaissances, la situation et la
tolérance au risque du client avant de lui proposer un produit
d'investissement. Tant que le client n'a pas répondu, le service part d'un
profil présumé « Neutre » (et ne proposera le fonds que si les chiffres du
compte le justifient).

@dataclass(frozen=True)
class Question:
key: str
text: str
options: tuple[tuple[str, int], ...] # (réponse, points)

QUESTIONS: tuple[Question, ...] = (
Question("horizon", "Dans combien de temps pensez-vous avoir besoin de cet argent ?", (
("Moins d'1 an", 0), ("1 à 3 ans", 1), ("3 à 5 ans", 2), ("Plus de 5 ans", 3))),
Question("perte", "Votre placement perd 10 % en un mois. Que faites-vous ?", (
("Je retire tout", 0), ("Je retire une partie", 1), ("J'attends", 2), ("J'en profite pour investir plus", 3))),
Question("connaissance", "Avez-vous déjà investi dans des fonds, actions ou obligations ?", (
("Jamais", 0), ("Une ou deux fois", 1), ("Régulièrement", 2), ("Je m'y connais bien", 3))),
Question("objectif", "Quel est votre objectif principal ?", (
("Ne jamais perdre d'argent", 0), ("Battre l'inflation sans risque", 1),
("Faire croître mon épargne", 2), ("Maximiser le rendement", 3))),
Question("reserve", "En dehors de ce compte, disposez-vous d'une épargne de secours ?", (
("Non", 0), ("Moins de 3 mois de dépenses", 1), ("3 à 6 mois", 2), ("Plus de 6 mois", 3))),
)

def score_answers(answers: dict[str, int]) -> int:
"""answers = {clé_question: index_de_la_réponse}. Retourne un score 0-15."""
total = 0
for q in QUESTIONS:
idx = answers.get(q.key)
if idx is None:
raise ValueError(f"Question sans réponse : {q.text}")
if not 0 <= idx < len(q.options):
raise ValueError(f"Réponse invalide pour « {q.key} ».")
total += q.options[idx][1]
return total

def profile_from_answers(answers: dict[str, int]) -> RiskProfile:
score = score_answers(answers)
# Un horizon < 1 an ou « ne jamais perdre » plafonne le profil à Défensif.
if answers.get("horizon") == 0 or answers.get("objectif") == 0:
return RiskProfile.DEFENSIVE
if score <= 5:
return RiskProfile.DEFENSIVE
if score <= 10:
return RiskProfile.NEUTRAL
return RiskProfile.DYNAMIC

=============================================================================
7. PRODUITS & PORTEFEUILLE
=============================================================================
Les trois produits d'investissement et le portefeuille du client.
#
* Compte épargne : 2 %/an, retrait libre et immédiat, sans risque.
* Placement à terme : 3 %/an, bloqué 12 mois. Chaque versement mensuel ouvre
une « tranche » (échelle de placements) : une tranche arrive à échéance
chaque mois après la première année, et est reconduite (capital + intérêts nets). Rupture anticipée possible mais
coûteuse (perte des intérêts courus + pénalité).
* Fonds géré : argent confié à des gestionnaires, rendement espéré ~5 %/an
mais valeur fluctuante (perte possible). Retrait possible à tout moment,
mais : délai de règlement (J+3), taxe sur opération de bourse, frais si le
retrait n'a pas été annoncé, et vente en plusieurs jours au-delà d'un plafond.

ZERO = Decimal("0")
UNIT_Q = Decimal("0.000001")

@dataclass
class TermTranche:
principal: Decimal
start: pd.Period
maturity: pd.Period
rate: Decimal
def months_elapsed(self, month: pd.Period) -> int:
return max(0, min((month - self.start).n, (self.maturity - self.start).n))
def accrued_interest(self, month: pd.Period) -> Decimal:
return money(self.principal * self.rate * Decimal(self.months_elapsed(month)) / Decimal(12))

@dataclass
class WithdrawalStep:
product: ProductType
gross: Decimal # montant prélevé sur le produit
costs: Decimal # frais / taxes / pénalités
net: Decimal # montant réellement versé sur le compte courant
delay_days: int
note: str

@dataclass
class WithdrawalPlan:
requested: Decimal
notice_days: int
steps: list[WithdrawalStep] = field(default_factory=list)
warnings: list[str] = field(default_factory=list)
blocked: Decimal = ZERO # montant non disponible (placement à terme non rompu)
@property
def total_net(self) -> Decimal:
return sum((s.net for s in self.steps), ZERO)
@property
def total_costs(self) -> Decimal:
return sum((s.costs for s in self.steps), ZERO)
@property
def max_delay_days(self) -> int:
return max((s.delay_days for s in self.steps), default=0)
@property
def fully_covered(self) -> bool:
return self.total_net + CENT >= self.requested

class Portfolio:
"""Positions du client dans les trois produits."""
def __init__(self, cfg: ProductConfig | None = None, fund_nav: Decimal = Decimal("100")):
self.cfg = cfg or ProductConfig()
self.savings = ZERO
self.savings_accrued = ZERO # intérêts courus de l'année (versés fin décembre)
self.savings_interest_year = ZERO # intérêts bruts versés sur l'année (pour l'exonération)
self.term_tranches: list[TermTranche] = []
self.fund_units = ZERO
self.fund_nav = fund_nav
self.fund_cost_basis = ZERO
self.ledger: list[dict] = [] # journal des opérations (traçabilité)
# Cumul des revenus nets (intérêts, plus-values réalisées) et des coûts
self.income_earned = ZERO
self.costs_paid = ZERO
# ------------------------------------------------------------------ valeurs
@property
def fund_value(self) -> Decimal:
return money(self.fund_units * self.fund_nav)
def term_value(self, month: pd.Period | None = None) -> Decimal:
return sum((t.principal for t in self.term_tranches), ZERO)
def term_accrued(self, month: pd.Period) -> Decimal:
"""Intérêts courus, nets du précompte (ce que le client touchera réellement)."""
gross = sum((t.accrued_interest(month) for t in self.term_tranches), ZERO)
return money(gross * (Decimal(1) - self.cfg.withholding_tax))
def total_value(self, month: pd.Period | None = None) -> Decimal:
accrued = self.term_accrued(month) if month is not None else ZERO
return money(self.savings + self.savings_accrued + self.term_value() + accrued + self.fund_value)
def breakdown(self, month: pd.Period | None = None) -> dict[ProductType, Decimal]:
accrued = self.term_accrued(month) if month is not None else ZERO
return {
ProductType.SAVINGS: money(self.savings + self.savings_accrued),
ProductType.TERM: money(self.term_value() + accrued),
ProductType.FUND: self.fund_value,
}
def _log(self, month, kind, product, amount, detail=""):
self.ledger.append({"mois": str(month), "opération": kind,
"produit": product.label if isinstance(product, ProductType) else product,
"montant": money(amount), "détail": detail})
# ------------------------------------------------------------ versements
def invest(self, amount: Decimal, allocation: Allocation, month: pd.Period) -> dict[ProductType, Decimal]:
amount = money(amount)
if amount <= 0:
return {}
parts: dict[ProductType, Decimal] = {}
items = list(allocation.weights.items())
remaining = amount
for i, (product, w) in enumerate(items):
part = remaining if i == len(items) - 1 else money(amount * w)
remaining -= part
parts[product] = part
if part <= 0:
continue
if product == ProductType.SAVINGS:
self.savings += part
self._log(month, "Versement", product, part)
elif product == ProductType.TERM:
self.term_tranches.append(TermTranche(part, month, month + self.cfg.term_months, self.cfg.term_rate))
self._log(month, "Versement", product, part, f"échéance {month + self.cfg.term_months}")
else:
fee = money(part * self.cfg.fund_entry_fee_pct)
units = ((part - fee) / self.fund_nav).quantize(UNIT_Q, rounding=ROUND_DOWN)
self.fund_units += units
self.fund_cost_basis += part
self.costs_paid += fee
self._log(month, "Versement", product, part, f"frais d'entrée {fee} €, {units} parts à {self.fund_nav}")
return parts
# ------------------------------------------------------------ évolution
def step_month(self, month: pd.Period, fund_return: float) -> Decimal:
"""
Fait vivre le portefeuille pendant `month`. `fund_return` = rendement brut
mensuel du fonds (les frais de gestion sont déduits ici).
Renvoie le montant des tranches arrivées à échéance (reconduites ou reversées sur le livret).
"""
c = self.cfg
# Compte épargne : intérêts courus, versés en fin d'année
self.savings_accrued += money(self.savings * c.savings_rate / Decimal(12))
if month.month == 12 and self.savings_accrued > 0:
gross = self.savings_accrued
exempt_left = max(ZERO, c.savings_tax_exemption - self.savings_interest_year)
taxable = max(ZERO, gross - exempt_left)
tax = money(taxable * c.savings_withholding_tax)
self.savings += gross - tax
self.income_earned += gross - tax
self.costs_paid += tax
self._log(month, "Intérêts", ProductType.SAVINGS, gross - tax, f"brut {gross} €, précompte {tax} €")
self.savings_interest_year = ZERO
self.savings_accrued = ZERO
# Fonds : valeur liquidative
net_r = fund_return - float(c.fund_annual_fee_pct) / 12
self.fund_nav = (self.fund_nav * Decimal(str(round(1 + net_r, 8)))).quantize(Decimal("0.0001"))
# Placement à terme : tranches échues -> compte épargne
matured_total = ZERO
still = []
for t in self.term_tranches:
if month >= t.maturity:
interest = money(t.principal * t.rate * Decimal((t.maturity - t.start).n) / Decimal(12))
tax = money(interest * c.withholding_tax)
back = t.principal + interest - tax
self.income_earned += interest - tax
self.costs_paid += tax
matured_total += back
if c.term_auto_renew:
renewed = TermTranche(back, month, month + c.term_months, c.term_rate)
still.append(renewed)
self._log(month, "Échéance + reconduction", ProductType.TERM, back,
f"capital {t.principal} € + intérêts nets {interest - tax} € replacés "
f"jusqu'au {renewed.maturity.strftime('%m/%Y')}")
else:
self.savings += back
self._log(month, "Échéance", ProductType.TERM, back,
f"capital {t.principal} € + intérêts nets {interest - tax} € → compte épargne")
else:
still.append(t)
self.term_tranches = still
return matured_total
# ------------------------------------------------------------ retraits
def plan_withdrawal(self, amount: Decimal, month: pd.Period, notice_days: int = 0,
allow_term_break: bool = False) -> WithdrawalPlan:
"""Construit (sans l'exécuter) le plan de retrait le moins coûteux."""
c = self.cfg
amount = money(amount)
plan = WithdrawalPlan(requested=amount, notice_days=notice_days)
need = amount
# 1) Compte épargne : immédiat et gratuit
avail = money(self.savings + self.savings_accrued)
if need > 0 and avail > 0:
take = min(need, avail)
plan.steps.append(WithdrawalStep(ProductType.SAVINGS, take, ZERO, take, 0, "Disponible immédiatement, sans frais."))
need -= take
# 2) Fonds géré : J+3, TOB, frais si non annoncé, vente en plusieurs jours
if need > 0 and self.fund_value > 0:
cost_pct = c.fund_exit_tax_pct
unannounced = notice_days < c.fund_notice_days
if unannounced:
cost_pct += c.fund_unannounced_exit_fee_pct
gross_needed = money(need / (Decimal(1) - cost_pct))
gross = min(gross_needed, self.fund_value)
costs = money(gross * cost_pct)
net = gross - costs
days_to_sell = max(1, math.ceil(gross / c.fund_daily_liquidity_cap))
delay = c.fund_settlement_days + days_to_sell - 1
note = f"Vente des parts, argent disponible sous {delay} jour(s) ouvrable(s)."
if days_to_sell > 1:
note += f" Ordre exécuté en {days_to_sell} fois (plafond {eur(c.fund_daily_liquidity_cap)} €/jour)."
plan.steps.append(WithdrawalStep(ProductType.FUND, gross, costs, net, delay, note))
need -= net
# Moins-value ?
if self.fund_cost_basis > 0:
ratio = gross / self.fund_value
basis_sold = money(self.fund_cost_basis * ratio)
if gross < basis_sold:
plan.warnings.append(
f"⚠ Le fonds vaut actuellement moins que ce qui a été investi : ce retrait "
f"concrétise une perte d'environ {eur(basis_sold - gross, 2)} €.")
if unannounced:
plan.warnings.append(
f"Retrait non annoncé au moins {c.fund_notice_days} jours à l'avance : frais supplémentaires de "
f"{c.fund_unannounced_exit_fee_pct * 100:.1f} %. Annoncez vos grosses dépenses pour les éviter.")
if days_to_sell > 1:
plan.warnings.append(
"⚠ Montant important : l'ordre de vente ne pourra pas être exécuté en une seule fois. "
f"L'argent arrivera progressivement sur {delay} jours ouvrables et le prix de vente peut "
"varier entre-temps (vous êtes susceptible de perdre un peu d'argent).")
# 3) Placement à terme : bloqué, sauf rupture anticipée explicitement acceptée
if need > 0 and self.term_tranches:
tranches = sorted(self.term_tranches, key=lambda t: t.maturity)
locked_total = sum((t.principal for t in tranches), ZERO)
if allow_term_break:
gross = ZERO
lost_interest = ZERO
for t in tranches:
if gross >= need / (Decimal(1) - c.term_early_break_fee_pct):
break
gross += t.principal
lost_interest += t.accrued_interest(month)
gross = min(gross, money(need / (Decimal(1) - c.term_early_break_fee_pct)))
penalty = money(gross * c.term_early_break_fee_pct)
net = gross - penalty
plan.steps.append(WithdrawalStep(
ProductType.TERM, gross, penalty, net, 2,
f"Rupture anticipée : pénalité {penalty} € et perte des intérêts courus (~{lost_interest} €)."))
plan.warnings.append("⚠ Rupture anticipée du placement à terme : c'est l'option la plus coûteuse.")
need -= net
else:
plan.blocked = min(need, locked_total)
next_mat = tranches[0].maturity
plan.warnings.append(
f"{eur(plan.blocked, 2)} € restent bloqués sur le placement à terme "
f"(prochaine échéance : {next_mat.strftime('%m/%Y')}). Une rupture anticipée est possible "
"mais entraîne une pénalité et la perte des intérêts.")
if need > CENT and not plan.blocked:
plan.warnings.append(f"Montant demandé supérieur à l'épargne disponible : il manque {eur(need, 2)} €.")
return plan
def execute_withdrawal(self, plan: WithdrawalPlan, month: pd.Period) -> Decimal:
received = ZERO
for s in plan.steps:
if s.product == ProductType.SAVINGS:
from_accrued = min(self.savings_accrued, max(ZERO, s.gross - self.savings))
self.savings_accrued -= from_accrued
self.savings -= s.gross - from_accrued
elif s.product == ProductType.FUND:
units = (s.gross / self.fund_nav).quantize(UNIT_Q)
units = min(units, self.fund_units)
ratio = units / self.fund_units if self.fund_units else ZERO
basis = money(self.fund_cost_basis * ratio)
self.fund_units -= units
self.fund_cost_basis -= basis
self.income_earned += s.gross - basis
else:
remaining = s.gross
keep = []
for t in sorted(self.term_tranches, key=lambda t: t.maturity):
if remaining <= 0:
keep.append(t)
elif t.principal <= remaining:
remaining -= t.principal
else:
t.principal -= remaining
remaining = ZERO
keep.append(t)
self.term_tranches = keep
self.costs_paid += s.costs
received += s.net
self._log(month, "Retrait", s.product, s.net, f"brut {s.gross} €, frais {s.costs} €")
return money(received)

=============================================================================
8. MOTEUR DE DÉCISION
=============================================================================
Moteur de décision : combien investir à la fin du mois ?
#
montant = taux_d'investissement × ( solde
− matelas de sécurité
− trésorerie du mois (creux avant l'arrivée du salaire)
− marge de sécurité (incertitude de la prédiction)
− réserve pour déficits prévus (ex. vacances d'été)
− provision grosses dépenses annoncées )
#
* Marge de sécurité = max(k × σ_net, 10 % des dépenses prévues) : plus le compte
est imprévisible, plus on garde de réserve (k = 1,645 ≈ 95 % de confiance).
* Réserve pour déficits = pire cumul négatif des flux nets prévus (en scénario
prudent) sur les 3 prochains mois.
* Provision dépense annoncée = montant − ce que les mois d'ici là devraient
rapporter (scénario prudent). La provision grossit à l'approche de l'échéance.
* Sous le montant minimum (50 €), on n'investit rien : les frais mangeraient le gain.
* Si le solde passe sous ce qu'il faut garder, le moteur propose de rapatrier
de l'argent (compte épargne d'abord) : c'est le mouvement inverse.

ZERO = Decimal("0")

@dataclass
class Decision:
month: pd.Period
balance: Decimal
cushion: Decimal
cash_need: Decimal # creux de trésorerie avant la prochaine rentrée
safety_margin: Decimal
deficit_reserve: Decimal
planned_provision: Decimal
investable: Decimal # avant application du taux
investment_rate: Decimal
amount: Decimal # montant finalement investi
repatriation: Decimal = ZERO # montant à ramener sur le compte courant
explanation: list[str] = field(default_factory=list)
@property
def components(self) -> list[tuple[str, Decimal]]:
"""Décomposition pour le graphique « cascade »."""
return [
("Solde fin de mois", self.balance),
("Matelas de sécurité", -self.cushion),
("Trésorerie du mois", -self.cash_need),
("Marge d'incertitude", -self.safety_margin),
("Réserve déficits prévus", -self.deficit_reserve),
("Dépenses annoncées", -self.planned_provision),
]

def _floor10(x: Decimal) -> Decimal:
"""Arrondi à la dizaine inférieure : montants lisibles pour le client."""
return (x / 10).to_integral_value(rounding=ROUND_DOWN) * 10

def decide(month: pd.Period, balance: Decimal, forecast_table: pd.DataFrame, net_sigma: float,
settings: ClientSettings, cfg: DecisionConfig | None = None,
intramonth_need: float = 0.0) -> Decision:
"""
month : mois qui se termine (la décision est prise le dernier jour).
balance : solde du compte courant ce jour-là.
forecast_table : prévisions des mois suivants (colonnes income, expenses, net, net_low),
indexées par pd.Period, en commençant par month + 1.
"""
cfg = cfg or DecisionConfig()
balance = money(balance)
fut = forecast_table.loc[forecast_table.index > month]
expl: list[str] = []
cash_need = money(intramonth_need)
if cash_need > 0:
expl.append(f"Trésorerie du mois : {eur(cash_need)} € (dépenses qui tombent avant les rentrées, "
"ex. loyer le 1er et salaire le 25).")
next_exp = float(fut["expenses"].iloc[0]) if len(fut) else 0.0
safety = money(max(cfg.safety_k * net_sigma, float(cfg.extra_margin_pct) * next_exp))
expl.append(f"Marge d'incertitude : {eur(safety)} € (écart possible entre prévu et réel, 95 %).")
# Réserve pour déficits : pire creux cumulé sur 3 mois en scénario prudent.
cum, worst = 0.0, 0.0
for v in fut["net_low"].iloc[:3]:
cum += float(v)
worst = min(worst, cum)
deficit = money(-worst)
if deficit > 0:
months_neg = [p.strftime("%m/%Y") for p, v in fut["net"].iloc[:3].items() if v < 0]
expl.append(f"Réserve pour déficits prévus : {eur(deficit)} €"
(f" (mois déficitaires attendus : {', '.join(months_neg)})." if months_neg else "."))
# Grosses dépenses annoncées par le client.
provision = ZERO
for pe in settings.planned_expenses:
due = pd.Period(pe.due, freq="M")
if due <= month:
continue
if (due - month).n > cfg.planned_expense_lookahead_months:
continue
before = fut.loc[fut.index < due, "net_low"].clip(lower=0).sum()
need = money(max(Decimal("0"), pe.amount - money(before)))
if need > 0:
provision += need
expl.append(f"Provision pour « {pe.label} » ({eur(pe.amount)} € en {due.strftime('%m/%Y')}) : "
f"{eur(need)} € gardés dès maintenant.")
investable = money(balance - settings.cushion - cash_need - safety - deficit - provision)
rate = settings.investment_rate
raw = money(max(ZERO, investable) * rate)
amount = min(_floor10(raw), cfg.max_investment)
repatriation = ZERO
if investable <= 0:
amount = ZERO
# Si le solde passe sous le matelas + provisions : on ramène de l'argent.
shortfall = money(settings.cushion + cash_need + provision + deficit - balance)
if shortfall > 0:
repatriation = shortfall
expl.append(f"Solde insuffisant : {eur(shortfall)} € à rapatrier depuis l'épargne pour "
"reconstituer le matelas et les provisions.")
else:
expl.append("Pas de surplus ce mois-ci : rien n'est investi.")
elif amount < cfg.min_investment:
expl.append(f"Surplus de {eur(raw, 2)} € inférieur au minimum de {eur(cfg.min_investment)} € : "
"rien n'est investi ce mois-ci (les frais dépasseraient le gain).")
amount = ZERO
else:
if raw > cfg.max_investment:
expl.append(f"Montant plafonné à {eur(cfg.max_investment)} € par mois ; le reste sera investi "
"les mois suivants.")
expl.insert(0, f"Investissement proposé : {eur(amount)} € ({rate * 100:.0f} % du surplus de "
f"{eur(investable)} €).")
return Decision(month=month, balance=balance, cushion=settings.cushion, cash_need=cash_need,
safety_margin=safety,
deficit_reserve=deficit, planned_provision=provision, investable=investable,
investment_rate=rate, amount=amount, repatriation=repatriation, explanation=expl)

=============================================================================
9. PROPOSITION PAR DÉFAUT
=============================================================================
Proposition par défaut faite au client.
#
À partir des chiffres du compte (et du profil de risque s'il est connu), le
service propose d'office :
* un matelas de sécurité,
* un taux d'investissement (part du surplus réellement investie),
* UN des trois produits (celui qui convient le mieux),
avec l'explication de chaque choix. Le client peut ensuite tout modifier :
matelas, taux, et répartition libre entre 1, 2 ou 3 produits.

@dataclass
class Proposal:
settings: ClientSettings
product: ProductType
product_scores: dict[ProductType, float]
reasons: dict[str, list[str]] = field(default_factory=dict)
risk_profile_assumed: bool = True

def default_cushion(analysis: AccountAnalysis, cfg: AppConfig = DEFAULT_CONFIG) -> Decimal:
"""Matelas = 1,5 mois de dépenses moyennes (min 1 000 €), arrondi à la centaine."""
raw = Decimal(str(analysis.mean_expenses)) * cfg.decision.default_cushion_months
raw = max(raw, Decimal("1000"))
return (raw / 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * 100

def default_investment_rate(stability: int) -> Decimal:
if stability >= 80:
return Decimal("1.00")
if stability >= 65:
return Decimal("0.90")
if stability >= 50:
return Decimal("0.75")
return Decimal("0.60")

def score_products(a: AccountAnalysis, profile: RiskProfile, has_planned_expenses: bool) -> dict[ProductType, float]:
stab = a.stability_score
savings = 40 + (100 - stab) * 0.6 + a.exceptional_ratio * 100
term = stab * 0.5 + a.income_regularity * 30
fund = stab * 0.5 + (10 if a.savings_rate > 0.20 else 0)
# Argent qui « dort » : solde bien supérieur à 3 mois de dépenses -> placement à terme intéressant
if float(a.current_balance) > 3 * a.mean_expenses:
term += 15
if has_planned_expenses:
term -= 30 # l'argent risque d'être nécessaire avant l'échéance
savings += 10
if profile == RiskProfile.DEFENSIVE:
savings += 25
fund -= 30
elif profile == RiskProfile.NEUTRAL:
fund += 15
else:
fund += 30
term -= 10
return {ProductType.SAVINGS: round(savings, 1), ProductType.TERM: round(term, 1),
ProductType.FUND: round(fund, 1)}

def build_proposal(a: AccountAnalysis, profile: RiskProfile | None = None, planned_expenses=None,
cfg: AppConfig = DEFAULT_CONFIG) -> Proposal:
assumed = profile is None
profile = profile or RiskProfile.NEUTRAL
planned_expenses = list(planned_expenses or [])
cushion = a.client.cushion or default_cushion(a, cfg)
rate = default_investment_rate(a.stability_score)
scores = score_products(a, profile, bool(planned_expenses))
product = max(scores, key=scores.get)
reasons: dict[str, list[str]] = {"matelas": [], "taux": [], "produit": []}
if a.client.cushion:
reasons["matelas"].append(f"Matelas choisi par le client : {eur(cushion)} €.")
else:
reasons["matelas"].append(
f"{str(cfg.decision.default_cushion_months).replace('.', ',')} mois de dépenses moyennes ({eur(a.mean_expenses)} €/mois), "
f"arrondi : {eur(cushion)} € restent toujours sur le compte courant.")
reasons["taux"].append(
f"Score de stabilité {a.stability_score}/100 : {rate * 100:.0f} % du surplus calculé est investi, "
f"le reste reste disponible sur le compte courant.")
labels = {
ProductType.SAVINGS: "vos dépenses varient ou des imprévus sont fréquents : la disponibilité "
"immédiate prime sur le rendement.",
ProductType.TERM: "revenus réguliers et argent qui dort sur le compte : un placement à terme "
"(une tranche de 12 mois par versement) offre le meilleur taux sans risque.",
ProductType.FUND: "compte stable et profil compatible avec un peu de risque : le fonds géré "
"offre le meilleur rendement espéré à long terme.",
}
reasons["produit"].append(f"{product.label} proposé : {labels[product]}")
if assumed:
reasons["produit"].append("Profil de risque présumé « Neutre » : remplissez le questionnaire "
"pour affiner la proposition.")
settings = ClientSettings(cushion=money(cushion), investment_rate=rate, allocation=Allocation.single(product),
risk_profile=profile, planned_expenses=planned_expenses)
return Proposal(settings=settings, product=product, product_scores=scores, reasons=reasons,
risk_profile_assumed=assumed)

def check_allocation_vs_profile(allocation: Allocation, profile: RiskProfile) -> list[str]:
"""Avertissements (non bloquants) si le choix du client dépasse son profil."""
warnings = []
fund_w = allocation.weights.get(ProductType.FUND, Decimal("0"))
if profile == RiskProfile.DEFENSIVE and fund_w > 0:
warnings.append("Votre profil est défensif : le fonds géré comporte un risque de perte. "
"Vous pouvez le choisir, mais en connaissance de cause.")
if profile == RiskProfile.NEUTRAL and fund_w > Decimal("0.7"):
warnings.append("Plus de 70 % en fonds géré dépasse ce qui est habituel pour un profil neutre.")
if allocation.weights.get(ProductType.TERM, Decimal("0")) == Decimal("1"):
warnings.append("Tout en placement à terme : aucun argent investi ne sera disponible avant 12 mois "
"(hors rupture anticipée coûteuse). Pensez à annoncer vos grosses dépenses.")
return warnings

=============================================================================
10. GÉNÉRATEUR DE CLIENTS DE DÉMO
=============================================================================
Générateur de clients fictifs réalistes (Belgique) pour démonstration et tests.
#
Il sert en attendant votre « utilisateur par défaut » : il reproduit ce qu'on
trouve sur un vrai compte courant belge — salaire, loyer, courses, abonnements,
prime de fin d'année (décembre), pécule de vacances (mai-juin), vacances d'été,
cadeaux de décembre, rentrée scolaire, et quelques grosses dépenses imprévues.

@dataclass(frozen=True)
class Profile:
key: str
name: str
description: str
months: int
salary: float
salary_noise: float # variabilité des rentrées (indépendant)
rent: float
grocery_week: float
leisure_month: float
leisure_noise: float
kids: int = 0
second_salary: float = 0.0
shock_prob: float = 0.04 # probabilité mensuelle d'une grosse dépense imprévue
irregular: bool = False # revenus par factures irrégulières (indépendant)
bonuses: bool = True # prime de fin d'année + pécule de vacances (salarié)
opening_balance: float = 3000.0

PROFILES: dict[str, Profile] = {
"stable": Profile("stable", "Sophie Martens", "Employée, revenus et dépenses réguliers", 30,
salary=2850, salary_noise=0.0, rent=920, grocery_week=85, leisure_month=260,
leisure_noise=0.20, opening_balance=4200),
"famille": Profile("famille", "Famille Peeters", "Couple avec 2 enfants, deux salaires", 36,
salary=3100, second_salary=2400, salary_noise=0.0, rent=1350, grocery_week=190,
leisure_month=420, leisure_noise=0.25, kids=2, opening_balance=6500),
"independant": Profile("independant", "Lucas Janssens", "Indépendant, revenus très irréguliers", 30,
salary=3600, salary_noise=0.65, rent=1050, grocery_week=95, leisure_month=500,
leisure_noise=0.9, shock_prob=0.12, opening_balance=5000,
irregular=True),
"etudiant": Profile("etudiant", "Emma Dubois", "Étudiante jobiste, pas de capacité d'épargne", 30,
salary=1600, salary_noise=0.12, rent=480, grocery_week=55, leisure_month=160,
leisure_noise=0.35, opening_balance=3500, bonuses=False),
"jeune": Profile("jeune", "Noah Lambert", "Compte ouvert il y a 14 mois", 14,
salary=2500, salary_noise=0.0, rent=850, grocery_week=80, leisure_month=250,
leisure_noise=0.2, opening_balance=1500),
}

def _month_starts(end: date, months: int) -> list[date]:
first = date(end.year, end.month, 1)
out = []
y, m = first.year, first.month
for _ in range(months):
out.append(date(y, m, 1))
m -= 1
if m == 0:
y, m = y - 1, 12
return list(reversed(out))

def generate_transactions(profile: Profile, end: date = date(2026, 9, 30), seed: int = 7) -> pd.DataFrame:
rng = np.random.default_rng(seed)
rows: list[tuple[date, float, str, str]] = []
def add(d: date, amount: float, desc: str, cat: str):
rows.append((d, round(float(amount), 2), desc, cat))
months = _month_starts(end, profile.months)
for i, m0 in enumerate(months):
year_idx = i // 12
dim = (pd.Timestamp(m0) + pd.offsets.MonthEnd(0)).day
def day(d: int) -> date:
return date(m0.year, m0.month, min(d, dim))
# --- Rentrées ---
indexation = (1.02 ** year_idx) # indexation automatique des salaires (BE)
sal = profile.salary * indexation
if profile.irregular:
# indépendant : factures payées de façon irrégulière, parfois 2 fois, parfois 0
n_pay = rng.choice([0, 1, 1, 2], p=[0.15, 0.45, 0.25, 0.15])
for _ in range(n_pay):
add(day(int(rng.integers(3, 28))), max(0, rng.normal(sal * 0.75, sal * profile.salary_noise * 0.6)),
"Paiement facture client", "Revenus")
else:
noise = rng.normal(1, profile.salary_noise) if profile.salary_noise else 1.0
add(day(25), sal * max(0.3, noise), "Salaire - Employeur SA", "Revenus")
if profile.second_salary:
add(day(28), profile.second_salary * indexation, "Salaire conjoint", "Revenus")
if profile.bonuses and not profile.irregular and m0.month == 12:
add(day(20), sal * 0.92, "Prime de fin d'année", "Revenus")
if profile.bonuses and not profile.irregular and m0.month == 6:
add(day(10), sal * 0.92, "Pécule de vacances", "Revenus")
if profile.kids:
add(day(10), 185 * profile.kids, "Allocations familiales (Famiris)", "Revenus")
# --- Charges fixes ---
add(day(1), -profile.rent * indexation, "Loyer", "Logement")
add(day(5), -rng.normal(165 + 40 * profile.kids, 12), "Énergie (électricité/gaz)", "Logement")
add(day(8), -(45 + 20 * profile.kids), "Télécom / Internet", "Abonnements")
add(day(12), -17.99, "Streaming", "Abonnements")
add(day(15), -rng.normal(95, 5) * (1 + 0.5 * profile.kids), "Mutuelle / assurances", "Assurances")
if m0.month == 3:
add(day(20), -rng.normal(420, 40), "Assurance auto (annuelle)", "Assurances")
# --- Courses (hebdomadaires) ---
for d in range(2, dim + 1, 7):
winter = 1.08 if m0.month in (11, 12, 1) else 1.0
add(day(d), -abs(rng.normal(profile.grocery_week * winter, profile.grocery_week * 0.18)),
"Supermarché", "Alimentation")
# --- Transport / carburant ---
for d in (7, 21):
add(day(d), -abs(rng.normal(58, 10)), "Carburant", "Transport")
# --- Loisirs / restaurants ---
n_leisure = int(rng.integers(3, 7))
leisure_total = max(20, rng.normal(profile.leisure_month, profile.leisure_month * profile.leisure_noise))
for w in rng.dirichlet(np.ones(n_leisure)):
add(day(int(rng.integers(1, dim + 1))), -leisure_total * w, "Restaurant / loisirs", "Loisirs")
# --- Saisonnalité ---
if m0.month == 12:
add(day(15), -abs(rng.normal(520 + 150 * profile.kids, 60)), "Cadeaux de fin d'année", "Shopping")
if m0.month in (7, 8):
add(day(int(rng.integers(1, 20))), -abs(rng.normal(900 + 450 * profile.kids, 120)) / (
2 if profile.key == "etudiant" else 1), "Vacances d'été", "Voyages")
if m0.month == 9 and profile.kids:
add(day(3), -abs(rng.normal(260 * profile.kids, 30)), "Rentrée scolaire", "Enfants")
if m0.month in (1, 7):
add(day(12), -abs(rng.normal(230, 50)), "Soldes", "Shopping")
# --- Chocs imprévus ---
if rng.random() < profile.shock_prob:
add(day(int(rng.integers(1, dim + 1))), -abs(rng.normal(1500, 400)),
rng.choice(["Réparation voiture", "Électroménager", "Dentiste", "Garagiste"]), "Imprévus")
df = pd.DataFrame(rows, columns=["date", "montant", "libelle", "categorie"])
df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(end)]
df = df[df["montant"] != 0].sort_values("date").reset_index(drop=True)
return df

def generate_client(key: str = "stable", end: date = date(2026, 9, 30), seed: int = 7) -> ClientData:
profile = PROFILES[key]
tx = generate_transactions(profile, end=end, seed=seed)
first_day = pd.to_datetime(tx["date"]).min().date() - timedelta(days=0)
return build_client(tx, {"nom": profile.name, "solde_initial": profile.opening_balance,
"date_ouverture": first_day.replace(day=1).isoformat()})

def export_demo_files(folder: str) -> None:
"""Écrit un CSV + un JSON par profil (exemples du format attendu)."""
import json
from pathlib import Path
out = Path(folder)
out.mkdir(parents=True, exist_ok=True)
for key, profile in PROFILES.items():
tx = generate_transactions(profile)
tx.to_csv(out / f"client_{key}.csv", index=False, sep=";", decimal=",")
first = pd.to_datetime(tx["date"]).min().date().replace(day=1)
payload = {
"client": {"nom": profile.name, "solde_initial": profile.opening_balance,
"date_ouverture": first.isoformat()},
"transactions": [
{"date": str(r.date), "montant": r.montant, "libelle": r.libelle, "categorie": r.categorie}
for r in tx.itertuples()
],
}
(out / f"client_{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

=============================================================================
11. SIMULATIONS
=============================================================================
Simulations du service.
#
1. `current_decision` : la décision de ce mois-ci (ce que le client verrait aujourd'hui).
2. `replay_history` : « et si le service avait tourné sur votre compte ? » — rejoue les
derniers mois réels, en ne regardant chaque mois QUE le passé
(pas de triche), et vérifie que le compte n'est jamais passé sous
le matelas au jour le jour.
3. `monte_carlo` : projection sur N mois avec des centaines de scénarios aléatoires
(rentrées/dépenses incertaines, imprévus, marchés) ; compare avec
le fait de tout laisser dormir sur le compte courant.

ZERO = Decimal("0")

def month_end_balance(client: ClientData, month: pd.Period) -> Decimal:
tx = client.transactions
upto = tx.loc[tx["date"] <= month.to_timestamp(how="end"), "amount"].sum()
return money(client.opening_balance + money(upto))

def current_decision(analysis: AccountAnalysis, settings: ClientSettings,
cfg: AppConfig = DEFAULT_CONFIG) -> Decision:
if analysis.forecast is None:
raise ValueError("Pas de prévision disponible (historique trop court).")
month = analysis.last_month
table = analysis.forecast.table(cfg.decision.safety_k)
return decide(month, month_end_balance(analysis.client, month), table,
analysis.forecast.net_sigma(1), settings, cfg.decision, analysis.intramonth_need)

---------------------------------------------------------------------------
Rejeu sur l'historique réel
---------------------------------------------------------------------------
@dataclass
class ReplayResult:
monthly: pd.DataFrame # une ligne par mois rejoué
daily_cash: pd.Series # solde courant quotidien AVEC le service
daily_baseline: pd.Series # solde courant quotidien SANS le service
portfolio: Portfolio
decisions: list[Decision]

def replay_history(client: ClientData, settings: ClientSettings, cfg: AppConfig = DEFAULT_CONFIG,
min_months: int | None = None, seed: int = 1) -> ReplayResult:
rng = np.random.default_rng(seed)
min_months = min_months or cfg.eligibility.min_history_months
fast_cfg = replace(cfg, forecast=replace(cfg.forecast, backtest_months=3))
tx = client.transactions
months = pd.period_range(tx["date"].min().to_period("M"), tx["date"].max().to_period("M"), freq="M")
# dernier mois complet uniquement
last_full = analyze_account(client, cfg, with_forecast=False).last_month
months = [m for m in months if m <= last_full]
if len(months) <= min_months:
raise ValueError("Historique trop court pour rejouer le service.")
portfolio = Portfolio(cfg.products)
transfers: dict[pd.Timestamp, float] = {} # sorties (+) / rentrées (-) du compte courant
rows, decisions = [], []
mu = float(cfg.products.fund_expected_return) / 12
vol = cfg.products.fund_volatility / np.sqrt(12)
for m in months[min_months - 1:]:
sub = tx[tx["date"] <= m.to_timestamp(how="end")]
sub_client = ClientData(transactions=sub, name=client.name, opening_balance=client.opening_balance,
opening_date=client.opening_date, cushion=client.cushion)
a = analyze_account(sub_client, fast_cfg, with_forecast=True)
real_balance = month_end_balance(client, m)
cash = money(real_balance - money(sum(transfers.values())))
d = decide(m, cash, a.forecast.table(cfg.decision.safety_k), a.forecast.net_sigma(1),
settings, cfg.decision, a.intramonth_need)
day = m.to_timestamp(how="end").normalize()
moved = ZERO
if d.amount > 0:
portfolio.invest(d.amount, settings.allocation, m)
moved = d.amount
elif d.repatriation > 0:
plan = portfolio.plan_withdrawal(d.repatriation, m, notice_days=30)
moved = -portfolio.execute_withdrawal(plan, m)
if moved:
transfers[day] = transfers.get(day, 0.0) + float(moved)
decisions.append(d)
rows.append({"month": m, "solde_réel": float(real_balance), "cash_avant": float(cash),
"investi": float(d.amount), "rapatrié": float(d.repatriation),
"cash_après": float(cash - moved), "statut": a.status.label})
if m != months[-1]:
portfolio.step_month(m + 1, float(rng.normal(mu, vol)))
if rows:
rows[-1]["portefeuille"] = float(portfolio.total_value(m + 1))
daily_baseline = client.balance_series()
idx = pd.date_range(daily_baseline.index.min(), daily_baseline.index.max(), freq="D")
daily_baseline = daily_baseline.reindex(idx).ffill()
t = pd.Series(transfers, dtype=float).reindex(idx, fill_value=0.0).cumsum()
daily_cash = daily_baseline - t
return ReplayResult(pd.DataFrame(rows).set_index("month"), daily_cash, daily_baseline, portfolio, decisions)

---------------------------------------------------------------------------
Projection Monte Carlo
---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
months: pd.PeriodIndex
wealth: np.ndarray # (paths, months) patrimoine total avec le service
baseline: np.ndarray # (paths, months) sans le service (tout sur le compte courant)
cash: np.ndarray # (paths, months) solde du compte courant avec le service
invested: np.ndarray # (paths, months) valeur des placements
by_product: dict[ProductType, np.ndarray]
monthly_invest: np.ndarray
below_cushion: np.ndarray # (paths,) nombre de mois où le solde passe sous le matelas (même en cours de mois)
overdraft: np.ndarray # (paths,) nombre de mois avec passage à découvert
emergency_costs: np.ndarray # (paths,) coûts des retraits d'urgence
cushion: float
def percentiles(self, arr: np.ndarray, q=(5, 25, 50, 75, 95)) -> pd.DataFrame:
return pd.DataFrame(np.percentile(arr, q, axis=0).T, index=self.months,
columns=[f"p{x}" for x in q])
def summary(self) -> dict:
gain = self.wealth[:, -1] - self.baseline[:, -1]
return {
"gain_median": float(np.median(gain)),
"gain_p5": float(np.percentile(gain, 5)),
"gain_p95": float(np.percentile(gain, 95)),
"proba_gain_positif": float(np.mean(gain > 0)),
"proba_sous_matelas": float(np.mean(self.below_cushion > 0)),
"proba_decouvert": float(np.mean(self.overdraft > 0)),
"investi_median_par_mois": float(np.median(self.monthly_invest.mean(axis=1))),
"valeur_placements_mediane": float(np.median(self.invested[:, -1])),
"cout_urgence_moyen": float(np.mean(self.emergency_costs)),
}

def monte_carlo(analysis: AccountAnalysis, settings: ClientSettings, months: int = 24, n_paths: int = 200,
cfg: AppConfig = DEFAULT_CONFIG, seed: int = 42) -> MonteCarloResult:
rng = np.random.default_rng(seed)
a = analysis
fc = forecast_cashflows(a.monthly["income"], a.monthly["expenses_clean"], a.shock_provision,
ForecastConfig(horizon_months=months + 3, backtest_months=cfg.forecast.backtest_months))
table = fc.table(cfg.decision.safety_k)
sigma_net = fc.net_sigma(1)
start = a.last_month
idx = pd.period_range(start + 1, periods=months, freq="M")
# Imprévus : fréquence et taille observées
exc = a.anomalies[a.anomalies["exceptional"]]
p_shock = float(len(exc) / max(1, len(a.anomalies)))
shock_size = float(exc["excess"].mean()) if len(exc) else 0.0
need = a.intramonth_need
dd_samples = a.drawdowns.iloc[-12:].to_numpy() if len(a.drawdowns) else np.zeros(1)
planned = {pd.Period(pe.due, freq="M"): float(pe.amount) for pe in settings.planned_expenses}
mu = float(cfg.products.fund_expected_return) / 12
vol = cfg.products.fund_volatility / np.sqrt(12)
cushion = float(settings.cushion)
start_cash = float(month_end_balance(a.client, start))
shape = (n_paths, months)
wealth, baseline, cash_arr, inv_arr, minv = (np.zeros(shape) for _ in range(5))
prod = {p: np.zeros(shape) for p in ProductType}
below, overd, emerg = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)
for i in range(n_paths):
pf = Portfolio(cfg.products)
cash = Decimal(str(round(start_cash, 2)))
base = start_cash
d = decide(start, cash, table, sigma_net, settings, cfg.decision, need)
if d.amount > 0:
pf.invest(d.amount, settings.allocation, start)
cash -= d.amount
for h, m in enumerate(idx):
pf.step_month(m, float(rng.normal(mu, vol)))
row = table.loc[m]
inc = max(0.0, rng.normal(row["income"], fc.income.sigma_at(h + 1)))
exp = max(0.0, rng.normal(row["expenses"] - fc.shock_provision, fc.expenses.sigma_at(h + 1)))
if rng.random() < p_shock:
exp += abs(rng.normal(shock_size, shock_size * 0.3))
exp += planned.get(m, 0.0)
flow = inc - exp
# Creux intra-mois (timing loyer / salaire) tiré de l'historique réel
intramonth_low = float(cash) - float(rng.choice(dd_samples)) - planned.get(m, 0.0)
if intramonth_low < cushion:
below[i] += 1
if intramonth_low < 0:
overd[i] += 1
cash += Decimal(str(round(flow, 2)))
base += flow
# Découvert : retrait d'urgence non annoncé sur les placements
if cash < 0:
plan = pf.plan_withdrawal(-cash + Decimal(str(cushion)), m, notice_days=0)
emerg[i] += float(plan.total_costs)
cash += pf.execute_withdrawal(plan, m)
d = decide(m, cash, table, sigma_net, settings, cfg.decision, need)
if d.amount > 0:
pf.invest(d.amount, settings.allocation, m)
cash -= d.amount
minv[i, h] = float(d.amount)
elif d.repatriation > 0:
plan = pf.plan_withdrawal(d.repatriation, m, notice_days=30)
cash += pf.execute_withdrawal(plan, m)
value = float(pf.total_value(m))
wealth[i, h] = float(cash) + value
baseline[i, h] = base
cash_arr[i, h] = float(cash)
inv_arr[i, h] = value
for p, v in pf.breakdown(m).items():
prod[p][i, h] = float(v)
return MonteCarloResult(idx, wealth, baseline, cash_arr, inv_arr, prod, minv, below, overd, emerg, cushion)

=============================================================================
12. INTERFACE STREAMLIT
=============================================================================
def run_app():
CFG = DEFAULT_CONFIG
# Palette (catégorielle, ordre fixe, validée daltonisme) — cf. README
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
PRODUCT_COLORS = {ProductType.SAVINGS: C_BLUE, ProductType.TERM: C_ORANGE, ProductType.FUND: C_AQUA}
st.set_page_config(page_title="Smart Invest", page_icon="💶", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
.si-banner {border-radius: 10px; padding: 14px 18px; margin: 4px 0 14px 0; border: 1px solid rgba(128,128,128,.25);}
.si-muted {opacity: .75; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

def euro(x, d=0) -> str:
return f"{eur(x, d)} €"

def style_fig(fig: go.Figure, height: int = 360, ytitle: str = "€") -> go.Figure:
fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
yaxis=dict(title=ytitle, gridcolor="rgba(128,128,128,.15)", zeroline=False),
xaxis=dict(gridcolor="rgba(0,0,0,0)"), separators=", ")
return fig

# ---------------------------------------------------------------------------
# Chargement (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Analyse du compte et entraînement des modèles…")
def get_analysis(source_key: str, payload: bytes | None, filename: str | None):
if payload is None:
client = generate_client(source_key)
else:
client = load_client(payload, filename)
return analyze_account(client, CFG)

def settings_key(s: ClientSettings) -> tuple:
return (str(s.cushion), str(s.investment_rate),
tuple(sorted((p.value, str(w)) for p, w in s.allocation.weights.items())),
tuple((pe.label, str(pe.amount), pe.due.isoformat()) for pe in s.planned_expenses))

@st.cache_resource(show_spinner="Rejeu du service sur votre historique réel…")
def get_replay(source_key: str, _analysis, skey: tuple, _settings):
return replay_history(_analysis.client, _settings, CFG)

@st.cache_resource(show_spinner="Simulation de centaines de scénarios…")
def get_mc(source_key: str, _analysis, skey: tuple, _settings, months: int, paths: int):
return monte_carlo(_analysis, _settings, months=months, n_paths=paths, cfg=CFG)

# ---------------------------------------------------------------------------
# Barre latérale : source des données
# ---------------------------------------------------------------------------
with st.sidebar:
st.markdown("## 💶 Smart Invest")
st.caption("Analyse le compte courant, prédit vos besoins et investit chaque fin de mois "
"l'argent qui dort au-delà de votre matelas de sécurité.")
st.divider()
source = st.radio("Données du client", ["Client de démonstration", "Importer un fichier"], index=0)
payload, filename = None, None
if source == "Client de démonstration":
key = st.selectbox("Profil", list(PROFILES), format_func=lambda k: f"{PROFILES[k].name} — {PROFILES[k].description}")
source_key = f"demo:{key}"
demo_key = key
else:
up = st.file_uploader("Historique de transactions (CSV ou JSON)", type=["csv", "json", "txt"])
st.caption("Colonnes : date, montant (négatif = dépense), libellé, catégorie. "
"Voir README pour le format complet.")
if up is None:
st.info("Importez un fichier pour commencer.")
st.stop()
payload, filename = up.getvalue(), up.name
source_key = "file:" + hashlib.sha256(payload).hexdigest()[:16]
demo_key = None
try:
analysis = get_analysis(demo_key or source_key, payload, filename)
except DataValidationError as exc:
st.error(f"Fichier invalide : {exc}")
st.stop()
client = analysis.client
# ---------------------------------------------------------------------------
# État de session : réglages proposés par défaut puis modifiables
# ---------------------------------------------------------------------------
ss = st.session_state
if ss.get("source_key") != source_key:
ss.source_key = source_key
ss.profile = None # profil présumé tant que le questionnaire n'est pas rempli
ss.planned = [] # dépenses annoncées
ss.custom = None # réglages modifiés par le client (None = proposition)
ss.pop("mc_done", None)
proposal = build_proposal(analysis, ss.profile, ss.planned, CFG)
settings: ClientSettings = ss.custom or proposal.settings
settings.planned_expenses = list(ss.planned)
settings.risk_profile = ss.profile or RiskProfile.NEUTRAL
with st.sidebar:
st.divider()
st.markdown(f"{client.name}")
st.markdown(f"Solde actuel : {euro(client.current_balance)}")
st.markdown(f"Historique : {analysis.history_months} mois")
st.markdown(f"Statut : {analysis.status.label}")
if ss.custom is not None:
st.markdown("Réglages : ✏️ personnalisés")
else:
st.markdown("Réglages : ⭐ proposition par défaut")
for w in client.warnings:
st.warning(w)
# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.title(f"Smart Invest · {client.name}")
status_style = {
EligibilityStatus.ELIGIBLE: ("✅", "rgba(27,175,122,.10)"),
EligibilityStatus.PENDING: ("⏳", "rgba(237,161,0,.12)"),
EligibilityStatus.REFUSED: ("⛔", "rgba(227,73,72,.10)"),
EligibilityStatus.TOO_YOUNG: ("🕒", "rgba(128,128,128,.10)"),
}
icon, bg = status_style[analysis.status]
st.markdown(f"<div class='si-banner' style='background:{bg}'><b>{icon} {analysis.status.label}</b><br>"
"<br>".join(analysis.reasons) + "</div>", unsafe_allow_html=True)
k = st.columns(5)
k[0].metric("Solde actuel", euro(client.current_balance))
k[1].metric("Rentrées / mois", euro(analysis.mean_income))
k[2].metric("Dépenses / mois", euro(analysis.mean_expenses))
k[3].metric("Capacité d'épargne / mois", euro(analysis.mean_net), f"{analysis.savings_rate:.0%} des rentrées",
delta_color="off", delta_arrow="off")
k[4].metric("Score de stabilité", f"{analysis.stability_score}/100")
tabs = st.tabs(["📊 Analyse", "🔮 Prédiction IA", "💡 Proposition & réglages", "📅 Dépenses annoncées",
"🧪 Simulation", "💸 Retrait"])
# ===========================================================================
# 1. Analyse
# ===========================================================================
with tabs[0]:
m = analysis.monthly
x = m.index.to_timestamp()
c1, c2 = st.columns([3, 2])
with c1:
st.subheader("Rentrées et dépenses mensuelles")
fig = go.Figure()
fig.add_bar(x=x, y=m["income"], name="Rentrées", marker_color=C_BLUE,
hovertemplate="%{y:,.0f} €")
fig.add_bar(x=x, y=m["expenses"], name="Dépenses", marker_color=C_ORANGE,
hovertemplate="%{y:,.0f} €")
exc = analysis.anomalies[analysis.anomalies["exceptional"]]
if len(exc):
fig.add_scatter(x=exc.index.to_timestamp(), y=exc["expenses"] * 1.04, mode="markers+text",
text=["imprévu"] * len(exc), textposition="top center", name="Mois exceptionnel",
marker=dict(symbol="triangle-down", size=10, color=C_GRAY), hoverinfo="skip")
fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
st.plotly_chart(style_fig(fig), use_container_width=True)
with c2:
st.subheader("Dépenses par catégorie")
cat = category_breakdown(client.transactions).head(10)
fig = go.Figure(go.Bar(x=cat["moyenne_mensuelle"], y=cat.index, orientation="h", marker_color=C_BLUE,
hovertemplate="%{y} : %{x:,.0f} €/mois<extra></extra>"))
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(style_fig(fig, ytitle="").update_layout(hovermode="closest",
xaxis_title="€ / mois en moyenne"), use_container_width=True)
st.subheader("Solde du compte courant (jour par jour)")
bal = client.balance_series()
fig = go.Figure(go.Scatter(x=bal.index, y=bal.values, mode="lines", line=dict(width=2, color=C_BLUE),
name="Solde", hovertemplate="%{y:,.0f} €"))
fig.add_hline(y=float(settings.cushion), line_dash="dot", line_color=C_GRAY,
annotation_text=f"Matelas {euro(settings.cushion)}", annotation_position="bottom left")
st.plotly_chart(style_fig(fig, 300), use_container_width=True)
st.caption(f"💤 Aujourd'hui, {euro(max(0, float(client.current_balance) - float(settings.cushion)))} "
"dorment au-delà du matelas, sans rapporter d'intérêts.")
c1, c2, c3 = st.columns(3)
c1.metric("Variabilité des dépenses (CV)", f"{analysis.cv_expenses:.0%}",
f"max {CFG.eligibility.max_expense_cv:.0%}", delta_color="off", delta_arrow="off")
c2.metric("Rentrées régulières", f"{analysis.income_regularity:.0%}", "min 60 %", delta_color="off", delta_arrow="off")
c3.metric("Creux de trésorerie du mois", euro(analysis.intramonth_need),
"avant l'arrivée du salaire", delta_color="off", delta_arrow="off")
with st.expander("Opérations récurrentes détectées"):
rec = analysis.recurring[analysis.recurring["récurrente"]][
["sign", "libellé", "montant_moyen", "fréquence", "cv_montant"]].rename(columns={
"sign": "Sens", "libellé": "Libellé", "montant_moyen": "Montant moyen (€)",
"fréquence": "Fréquence", "cv_montant": "Variabilité"})
st.dataframe(rec.style.format({"Montant moyen (€)": "{:,.2f}", "Fréquence": "{:.0%}",
"Variabilité": "{:.0%}"}), use_container_width=True, hide_index=True)
with st.expander("Mois exceptionnels (exclus de l'apprentissage, couverts par une provision)"):
if len(exc):
st.write(f"Provision mensuelle pour imprévus : {euro(analysis.shock_provision)}")
t = exceptional_transactions(client.transactions, list(exc.index))
st.dataframe(t.assign(date=t["date"].dt.date), use_container_width=True, hide_index=True)
else:
st.write("Aucun mois exceptionnel détecté.")
# ===========================================================================
# 2. Prédiction
# ===========================================================================
with tabs[1]:
fc = analysis.forecast
if fc is None:
st.info("Il faut au moins 12 mois d'historique pour entraîner les modèles de prédiction.")
else:
table = fc.table(CFG.decision.safety_k)
st.subheader("Flux net mensuel : historique et prévision à 12 mois")
hist_x = analysis.monthly.index.to_timestamp()
fx = table.index.to_timestamp()
fig = go.Figure()
fig.add_scatter(x=list(fx) + list(fx[::-1]), y=list(table["net_high"]) + list(table["net_low"][::-1]),
fill="toself", fillcolor="rgba(42,120,214,.15)", line=dict(width=0),
name="Intervalle 90 %", hoverinfo="skip")
fig.add_scatter(x=hist_x, y=analysis.monthly["net"], mode="lines+markers", name="Réel",
line=dict(width=2, color=C_GRAY), marker=dict(size=6), hovertemplate="%{y:,.0f} €")
fig.add_scatter(x=fx, y=table["net"], mode="lines+markers", name="Prévu (ensemble IA)",
line=dict(width=2, color=C_BLUE, dash="dash"), marker=dict(size=7),
hovertemplate="%{y:,.0f} €")
fig.add_hline(y=0, line_color="rgba(128,128,128,.5)", line_width=1)
st.plotly_chart(style_fig(fig, 380), use_container_width=True)
c1, c2 = st.columns(2)
for col, sf, color in ((c1, fc.income, C_BLUE), (c2, fc.expenses, C_ORANGE)):
with col:
st.markdown(f"{sf.name} : réel et prévu")
fig = go.Figure()
fig.add_scatter(x=sf.history.index.to_timestamp(), y=sf.history.values, name="Réel",
line=dict(width=2, color=C_GRAY), hovertemplate="%{y:,.0f} €")
fig.add_scatter(x=sf.forecast.index.to_timestamp(), y=sf.forecast.values, name="Prévu",
line=dict(width=2, color=color, dash="dash"), hovertemplate="%{y:,.0f} €")
st.plotly_chart(style_fig(fig, 260), use_container_width=True)
c1, c2 = st.columns([1, 1])
with c1:
st.markdown("Saisonnalité apprise des dépenses (écart au mois moyen)")
prof = fc.expenses.seasonal_profile
names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
fig = go.Figure(go.Bar(x=names, y=prof.values,
marker_color=[C_ORANGE if v > 0 else C_BLUE for v in prof.values],
hovertemplate="%{x} : %{y:+,.0f} €<extra></extra>"))
st.plotly_chart(style_fig(fig, 280).update_layout(hovermode="closest"), use_container_width=True)
st.caption(f"Tendance des dépenses : {fc.expenses.trend_per_month:+.1f} €/mois · "
f"des rentrées : {fc.income.trend_per_month:+.1f} €/mois.")
with c2:
st.markdown("Précision des modèles (validation sur les derniers mois)")
for sf in (fc.income, fc.expenses):
st.caption(sf.name)
st.dataframe(sf.metrics.style.format({"MAE (€)": "{:,.0f}", "RMSE (€)": "{:,.0f}",
"MAPE (%)": "{:.1f}", "Poids": "{:.0%}"}, na_rep="—"),
use_container_width=True)
with st.expander("Tableau des prévisions"):
show = table.copy()
show.index = show.index.strftime("%m/%Y")
show.columns = ["Rentrées", "Dépenses (+ imprévus)", "Net", "Net prudent", "Net optimiste",
"Dépenses (haut)", "Rentrées (bas)"]
st.dataframe(show.style.format("{:,.0f} €"), use_container_width=True)
with st.expander("Comment fonctionne la prédiction ?"):
st.markdown(f"""
Trois modèles sont entraînés sur l'historique mensuel : {', '.join(MODEL_LABELS.values())}.
Chacun est testé hors échantillon : on se replace à chacun des {CFG.forecast.backtest_months} derniers mois,
on n'utilise que le passé et on compare la prédiction à la réalité.
L'ensemble combine les modèles avec des poids ∝ 1/erreur² : le plus précis compte le plus.
Les mois exceptionnels (grosses dépenses imprévues) sont retirés de l'apprentissage puis
réintégrés sous forme de provision mensuelle ({euro(fc.shock_provision)}/mois).
L'écart-type des erreurs (σ net ≈ {euro(fc.net_sigma(1))}) sert à calculer la marge de sécurité.
""")
# ===========================================================================
# 3. Proposition & réglages
# ===========================================================================
with tabs[2]:
if analysis.status != EligibilityStatus.ELIGIBLE:
st.warning("Le service n'est pas proposé à ce client pour l'instant (voir le statut en haut). "
"Les réglages ci-dessous sont affichés à titre indicatif.")
left, right = st.columns([1, 1])
with left:
st.subheader("⭐ Proposition par défaut")
ps = proposal.settings
st.markdown(f"""
| | Proposé |
|---|---|
| Produit | {proposal.product.label} |
| Matelas de sécurité | {euro(ps.cushion)} |
| Taux d'investissement | {ps.investment_rate * 100:.0f} % du surplus |
| Profil de risque | {(ss.profile or RiskProfile.NEUTRAL).label}{' (présumé)' if proposal.risk_profile_assumed else ''} |
""")
for part in ("produit", "matelas", "taux"):
for r in proposal.reasons[part]:
st.markdown(f"<span class='si-muted'>• {r}</span>", unsafe_allow_html=True)
sc = proposal.product_scores
st.markdown("Adéquation de chaque produit (score)")
fig = go.Figure(go.Bar(x=[sc[p] for p in ProductType], y=[p.label for p in ProductType], orientation="h",
marker_color=[PRODUCT_COLORS[p] for p in ProductType],
text=[f"{sc[p]:.0f}" for p in ProductType], textposition="outside",
hovertemplate="%{y} : %{x:.0f}<extra></extra>"))
st.plotly_chart(style_fig(fig, 200, "").update_layout(hovermode="closest"), use_container_width=True)
with st.expander("📝 Questionnaire profil d'investisseur (affine la proposition)"):
with st.form("form_questionnaire"):
answers = {}
for q in QUESTIONS:
labels = [o[0] for o in q.options]
answers[q.key] = labels.index(st.radio(q.text, labels, key=f"q_{q.key}", index=1))
if st.form_submit_button("Calculer mon profil"):
ss.profile = profile_from_answers(answers)
ss.custom = None
st.rerun()
if ss.profile:
st.success(f"Profil : {ss.profile.label} — la proposition a été mise à jour.")
with right:
st.subheader("✏️ Vos réglages")
cur = settings
with st.form("form_custom"):
cushion = st.number_input("Matelas de sécurité (€)", min_value=0.0, step=100.0,
value=float(cur.cushion))
rate = st.slider("Taux d'investissement (% du surplus calculé)", 10, 100,
int(cur.investment_rate * 100), step=5)
st.markdown("Répartition entre les produits")
weights = {}
cols = st.columns(3)
for i, p in enumerate(ProductType):
with cols[i]:
rate_txt = {ProductType.SAVINGS: f"{CFG.products.savings_rate * 100:.1f} %",
ProductType.TERM: f"{CFG.products.term_rate * 100:.1f} % · {CFG.products.term_months} mois",
ProductType.FUND: f"~{CFG.products.fund_expected_return * 100:.0f} % espéré"}[p]
weights[p] = st.number_input(f"{p.label} (%)", 0, 100,
int(cur.allocation.weights.get(p, 0) * 100), step=5,
help=f"{p.short_description} Taux : {rate_txt}")
st.caption(rate_txt)
submitted = st.form_submit_button("Appliquer mes réglages", type="primary")
if submitted:
total = sum(weights.values())
if total != 100:
st.error(f"La répartition doit totaliser 100 % (actuellement {total} %).")
else:
try:
alloc = Allocation({p: Decimal(w) / 100 for p, w in weights.items() if w > 0})
ss.custom = ClientSettings(cushion=money(cushion), investment_rate=Decimal(rate) / 100,
allocation=alloc, risk_profile=ss.profile or RiskProfile.NEUTRAL,
planned_expenses=list(ss.planned))
st.rerun()
except ValueError as exc:
st.error(str(exc))
if ss.custom is not None and st.button("↩️ Revenir à la proposition par défaut"):
ss.custom = None
st.rerun()
st.markdown(f"Répartition actuelle : {settings.allocation.describe()}")
for w in check_allocation_vs_profile(settings.allocation, ss.profile or RiskProfile.NEUTRAL):
st.warning(w)
if float(settings.cushion) < analysis.mean_expenses * 0.5:
st.warning("Matelas inférieur à 2 semaines de dépenses : risque de découvert en cas d'imprévu.")
st.divider()
st.subheader(f"💶 Décision de fin de mois ({analysis.last_month.strftime('%m/%Y')})")
if analysis.forecast is None:
st.info("Pas encore de prévision disponible.")
else:
d = current_decision(analysis, settings, CFG)
c1, c2 = st.columns([3, 2])
with c1:
comps = d.components
fig = go.Figure(go.Waterfall(
x=[c[0].replace(" ", "<br>", 1) for c in comps] + ["Surplus<br>investissable"],
y=[float(c[1]) for c in comps] + [0],
measure=["absolute"] + ["relative"] * (len(comps) - 1) + ["total"],
connector=dict(line=dict(color="rgba(128,128,128,.4)", width=1)),
increasing=dict(marker_color=C_BLUE), decreasing=dict(marker_color=C_ORANGE),
totals=dict(marker_color=C_AQUA),
hovertemplate="%{x} : %{y:,.0f} €<extra></extra>"))
st.plotly_chart(style_fig(fig, 340).update_layout(hovermode="closest", showlegend=False),
use_container_width=True)
with c2:
st.metric("Montant investi ce mois-ci", euro(d.amount))
if d.amount > 0:
parts = {p: money(d.amount * w) for p, w in settings.allocation.weights.items()}
for p, v in parts.items():
st.markdown(f"• {p.label} : {euro(v)}")
if d.repatriation > 0:
st.warning(f"À rapatrier sur le compte courant : {euro(d.repatriation)}")
for e in d.explanation:
st.markdown(f"<span class='si-muted'>• {e}</span>", unsafe_allow_html=True)
# ===========================================================================
# 4. Dépenses annoncées
# ===========================================================================
with tabs[3]:
st.subheader("Prévenir une grosse dépense")
st.markdown("Annoncez à l'avance une dépense inhabituelle (voiture, travaux, mariage…). Le service "
"réduit progressivement les montants investis pour que l'argent soit disponible à temps, "
"sans devoir revendre des placements en urgence.")
with st.form("form_planned", clear_on_submit=True):
c1, c2, c3 = st.columns([2, 1, 1])
label = c1.text_input("Description", placeholder="Achat voiture")
amount = c2.number_input("Montant (€)", min_value=0.0, step=100.0)
base = pd.Period(analysis.last_month, freq="M").to_timestamp(how="end").date()
due = c3.date_input("Date prévue", value=base + timedelta(days=90), min_value=base + timedelta(days=1))
if st.form_submit_button("Ajouter"):
if not label.strip() or amount <= 0:
st.error("Indiquez une description et un montant positif.")
else:
ss.planned.append(PlannedExpense(label.strip()[:60], money(amount), due))
if ss.custom is not None:
ss.custom.planned_expenses = list(ss.planned)
st.rerun()
if ss.planned:
for i, pe in enumerate(ss.planned):
c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
c1.write(f"{pe.label}")
c2.write(euro(pe.amount))
c3.write(pe.due.strftime("%d/%m/%Y"))
if c4.button("Supprimer", key=f"del_{i}"):
ss.planned.pop(i)
st.rerun()
if analysis.forecast is not None:
d0 = current_decision(analysis, ClientSettings(settings.cushion, settings.investment_rate,
settings.allocation, settings.risk_profile, []), CFG)
d1 = current_decision(analysis, settings, CFG)
st.info(f"Effet sur l'investissement de ce mois-ci : {euro(d0.amount)} → {euro(d1.amount)}. "
f"Provision gardée : {euro(d1.planned_provision)}.")
else:
st.caption("Aucune dépense annoncée.")
# ===========================================================================
# 5. Simulation
# ===========================================================================
with tabs[4]:
if analysis.forecast is None or analysis.history_months < CFG.eligibility.min_history_months:
st.info("Simulation disponible à partir de 24 mois d'historique.")
else:
skey = settings_key(settings)
st.subheader("Et si le service avait tourné sur votre compte ?")
st.caption("Rejeu des derniers mois réels : chaque fin de mois, le moteur ne voit que le passé "
"(pas de triche), décide, et on vérifie le solde jour par jour.")
rp = get_replay(source_key, analysis, skey, settings)
dc, db = rp.daily_cash, rp.daily_baseline
start = rp.monthly.index[0].to_timestamp(how="end")
mask = dc.index >= start - pd.Timedelta(days=60)
fig = go.Figure()
fig.add_scatter(x=db.index[mask], y=db[mask], name="Sans le service", line=dict(width=2, color=C_GRAY))
fig.add_scatter(x=dc.index[mask], y=dc[mask], name="Avec le service (compte courant)",
line=dict(width=2, color=C_BLUE))
fig.add_hline(y=float(settings.cushion), line_dash="dot", line_color=C_ORANGE,
annotation_text="Matelas", annotation_position="bottom left")
st.plotly_chart(style_fig(fig, 320), use_container_width=True)
after = dc[dc.index >= start]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total investi", euro(rp.monthly["investi"].sum()))
c2.metric("Valeur des placements", euro(rp.portfolio.total_value(rp.monthly.index[-1] + 1)))
c3.metric("Solde minimum atteint", euro(after.min()))
c4.metric("Jours sous le matelas", int((after < float(settings.cushion) - 0.01).sum()))
with st.expander("Détail mois par mois"):
show = rp.monthly.copy()
show.index = show.index.strftime("%m/%Y")
st.dataframe(show.style.format({c: "{:,.0f} €" for c in show.columns if c != "statut"}),
use_container_width=True)
st.dataframe(pd.DataFrame(rp.portfolio.ledger), use_container_width=True, hide_index=True)
st.divider()
st.subheader("Projection : des centaines de futurs possibles")
c1, c2 = st.columns(2)
months = c1.slider("Horizon (mois)", 6, 60, 24, step=6)
paths = c2.select_slider("Nombre de scénarios", [100, 200, 500], value=200)
mc = get_mc(source_key, analysis, skey, settings, months, paths)
s = mc.summary()
c = st.columns(4)
c[0].metric("Gain médian vs argent qui dort", euro(s["gain_median"]),
f"{euro(s['gain_p5'])} à {euro(s['gain_p95'])} (90 %)", delta_color="off", delta_arrow="off")
c[1].metric("Probabilité d'être gagnant", f"{s['proba_gain_positif']:.0%}")
c[2].metric("Investi en moyenne / mois", euro(s["investi_median_par_mois"]))
c[3].metric("Risque de passer sous le matelas", f"{s['proba_sous_matelas']:.0%}",
f"découvert : {s['proba_decouvert']:.0%}", delta_color="off", delta_arrow="off")
st.markdown("Gain cumulé par rapport au fait de tout laisser sur le compte courant")
gain = mc.wealth - mc.baseline
pg = mc.percentiles(gain)
x = pg.index.to_timestamp()
fig = go.Figure()
fig.add_scatter(x=list(x) + list(x[::-1]), y=list(pg["p95"]) + list(pg["p5"][::-1]), fill="toself",
fillcolor="rgba(27,175,122,.12)", line=dict(width=0), name="90 % des scénarios",
hoverinfo="skip")
fig.add_scatter(x=list(x) + list(x[::-1]), y=list(pg["p75"]) + list(pg["p25"][::-1]), fill="toself",
fillcolor="rgba(27,175,122,.25)", line=dict(width=0), name="50 % des scénarios",
hoverinfo="skip")
fig.add_scatter(x=x, y=pg["p50"], name="Gain médian", line=dict(width=2, color=C_AQUA),
hovertemplate="%{y:,.0f} €")
fig.add_hline(y=0, line_color="rgba(128,128,128,.5)", line_width=1)
st.plotly_chart(style_fig(fig, 340), use_container_width=True)
st.markdown("Valeur des placements par produit (scénario médian)")
med = {p: np.median(v, axis=0) for p, v in mc.by_product.items()}
fig = go.Figure()
for p in ProductType:
if med[p].max() > 0:
fig.add_scatter(x=x, y=med[p], name=p.label, stackgroup="one", line=dict(width=0.5),
fillcolor=PRODUCT_COLORS[p], marker_color=PRODUCT_COLORS[p],
hovertemplate="%{y:,.0f} €")
st.plotly_chart(style_fig(fig, 300), use_container_width=True)
st.caption("Hypothèses : compte épargne 2 %, placement à terme 3 % (12 mois), fonds ~5 %/an "
"(volatilité 12 %, frais d'entrée 1 %, gestion 1,2 %/an, TOB 1,32 % à la sortie), "
"précompte mobilier inclus. Rendements passés ou simulés ≠ rendements futurs.")
# ===========================================================================
# 6. Retrait
# ===========================================================================
with tabs[5]:
st.subheader("Retirer de l'argent investi")
if analysis.forecast is None or analysis.history_months < CFG.eligibility.min_history_months:
st.info("Disponible pour les comptes éligibles.")
else:
rp = get_replay(source_key, analysis, settings_key(settings), settings)
pf = rp.portfolio
month = rp.monthly.index[-1] + 1
bd = pf.breakdown(month)
c = st.columns(3)
for i, p in enumerate(ProductType):
c[i].metric(p.label, euro(bd[p]))
st.caption("Portefeuille issu du rejeu du service sur l'historique (onglet Simulation).")
with st.form("form_withdraw"):
c1, c2, c3 = st.columns(3)
amt = c1.number_input("Montant souhaité (€)", min_value=0.0, step=500.0,
value=float(min(5000, pf.total_value(month))))
notice = c2.number_input("Annoncé combien de jours à l'avance ?", 0, 365, 0)
brk = c3.checkbox("J'accepte une rupture anticipée du placement à terme si nécessaire")
go_btn = st.form_submit_button("Simuler le retrait")
if go_btn and amt > 0:
plan = pf.plan_withdrawal(money(amt), month, notice_days=int(notice), allow_term_break=brk)
for w in plan.warnings:
(st.error if w.startswith("⚠") else st.warning)(w)
if plan.steps:
st.dataframe(pd.DataFrame([{
"Produit": s.product.label, "Vendu (€)": float(s.gross), "Frais (€)": float(s.costs),
"Reçu (€)": float(s.net), "Délai (jours)": s.delay_days, "Remarque": s.note}
for s in plan.steps]).style.format({"Vendu (€)": "{:,.2f}", "Frais (€)": "{:,.2f}",
"Reçu (€)": "{:,.2f}"}),
use_container_width=True, hide_index=True)
c1, c2, c3 = st.columns(3)
c1.metric("Vous recevez", euro(plan.total_net, 2))
c2.metric("Coût du retrait", euro(plan.total_costs, 2))
c3.metric("Disponible sous", f"{plan.max_delay_days} jour(s)")
if plan.fully_covered and not plan.warnings:
st.success("Retrait sans frais, disponible immédiatement.")
st.info("💡 Conseil : annoncez vos grosses dépenses dans l'onglet « Dépenses annoncées » — "
"le service les prépare à l'avance et vous évitez frais et pertes.")

=============================================================================
13. LIGNE DE COMMANDE
=============================================================================
def run_cli():
ap = argparse.ArgumentParser(description="Smart Invest - rapport client")
ap.add_argument("--demo", choices=list(PROFILES), default="stable")
ap.add_argument("--fichier", help="CSV ou JSON de transactions")
args = ap.parse_args()
client = load_client(args.fichier) if args.fichier else generate_client(args.demo)
a = analyze_account(client)
print(f"
=== {client.name} ===")
for w in client.warnings:
print(" !", w)
print(f"Solde actuel : {eur(client.current_balance)} €")
print(f"Historique : {a.history_months} mois")
print(f"Rentrées / dépenses : {eur(a.mean_income)} € / {eur(a.mean_expenses)} € par mois")
print(f"Score de stabilité : {a.stability_score}/100")
print(f"Statut : {a.status.label}")
for r in a.reasons:
print(" -", r)
if a.forecast is None or not a.eligible:
print("
→ Le service n'est pas proposé à ce client pour l'instant.")
return
p = build_proposal(a)
s = p.settings
print(f"
Proposition par défaut : {p.product.label}, matelas {eur(s.cushion)} €, "
f"taux {s.investment_rate * 100:.0f} %")
for part in p.reasons.values():
for r in part:
print(" -", r)
d = current_decision(a, s)
print(f"
Décision fin {a.last_month.strftime('%m/%Y')} : investir {eur(d.amount)} €")
for e in d.explanation:
print(" -", e)
print("
Prévisions (12 mois) :")
t = a.forecast.table()
for m, r in t.iterrows():
print(f" {m.strftime('%m/%Y')} rentrées {eur(r.income):>8} € dépenses {eur(r.expenses):>8} € "
f"net {eur(r.net):>8} € (prudent {eur(r.net_low)} €)")

def _inside_streamlit() -> bool:
try:
from streamlit.runtime import exists
return exists()
except Exception: # noqa: BLE001
return False

if __name__ == "__main__":
if _inside_streamlit():
run_app()
else:
run_cli()