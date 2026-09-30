"""
Smart Invest — interface web (Streamlit).

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_invest.analysis import analyze_account, category_breakdown, exceptional_transactions
from smart_invest.config import DEFAULT_CONFIG, eur, money
from smart_invest.data_generator import PROFILES, generate_client
from smart_invest.data_loader import DataValidationError, load_client
from smart_invest.forecasting import MODEL_LABELS
from smart_invest.models import Allocation, ClientSettings, EligibilityStatus, PlannedExpense, ProductType, RiskProfile
from smart_invest.recommender import build_proposal, check_allocation_vs_profile
from smart_invest.risk_profile import QUESTIONS, profile_from_answers
from smart_invest.simulator import current_decision, monte_carlo, replay_history

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
    ss.profile = None                     # profil présumé tant que le questionnaire n'est pas rempli
    ss.planned = []                       # dépenses annoncées
    ss.custom = None                      # réglages modifiés par le client (None = proposition)
    ss.pop("mc_done", None)

proposal = build_proposal(analysis, ss.profile, ss.planned, CFG)
settings: ClientSettings = ss.custom or proposal.settings
settings.planned_expenses = list(ss.planned)
settings.risk_profile = ss.profile or RiskProfile.NEUTRAL

with st.sidebar:
    st.divider()
    st.markdown(f"**{client.name}**")
    st.markdown(f"Solde actuel : **{euro(client.current_balance)}**")
    st.markdown(f"Historique : **{analysis.history_months} mois**")
    st.markdown(f"Statut : **{analysis.status.label}**")
    if ss.custom is not None:
        st.markdown("Réglages : ✏️ *personnalisés*")
    else:
        st.markdown("Réglages : ⭐ *proposition par défaut*")
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
            + "<br>".join(analysis.reasons) + "</div>", unsafe_allow_html=True)

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
    st.caption(f"💤 Aujourd'hui, **{euro(max(0, float(client.current_balance) - float(settings.cushion)))}** "
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
            st.write(f"Provision mensuelle pour imprévus : **{euro(analysis.shock_provision)}**")
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
                st.markdown(f"**{sf.name} : réel et prévu**")
                fig = go.Figure()
                fig.add_scatter(x=sf.history.index.to_timestamp(), y=sf.history.values, name="Réel",
                                line=dict(width=2, color=C_GRAY), hovertemplate="%{y:,.0f} €")
                fig.add_scatter(x=sf.forecast.index.to_timestamp(), y=sf.forecast.values, name="Prévu",
                                line=dict(width=2, color=color, dash="dash"), hovertemplate="%{y:,.0f} €")
                st.plotly_chart(style_fig(fig, 260), use_container_width=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Saisonnalité apprise des dépenses** (écart au mois moyen)")
            prof = fc.expenses.seasonal_profile
            names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
            fig = go.Figure(go.Bar(x=names, y=prof.values,
                                   marker_color=[C_ORANGE if v > 0 else C_BLUE for v in prof.values],
                                   hovertemplate="%{x} : %{y:+,.0f} €<extra></extra>"))
            st.plotly_chart(style_fig(fig, 280).update_layout(hovermode="closest"), use_container_width=True)
            st.caption(f"Tendance des dépenses : {fc.expenses.trend_per_month:+.1f} €/mois · "
                       f"des rentrées : {fc.income.trend_per_month:+.1f} €/mois.")
        with c2:
            st.markdown("**Précision des modèles (validation sur les derniers mois)**")
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
* **Trois modèles** sont entraînés sur l'historique mensuel : {', '.join(MODEL_LABELS.values())}.
* Chacun est **testé hors échantillon** : on se replace à chacun des {CFG.forecast.backtest_months} derniers mois,
  on n'utilise que le passé et on compare la prédiction à la réalité.
* L'**ensemble** combine les modèles avec des poids ∝ 1/erreur² : le plus précis compte le plus.
* Les **mois exceptionnels** (grosses dépenses imprévues) sont retirés de l'apprentissage puis
  réintégrés sous forme de **provision mensuelle** ({euro(fc.shock_provision)}/mois).
* L'**écart-type des erreurs** (σ net ≈ {euro(fc.net_sigma(1))}) sert à calculer la marge de sécurité.
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
| **Produit** | {proposal.product.label} |
| **Matelas de sécurité** | {euro(ps.cushion)} |
| **Taux d'investissement** | {ps.investment_rate * 100:.0f} % du surplus |
| **Profil de risque** | {(ss.profile or RiskProfile.NEUTRAL).label}{' (présumé)' if proposal.risk_profile_assumed else ''} |
""")
        for part in ("produit", "matelas", "taux"):
            for r in proposal.reasons[part]:
                st.markdown(f"<span class='si-muted'>• {r}</span>", unsafe_allow_html=True)
        sc = proposal.product_scores
        st.markdown("**Adéquation de chaque produit (score)**")
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
                st.success(f"Profil : **{ss.profile.label}** — la proposition a été mise à jour.")

    with right:
        st.subheader("✏️ Vos réglages")
        cur = settings
        with st.form("form_custom"):
            cushion = st.number_input("Matelas de sécurité (€)", min_value=0.0, step=100.0,
                                      value=float(cur.cushion))
            rate = st.slider("Taux d'investissement (% du surplus calculé)", 10, 100,
                             int(cur.investment_rate * 100), step=5)
            st.markdown("**Répartition entre les produits**")
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
        st.markdown(f"Répartition actuelle : **{settings.allocation.describe()}**")
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
                    st.markdown(f"• {p.label} : **{euro(v)}**")
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
                "**réduit progressivement les montants investis** pour que l'argent soit disponible à temps, "
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
            c1.write(f"**{pe.label}**")
            c2.write(euro(pe.amount))
            c3.write(pe.due.strftime("%d/%m/%Y"))
            if c4.button("Supprimer", key=f"del_{i}"):
                ss.planned.pop(i)
                st.rerun()
        if analysis.forecast is not None:
            d0 = current_decision(analysis, ClientSettings(settings.cushion, settings.investment_rate,
                                                           settings.allocation, settings.risk_profile, []), CFG)
            d1 = current_decision(analysis, settings, CFG)
            st.info(f"Effet sur l'investissement de ce mois-ci : **{euro(d0.amount)} → {euro(d1.amount)}**. "
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

        st.markdown("**Gain cumulé par rapport au fait de tout laisser sur le compte courant**")
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

        st.markdown("**Valeur des placements par produit (scénario médian)**")
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
