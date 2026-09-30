# Smart Invest — projet KBC

Service qui analyse le compte courant d'un client, **prédit** ses rentrées et dépenses (IA :
régression saisonnière, Holt-Winters, ensemble validé hors échantillon) et **investit
automatiquement chaque fin de mois** l'argent qui dort au-delà de son matelas de sécurité,
dans un ou plusieurs des trois produits (compte épargne, placement à terme, fonds géré).

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py          # interface web complète  → http://localhost:8501
python main.py --demo stable  # rapport en ligne de commande
python main.py --fichier exemples/client_famille.json
python -m pytest -q           # 29 tests (moteur + interface)
```

## Brancher votre « utilisateur par défaut »

Dans l'interface : barre latérale → *Importer un fichier*. En ligne de commande : `--fichier`.
Formats acceptés (voir le dossier `exemples/`) :

**JSON (recommandé, car il porte aussi les infos du client)**
```json
{
  "client": {"nom": "Sophie Martens", "solde_initial": 4200, "date_ouverture": "2024-04-01",
             "matelas": 3000},
  "transactions": [
    {"date": "2024-04-01", "montant": -920.0, "libelle": "Loyer", "categorie": "Logement"},
    {"date": "2024-04-25", "montant": 2850.0, "libelle": "Salaire - Employeur SA", "categorie": "Revenus"}
  ]
}
```
**CSV** : `date;montant;libelle;categorie` (séparateur `;` ou `,`, décimales `,` ou `.`,
dates `JJ/MM/AAAA` ou `AAAA-MM-JJ`). Les colonnes `debit` + `credit` sont aussi reconnues.

* `montant` négatif = dépense, positif = rentrée.
* `solde_initial` (ou `solde_actuel`) est optionnel : s'il manque, il est estimé et un
  avertissement s'affiche. `matelas` est optionnel : s'il manque, le service en propose un.
* Il faut **au moins 24 mois** d'historique pour que le service soit proposé.

## Architecture

| Fichier | Rôle |
|---|---|
| `smart_invest/config.py` | Tous les paramètres chiffrés (taux, seuils, fiscalité) |
| `smart_invest/models.py` | Objets métier : produits, profil de risque, répartition, dépense annoncée |
| `smart_invest/data_loader.py` | Import CSV/JSON souple + validation + agrégation mensuelle |
| `smart_invest/data_generator.py` | 5 clients fictifs réalistes (Belgique) pour la démo |
| `smart_invest/analysis.py` | Statistiques, opérations récurrentes, mois exceptionnels, éligibilité |
| `smart_invest/forecasting.py` | Moteur de prédiction (3 modèles + ensemble) |
| `smart_invest/recommender.py` | Proposition par défaut (produit, matelas, taux) |
| `smart_invest/risk_profile.py` | Questionnaire profil d'investisseur (type MiFID) |
| `smart_invest/decision_engine.py` | Montant à investir chaque fin de mois |
| `smart_invest/products.py` | Les 3 produits, portefeuille, retraits (frais, délais, avertissements) |
| `smart_invest/simulator.py` | Rejeu sur l'historique réel + projection Monte Carlo |
| `app.py` | Interface Streamlit (6 onglets) |

## Règles métier (décisions prises avec le groupe)

1. **Flux** : compte courant → analyse → virement de fin de mois vers les produits choisis.
2. **Éligibilité** (dans cet ordre) :
   historique ≥ 24 mois → sinon *Historique insuffisant* ;
   capacité d'épargne moyenne ≥ 50 €/mois → sinon *Non proposé* ;
   stabilité → sinon *En attente* (réanalysé chaque mois) : coefficient de variation des
   dépenses ≤ 30 % (hors imprévus), ≥ 60 % des rentrées de sources régulières,
   ≤ 25 % de mois exceptionnels.
3. **Montant investi** = taux × (solde − matelas − trésorerie du mois − marge d'incertitude
   − réserve pour déficits prévus − provision des dépenses annoncées).
   * *Trésorerie du mois* : creux observé avant l'arrivée du salaire (loyer le 1er, salaire
     le 25) : sans elle, le compte passerait sous le matelas en milieu de mois.
   * *Marge d'incertitude* = max(1,645 × σ de l'erreur de prédiction, 10 % des dépenses prévues).
     Plus le compte est imprévisible, plus on garde de réserve.
   * *Réserve pour déficits* : pire creux cumulé des 3 prochains mois (ex. vacances d'été).
   * *Dépense annoncée* : montant − ce que les mois d'ici là rapporteront (scénario prudent).
   * Minimum 50 € (sinon rien ce mois-là), plafond 25 000 €/mois (l'argent qui dort déjà est
     donc placé en quelques mois).
   * Si le solde passe sous ce qu'il faut garder, le moteur **rapatrie** de l'argent.
4. **Proposition par défaut** : le service propose d'office UN produit, un matelas
   (1,5 mois de dépenses) et un taux (selon le score de stabilité), avec l'explication.
   Le client peut tout modifier : matelas, taux, et répartition libre sur 1, 2 ou 3 produits.
   Le questionnaire de profil affine la proposition ; un choix plus risqué que le profil
   déclenche un avertissement (non bloquant).
5. **Produits**

| | Compte épargne | Placement à terme | Fonds géré |
|---|---|---|---|
| Rendement | 2 %/an | 3 %/an | ~5 %/an espéré, volatilité 12 % |
| Disponibilité | immédiate | bloqué 12 mois (rupture : 1 % + intérêts perdus) | J+3, vente en plusieurs jours au-delà de 10 000 € |
| Frais / taxes | précompte 15 % au-delà de 1 050 € d'intérêts/an | précompte 30 % | entrée 1 %, gestion 1,2 %/an, TOB 1,32 %, +0,5 % si retrait non annoncé 30 j avant |

   Chaque versement en placement à terme ouvre une tranche de 12 mois, reconduite à l'échéance.
6. **Retrait** : ordre le moins coûteux (épargne → fonds → terme). Messages d'avertissement si
   le retrait est important (exécution en plusieurs fois, prix qui peut varier), non
   annoncé, à perte, ou bloqué sur un placement à terme.

## La prédiction (IA)

* **Régression saisonnière Ridge** : tendance + un effet par mois de l'année.
* **Holt-Winters** : lissage exponentiel, tendance amortie, saisonnalité 12 mois.
* **Saisonnier naïf ajusté** : même mois des années passées, recalé sur le niveau récent.
* **Validation « rolling origin »** sur les 6 derniers mois (on ne voit que le passé) ;
  poids de l'ensemble ∝ 1/MAE². L'erreur de l'ensemble fournit l'intervalle de confiance.
* Les **mois exceptionnels** (score robuste > 3,5 par rapport à la saisonnalité) sont retirés de
  l'apprentissage et remplacés par une **provision mensuelle d'imprévus**.

## Clients de démonstration

| Profil | Résultat attendu |
|---|---|
| Sophie Martens (employée) | Éligible → placement à terme proposé |
| Famille Peeters | Éligible → placement à terme proposé |
| Lucas Janssens (indépendant) | En attente : rentrées irrégulières |
| Emma Dubois (étudiante) | Non proposé : pas de capacité d'épargne |
| Noah Lambert | Historique insuffisant (14 mois) |

## Limites / à vérifier

* Les valeurs fiscales belges (exonération, précompte, TOB) changent chaque année :
  à vérifier dans `config.py` avant une présentation.
* Les rendements du fonds sont simulés (loi normale), pas des rendements réels.
* La sécurité (authentification, chiffrement, journalisation) sera traitée dans un module
  séparé ; les montants sont déjà en `Decimal` et les imports sont validés.
