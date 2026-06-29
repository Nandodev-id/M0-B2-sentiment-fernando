# M0-B2 — Analyse de sentiment FR pour Aubergine Hôtels

## Présentation

Ce projet a été réalisé dans le cadre du parcours IA DEV-ID de Simplon.

L’objectif est de rendre exploitable un modèle NLP français pré-entraîné afin de qualifier automatiquement les avis clients d’Aubergine Hôtels.

Le modèle utilisé est `cmarkea/distilcamembert-base-sentiment`. Il produit initialement une classification en cinq étoiles. Cette sortie est adaptée en trois classes métier :

* négatif ;
* neutre ;
* positif.

## Objectifs

* Déployer un modèle NLP français avec Docker Compose.
* Exposer le modèle avec une API REST FastAPI.
* Adapter une sortie cinq étoiles en trois classes métier.
* Consommer l’API depuis une interface Streamlit.
* Conserver les probabilités brutes du modèle pour la traçabilité.
* Gérer les erreurs réseau, les timeouts et les entrées invalides.
* Journaliser les prédictions et leur latence.
* Tester le fonctionnement de l’API avec pytest.

## Technologies utilisées

* Python 3.11
* FastAPI
* Streamlit
* Hugging Face Transformers
* DistilCamemBERT
* Docker
* Docker Compose
* Pydantic
* HTTPX
* Loguru
* Pytest

## Installation

Créer le fichier `.env` :

```bash
cp .env.example .env
```

Construire et démarrer les services :

```bash
docker compose up --build -d
```

Vérifier leur état :

```bash
docker compose ps
```

## Accès aux services

* Interface Streamlit : `http://localhost:8501`
* Documentation Swagger : `http://localhost:8000/docs`
* Healthcheck : `http://localhost:8000/health`
* Informations du service : `http://localhost:8000/info`

## Tests

Les tests peuvent être exécutés directement dans le conteneur FastAPI :

```bash
docker compose exec api-nlp pytest -v

## Architecture

```mermaid
flowchart LR
    U[Utilisateur / Équipe qualité] -->|Navigateur| S[Streamlit UI<br/>Port 8501]

    S -->|HTTP POST /predict<br/>timeout 10 s| A[FastAPI api-nlp<br/>Port 8000]
    S -->|HTTP GET /health| A

    A -->|Chargement du pipeline| M[DistilCamemBERT FR<br/>cmarkea/distilcamembert-base-sentiment]
    M -->|Cache Hugging Face| V1[(./models)]

    A -->|Journalisation Loguru| V2[(./logs)]

    A --> R1[/GET /health/]
    A --> R2[/GET /info/]
    A --> R3[/POST /predict/]

    D[Docker Compose] --> S
    D --> A
    D --> H[Healthcheck Docker]

    H -->|Vérifie que l'API répond| A
```

L’application repose sur deux services orchestrés par Docker Compose. L’interface Streamlit envoie les avis clients à l’API FastAPI via le réseau Docker interne. L’API exécute l’inférence avec DistilCamemBERT, adapte le résultat cinq étoiles en trois classes métier, puis retourne le sentiment et les probabilités brutes. Les volumes `./models` et `./logs` permettent de conserver respectivement le cache du modèle et les journaux d’exécution.

## Justification du mapping 5 étoiles vers 3 classes

Le modèle `cmarkea/distilcamembert-base-sentiment` retourne cinq classes natives : `1 star`, `2 stars`, `3 stars`, `4 stars` et `5 stars`.

Le mapping suivant a été retenu :

| Sortie du modèle | Classe métier |
| ---------------- | ------------- |
| `1 star`         | négatif       |
| `2 stars`        | négatif       |
| `3 stars`        | neutre        |
| `4 stars`        | positif       |
| `5 stars`        | positif       |

Les notes de 1 et 2 étoiles représentent une insatisfaction claire et sont donc classées comme négatives. Les notes de 4 et 5 étoiles correspondent à une expérience globalement satisfaisante et sont classées comme positives. La note de 3 étoiles est conservée comme classe neutre, car elle correspond généralement à une expérience moyenne, mitigée ou sans opinion fortement marquée.

Ce découpage permet de limiter les faux positifs. Si toutes les notes allant jusqu’à 3 étoiles étaient considérées comme négatives, l’équipe qualité recevrait trop d’alertes pour des avis simplement moyens. Cela augmenterait inutilement la charge de traitement et réduirait la pertinence du système.

Le mapping cherche également à limiter les faux négatifs métier. Un avis réellement négatif classé comme positif pourrait masquer un problème important, comme une climatisation en panne, une chambre sale ou un service très lent. Les probabilités brutes des cinq étoiles sont donc conservées dans la réponse de l’API afin d’assurer la traçabilité de la décision et d’identifier les prédictions incertaines.

Ce mapping constitue un compromis initial. Il pourra être ajusté après observation des données réelles d’Aubergine Hôtels et discussion avec l’équipe qualité.

## Analyse des erreurs du modèle

Le modèle a été évalué sur les 30 avis présents dans `data/sample_reviews.csv`. Pour chaque avis, la classe prédite par l’API a été comparée au sentiment attendu dans le dataset.

### Résultat global

* Reviews évaluées : **30**
* Reviews correctement classées : **24**
* Reviews mal classées : **6**
* Exactitude observée : **80 %**

Ces résultats montrent que le modèle fonctionne correctement sur les avis contenant un sentiment explicite, mais rencontre davantage de difficultés avec l’ironie, les négations, les comparaisons temporelles et les avis qui mélangent plusieurs opinions.

### Cas 1 — Ironie

> Personnel charmant, comme une porte de prison. Trois quarts d'heure d'attente au check-in.

* Sentiment attendu : `négatif`
* Sentiment prédit : `neutre`
* Étoile dominante : `3 stars`
* Confiance : `32,22 %`
* Type d’erreur : **ironie**

Le début de la phrase contient l’expression positive « personnel charmant ». Le sens négatif repose sur la comparaison ironique « comme une porte de prison ». Le modèle identifie difficilement que le mot positif est utilisé de manière sarcastique.

La confiance relativement faible confirme que le modèle hésite entre plusieurs classes.

### Cas 2 — Négation et litote

> Pas mauvais du tout, on s'attendait à pire vu les avis. Bonne surprise sur le rapport qualité-prix.

* Sentiment attendu : `positif`
* Sentiment prédit : `neutre`
* Étoile dominante : `3 stars`
* Confiance : `66,79 %`
* Type d’erreur : **négation / litote**

L’expression « pas mauvais du tout » exprime une opinion positive à travers une négation. Le texte contient aussi des formulations négatives comme « pire », ce qui peut perturber le modèle.

Le sentiment positif est implicite et moins direct qu’une phrase comme « le séjour était excellent ». Le modèle interprète donc l’avis comme modéré plutôt que réellement positif.

### Cas 3 — Comparatif temporel et sentiment mixte

> Mieux que la dernière fois, c'est déjà ça. Reste que la douche fuit toujours dans la salle de bain.

* Sentiment attendu : `négatif`
* Sentiment prédit : `neutre`
* Étoile dominante : `3 stars`
* Confiance : `36,16 %`
* Type d’erreur : **comparatif temporel / mixte**

L’expression « mieux que la dernière fois » est positive uniquement dans le cadre d’une comparaison avec une expérience antérieure. Elle ne signifie pas que le séjour actuel est satisfaisant.

La seconde phrase indique que le problème principal existe toujours. Le modèle semble équilibrer les deux parties de la review et produire une classe neutre, alors que le défaut persistant justifie une classification négative pour l’équipe qualité.

### Cas 4 — Neutralité factuelle

> Établissement standard, conforme à la description. Rien à signaler de particulier.

* Sentiment attendu : `neutre`
* Sentiment prédit : `positif`
* Étoile dominante : `4 stars`
* Confiance : `54,53 %`
* Type d’erreur : **neutralité interprétée comme satisfaction**

Les expressions « conforme à la description » et « rien à signaler » peuvent être associées à une expérience satisfaisante. Cependant, l’avis ne contient ni enthousiasme ni critique particulière.

Le modèle semble interpréter l’absence de problème comme un sentiment positif, alors que la vérité terrain considère cet avis comme neutre.

### Cas 5 — Ironie et sarcasme

> On a passé un séjour qu'on n'oubliera pas. La climatisation en panne en plein août, sympa.

* Sentiment attendu : `négatif`
* Sentiment prédit : `positif`
* Étoile dominante : `4 stars`
* Confiance : `50,83 %`
* Type d’erreur : **ironie / sarcasme**

Les expressions « séjour qu’on n’oubliera pas » et « sympa » ont une apparence positive lorsqu’elles sont analysées littéralement. Dans le contexte, elles sont utilisées de manière sarcastique pour dénoncer une climatisation en panne pendant une période de forte chaleur.

Il s’agit de l’erreur la plus problématique observée, car un avis négatif nécessitant potentiellement une action rapide est classé comme positif. C’est un faux négatif métier : l’équipe qualité risque de ne pas prioriser cet avis.

### Cas 6 — Ambivalence lexicale

> Séjour neutre, sans plus. Le personnel fait son travail, l'hôtel sa fonction. Rien de mémorable.

* Sentiment attendu : `neutre`
* Sentiment prédit : `négatif`
* Étoile dominante : `2 stars`
* Confiance : `48,86 %`
* Type d’erreur : **ambivalence / expressions modérément négatives**

Le texte indique explicitement que le séjour est neutre. Toutefois, les formulations « sans plus » et « rien de mémorable » comportent une connotation négative.

Le modèle semble donner davantage de poids à cette déception implicite qu’au mot « neutre », ce qui produit une classification négative.

### Conclusion de l’analyse

Après correction du mapping, le modèle classe correctement **24 reviews sur 30**, soit une exactitude observée de **80 %**.

Les erreurs restantes concernent principalement :

* l’ironie et le sarcasme ;
* les négations et les litotes ;
* les comparaisons temporelles ;
* les avis contenant des éléments positifs et négatifs ;
* la différence entre neutralité et satisfaction modérée.

Le cas le plus problématique est l’avis sur la climatisation en panne, classé comme positif. Il s’agit d’un faux négatif métier : un avis réellement négatif ne serait pas priorisé par l’équipe qualité.

Dans une application réelle, les avis avec une faible confiance ou contenant des formulations complexes pourraient être transmis à un opérateur humain. Une autre amélioration serait d’ajouter un seuil de confiance sous lequel aucune décision automatique ne serait prise.
