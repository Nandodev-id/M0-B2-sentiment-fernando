"""Logique d'inférence pour la classification de sentiment.

Le modèle CamemBERT chargé sort des labels 5 étoiles
(`'1 star'`, `'2 stars'`, ..., `'5 stars'`). Le métier (Aubergine Hôtels)
veut 3 classes (`négatif`, `neutre`, `positif`).

→ Ton travail dans ce fichier :
   1. Implémenter `predict_sentiment()` (récupère scores 5★ + appelle map).
   2. Implémenter `map_stars_to_sentiment()` (mapping 5★ → 3 classes).
   3. Justifier le seuil retenu en commentaire (cf. brief).

Le pipeline transformers est chargé une seule fois au démarrage, dans
`main.py` (lifespan), et stocké dans `state["pipeline"]`. Tu le récupères
en argument.
"""
from __future__ import annotations

import time
from typing import Any

from app.schemas import Sentiment, SentimentOut


def map_stars_to_sentiment(star_label: str) -> Sentiment:
    """Convertit un label 5 étoiles en classe métier.

    Mapping retenu :
    - 1 et 2 étoiles : négatif
    - 3 étoiles : neutre
    - 4 et 5 étoiles : positif
    """

    match star_label:
        case "1 star" | "2 stars":
            return "négatif"

        case "3 stars":
            return "neutre"

        case "4 stars" | "5 stars":
            return "positif"

        case _:
            raise ValueError(
                f"Label étoile inattendu : {star_label!r}"
            )
        
    # le raisonnement métier dans le README perso async.


def predict_sentiment(
    pipeline: Any,
    text: str,
    model_name: str,
) -> SentimentOut:
    """Exécute l'inférence et adapte le résultat au format métier."""

    start_time = time.perf_counter()

    probabilities = pipeline(
        text,
        top_k=None,
    )

    scores_5_stars = {
        str(entry["label"]): float(entry["score"])
        for entry in probabilities
    }

    top_star_label = max(
        scores_5_stars,
        key=scores_5_stars.get,
    )

    sentiment = map_stars_to_sentiment(top_star_label)

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1_000

    return SentimentOut(
        sentiment=sentiment,
        scores_5_stars=scores_5_stars,
        model_name=model_name,
        latence_ms=round(latency_ms, 2),
    )
    # TODO Tâche 3 — compléter :
    #
    # 1. Mesurer le temps d'inférence (time.perf_counter() avant/après).
    # 2. Appeler `pipeline(text, top_k=None)` pour récupérer toutes les
    #    probabilités (5 entrées, une par étoile).
    # 3. Construire `scores_5_stars: dict[str, float]` à partir du résultat.
    # 4. Identifier le label argmax (la plus haute proba).
    # 5. Appeler `map_stars_to_sentiment(label_argmax)` pour obtenir la
    #    classe métier.
    # 6. Renvoyer un `SentimentOut(...)`.
    raise NotImplementedError("Compléter `predict_sentiment` (Tâche 3).")