"""Évalue le modèle sur les reviews du dataset de démonstration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://localhost:8000/predict"
INPUT_FILE = Path("data/sample_reviews.csv")
OUTPUT_FILE = Path("data/evaluation_results.csv")
REQUEST_TIMEOUT_SECONDS = 30

OUTPUT_FIELDS = [
    "id",
    "hotel",
    "texte",
    "sentiment_attendu",
    "sentiment_predit",
    "top_star",
    "confiance",
    "correct",
]


def predict_sentiment(text: str) -> dict[str, Any]:
    """Appelle l'API de prédiction."""

    request = Request(
        API_URL,
        data=json.dumps({"texte": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def find_top_star(scores: dict[str, float]) -> tuple[str, float]:
    """Retourne le label étoile ayant la probabilité maximale."""

    top_star = max(scores, key=scores.get)

    return top_star, scores[top_star]


def evaluate_reviews() -> list[dict[str, str]]:
    """Compare les prédictions avec la vérité terrain."""

    results: list[dict[str, str]] = []

    with INPUT_FILE.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reviews = csv.DictReader(input_file)

        for review in reviews:
            prediction = predict_sentiment(review["texte"])

            expected = review["sentiment_attendu"]
            predicted = prediction["sentiment"]
            scores = prediction["scores_5_stars"]

            top_star, confidence = find_top_star(scores)
            is_correct = expected == predicted

            result = {
                "id": review["id"],
                "hotel": review["hotel"],
                "texte": review["texte"],
                "sentiment_attendu": expected,
                "sentiment_predit": predicted,
                "top_star": top_star,
                "confiance": f"{confidence:.4f}",
                "correct": str(is_correct),
            }

            results.append(result)

            status = "OK" if is_correct else "ERREUR"

            print(
                f"[{status}] ID {review['id']} — "
                f"attendu={expected}, prédit={predicted}, "
                f"top={top_star}, confiance={confidence:.2%}"
            )

    return results


def save_results(results: list[dict[str, str]]) -> None:
    """Enregistre les résultats dans un fichier CSV."""

    with OUTPUT_FILE.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(results)


def display_summary(results: list[dict[str, str]]) -> None:
    """Affiche uniquement les reviews mal classées."""

    errors = [
        result
        for result in results
        if result["correct"] == "False"
    ]

    print("\n" + "=" * 72)
    print(
        f"Résultat : {len(results) - len(errors)}/{len(results)} "
        "reviews correctement classées"
    )
    print(f"Reviews mal classées : {len(errors)}")
    print("=" * 72)

    for error in errors:
        print(
            f"\nID {error['id']} — {error['hotel']}\n"
            f"Texte    : {error['texte']}\n"
            f"Attendu  : {error['sentiment_attendu']}\n"
            f"Prédit   : {error['sentiment_predit']}\n"
            f"Étoile   : {error['top_star']}\n"
            f"Confiance: {error['confiance']}"
        )


def main() -> None:
    try:
        results = evaluate_reviews()
        save_results(results)
        display_summary(results)

        print(f"\nRésultats enregistrés dans {OUTPUT_FILE}")

    except FileNotFoundError:
        raise SystemExit(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    except HTTPError as error:
        raise SystemExit(
            f"Erreur HTTP {error.code} retournée par l'API."
        )

    except URLError as error:
        raise SystemExit(
            "Impossible de contacter l'API. "
            "Vérifie que Docker est démarré et que api-nlp est healthy. "
            f"Détail : {error.reason}"
        )


if __name__ == "__main__":
    main()

