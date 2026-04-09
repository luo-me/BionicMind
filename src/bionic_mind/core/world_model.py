from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class Prediction:
    predicted_response: str = ""
    confidence: float = 0.5
    context_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_response": self.predicted_response,
            "confidence": self.confidence,
            "context_hash": self.context_hash,
        }


@dataclass
class PredictionError:
    value: float = 0.0
    surprise_level: float = 0.0
    prediction_id: str = ""

    @property
    def is_surprising(self) -> bool:
        return self.surprise_level > 0.5


class WorldModel:
    def __init__(
        self,
        error_threshold: float = 0.5,
        history_size: int = 100,
        adaptation_rate: float = 0.1,
    ):
        self.error_threshold = error_threshold
        self.history_size = history_size
        self.adaptation_rate = adaptation_rate
        self._predictions: dict[str, Prediction] = {}
        self._error_history: list[PredictionError] = []
        self._context_patterns: dict[str, list[str]] = {}
        self._total_predictions = 0
        self._correct_predictions = 0

    def predict(self, context: str, context_hash: str = "") -> Prediction:
        self._total_predictions += 1
        pattern = self._find_matching_pattern(context)
        if pattern and pattern in self._context_patterns:
            past_responses = self._context_patterns[pattern]
            if past_responses:
                most_common = self._most_frequent(past_responses)
                confidence = past_responses.count(most_common) / len(past_responses)
                prediction = Prediction(
                    predicted_response=most_common,
                    confidence=min(confidence + 0.1, 1.0),
                    context_hash=context_hash,
                )
                self._predictions[context_hash] = prediction
                logger.debug(f"WorldModel prediction: confidence={prediction.confidence:.2f}")
                return prediction

        prediction = Prediction(
            predicted_response="",
            confidence=0.3,
            context_hash=context_hash,
        )
        self._predictions[context_hash] = prediction
        return prediction

    def update(self, context_hash: str, actual_response: str) -> PredictionError:
        prediction = self._predictions.get(context_hash)
        if not prediction:
            return PredictionError(value=0.5, surprise_level=0.3)

        if prediction.predicted_response and actual_response:
            similarity = self._compute_similarity(prediction.predicted_response, actual_response)
            error_value = 1.0 - similarity
        else:
            error_value = 0.5

        surprise_level = max(0.0, error_value - prediction.confidence * 0.3)

        pred_error = PredictionError(
            value=error_value,
            surprise_level=surprise_level,
            prediction_id=context_hash,
        )
        self._error_history.append(pred_error)
        if len(self._error_history) > self.history_size:
            self._error_history.pop(0)

        if error_value < self.error_threshold:
            self._correct_predictions += 1
            pattern = self._extract_pattern(prediction.predicted_response or actual_response)
            if pattern:
                if pattern not in self._context_patterns:
                    self._context_patterns[pattern] = []
                self._context_patterns[pattern].append(actual_response)
                if len(self._context_patterns[pattern]) > 50:
                    self._context_patterns[pattern].pop(0)

        del self._predictions[context_hash]
        logger.debug(f"WorldModel update: error={error_value:.2f}, surprise={surprise_level:.2f}")
        return pred_error

    def simulate(self, context: str, candidate_actions: list[str]) -> list[dict[str, Any]]:
        results = []
        for action in candidate_actions:
            pattern = self._extract_pattern(context + " " + action)
            if pattern and pattern in self._context_patterns:
                past = self._context_patterns[pattern]
                success_rate = len(past) / max(len(past), 1)
                results.append({
                    "action": action,
                    "predicted_outcome": self._most_frequent(past) if past else "unknown",
                    "confidence": min(success_rate, 1.0),
                })
            else:
                results.append({
                    "action": action,
                    "predicted_outcome": "unknown",
                    "confidence": 0.2,
                })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def get_avg_error(self) -> float:
        if not self._error_history:
            return 0.5
        return sum(e.value for e in self._error_history) / len(self._error_history)

    def get_accuracy(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_predictions": self._total_predictions,
            "correct_predictions": self._correct_predictions,
            "accuracy": self.get_accuracy(),
            "avg_error": self.get_avg_error(),
            "patterns_learned": len(self._context_patterns),
            "pending_predictions": len(self._predictions),
        }

    def _find_matching_pattern(self, context: str) -> str | None:
        context_lower = context.lower()
        best_match = None
        best_score = 0.0
        for pattern in self._context_patterns:
            score = self._compute_similarity(context_lower, pattern)
            if score > best_score and score > 0.5:
                best_score = score
                best_match = pattern
        return best_match

    @staticmethod
    def _extract_pattern(text: str) -> str:
        words = text.lower().split()
        if len(words) <= 5:
            return " ".join(words)
        return " ".join(words[:5])

    @staticmethod
    def _compute_similarity(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        word_sim = len(words_a & words_b) / len(words_a | words_b)
        chars_a = set(a.lower())
        chars_b = set(b.lower())
        if chars_a and chars_b:
            char_sim = len(chars_a & chars_b) / len(chars_a | chars_b)
        else:
            char_sim = 0.0
        return 0.7 * word_sim + 0.3 * char_sim

    @staticmethod
    def _most_frequent(items: list[str]) -> str:
        if not items:
            return ""
        counts: dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return max(counts, key=counts.get)
