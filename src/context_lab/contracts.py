"""Minimal extension contracts; callers may supply local models without framework dependencies."""

from typing import Protocol


class JSONModel(Protocol):
    def generate(self, instruction: str, data: dict) -> dict: ...


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Classifier(Protocol):
    def classify(self, text: str, labels: list[str]) -> dict: ...
