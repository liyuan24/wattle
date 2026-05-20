"""Flower status artifacts for Wattle's TUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flower:
    name: str
    verb: str
    shape: str
    palette: tuple[int, ...]


FLOWER_ROTATION_SECONDS = 4

FLOWERS: tuple[Flower, ...] = (
    Flower("wattle", "wattling", "✿", (227, 221, 215, 149, 113)),
    Flower("sunflower", "sun flowering", "✹", (220, 214, 208, 178, 136)),
    Flower("daisy", "daisying", "❀", (255, 229, 223, 157, 121)),
    Flower("iris", "irising", "✻", (147, 141, 135, 99, 63)),
    Flower("lotus", "lotusing", "✽", (218, 212, 206, 170, 134)),
    Flower("rose", "rosing", "✾", (211, 205, 199, 163, 127)),
    Flower("tulip", "tuliping", "✤", (203, 197, 191, 155, 119)),
    Flower("peony", "peonying", "✺", (225, 219, 213, 177, 141)),
)


def flower_for_elapsed(elapsed_seconds: int) -> Flower:
    index = max(0, elapsed_seconds) // FLOWER_ROTATION_SECONDS
    return FLOWERS[index % len(FLOWERS)]


def gradient_style(flower: Flower, *, frame: int) -> str:
    color = flower.palette[(frame // 3) % len(flower.palette)]
    return f"\x1b[40;38;5;{color};1m"
