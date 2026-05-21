"""Visualization helpers."""


def summarize_scene(stats: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(stats.items()))
