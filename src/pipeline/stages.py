"""Pipeline stage definitions."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class Stage:
    """A named pipeline step."""

    name: str
    handler: Callable[[dict], dict]


def passthrough_stage(config: dict) -> dict:
    """Default placeholder for stages that are not implemented yet."""
    return {"status": "skipped", "config": config}


def build_default_stages(stage_names: Iterable[str]) -> List[Stage]:
    return [Stage(name=name, handler=passthrough_stage) for name in stage_names]


def run_stages(stages: Iterable[Stage], config: dict) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for stage in stages:
        results[stage.name] = stage.handler(config)
    return results
