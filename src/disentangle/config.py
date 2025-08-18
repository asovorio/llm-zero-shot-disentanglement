from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml

@dataclass
class RunConfig:
    method: str
    dataset: str
    split: str
    chunk_size: int
    seed: int

@dataclass
class ModelConfig:
    provider: str
    name: str
    temperature: float
    max_tokens: int
    structured_outputs: bool

@dataclass
class PathsConfig:
    data_dir: Path
    processed_dir: Path
    results_dir: Path
    prompts_dir: Path

@dataclass
class EvalConfig:
    metrics: List[str]
    report_json: bool
    report_csv: bool

@dataclass
class Config:
    run: RunConfig
    model: ModelConfig
    paths: PathsConfig
    eval: EvalConfig

def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    run = RunConfig(**raw["run"])
    model = ModelConfig(**raw["model"])
    paths = PathsConfig(**raw["paths"])
    evalc = EvalConfig(**raw["eval"])
    return Config(run=run, model=model, paths=paths, eval=evalc)