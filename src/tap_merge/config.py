"""Validated paper defaults. Experiment overrides remain visible in resolved configs."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import yaml


@dataclass
class OptimizerConfig:
    lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.999
    rho: float = 0.9
    kappa_ef: float = 9.0
    eps: float = 1e-8
    rank: int = 16
    rank_min: int = 8
    rank_max: int = 128
    rank_increment: int = 8
    refresh_every: int = 10
    check_every: int = 50
    power_steps: int = 1
    theta_low: float = 0.85
    theta_high: float = 0.95
    hysteresis: int = 3
    beta_c: float = 0.99
    beta_b: float = 0.9
    beta_f: float = 0.9
    # Fraction of the TWO FP32 Adam moment tensors, globally across unique parameters.
    state_budget_ratio: float = 0.75
    state_transport: bool = True
    error_feedback: bool = True
    adaptive_rank: bool = True
    refresh_basis: bool = True
    analytical_direction: bool = False

    def __post_init__(self):
        for k in ("beta1", "beta2", "beta_c", "beta_b", "beta_f"):
            if not 0 <= getattr(self, k) < 1:
                raise ValueError(f"{k} must be in [0,1)")
        if not 0 <= self.rho <= 1 or self.kappa_ef < 0 or self.lr <= 0 or self.eps <= 0:
            raise ValueError("Invalid optimizer scalar")
        if not 1 <= self.rank_min <= self.rank <= self.rank_max:
            raise ValueError("Require 1 <= rank_min <= rank <= rank_max")
        if not 0 <= self.theta_low < self.theta_high <= 1:
            raise ValueError("Invalid rank thresholds")
        for k in ("rank_increment", "refresh_every", "check_every", "power_steps", "hysteresis"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be positive")
        if self.state_budget_ratio <= 0:
            raise ValueError("state_budget_ratio must be positive")


@dataclass
class MergeConfig:
    eps: float = 1e-8
    lambda_rc: float = 0.5
    lambda_ld: float = 0.5
    c_max: float = 1.0
    gamma_q: float = 1 / 3
    gamma_b: float = 1 / 3
    gamma_f: float = 1 / 3
    early_min: float = 0.5
    early_max: float = 0.8
    late_min: float = 0.05
    late_max: float = 0.2
    kappa_early: float = 1.0
    kappa_late: float = 2.0
    lambda_early: float = 0.01
    lambda_late: float = 0.1
    alpha_early: float = 1.0
    # Only ablations override these switches.
    mode: str = "dual"
    uniform_budgets: bool = False
    sign_attenuation: bool = True
    masking: str = "saliency"
    curvature: bool = True
    trajectory: bool = True
    seed: int = 42

    def __post_init__(self):
        if not 0 < self.c_max <= 1 or self.eps <= 0:
            raise ValueError("Invalid c_max or epsilon")
        if not 0 < self.late_min <= self.late_max < self.early_min <= self.early_max <= 1:
            raise ValueError("Require disjoint ordered late/early budgets")
        if not 1 <= self.kappa_early <= self.kappa_late:
            raise ValueError("Require 1 <= kappa_early <= kappa_late")
        if not 0 < self.lambda_early <= self.lambda_late:
            raise ValueError("Require 0 < lambda_early <= lambda_late")
        for values in ((self.lambda_rc, self.lambda_ld), (self.gamma_q, self.gamma_b, self.gamma_f)):
            if min(values) < 0 or not math.isclose(sum(values), 1.0, abs_tol=1e-7):
                raise ValueError("Mixture weights must be nonnegative and sum to one")
        if self.mode not in ("dual", "single_early", "single_late"):
            raise ValueError("Unknown merge mode")
        if self.masking not in ("saliency", "magnitude", "random"):
            raise ValueError("Unknown masking mode")


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Config root must be a mapping")
    return value


def save_config(path, value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))
