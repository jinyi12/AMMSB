from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from scripts.fae.tran_evaluation.run_support import (
    load_json_dict,
    resolve_run_checkpoint,
)

CANONICAL_PAIR_TRACK = "transformer_pair_geometry"
RUN_ROLE_BASELINE = "baseline"
RUN_ROLE_TREATMENT = "treatment"

_CANONICAL_PAIR_SPECS = (
    {
        "matrix_cell_id": "transformer|adamw|beta1e3|multi_1248|baseline",
        "relative_run_dir": (
            "results/fae_transformer_patch8_adamw_beta1e3_l128x128/"
            "transformer_patch8_adamw_beta1e3_l128x128"
        ),
        "run_role": RUN_ROLE_BASELINE,
        "run_label": "AdamW + beta1e3",
        "notes": "canonical transformer pair baseline",
    },
    {
        "matrix_cell_id": "transformer|adamw|ntk_prior_balanced|multi_1248|treatment",
        "relative_run_dir": (
            "results/fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5/"
            "transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5"
        ),
        "run_role": RUN_ROLE_TREATMENT,
        "run_label": "AdamW + NTK+Prior",
        "notes": "canonical transformer pair treatment",
    },
)


@dataclass
class RegistryRun:
    matrix_cell_id: str
    decoder_type: str
    optimizer: str
    loss_type: str
    scale: str
    regularizer: str
    prior_flag: int
    track: str
    status: str
    paper_track: str
    best_run_dir: str
    notes: str
    run_role: str = ""
    run_label: str = ""


def resolve_regularizer(
    *,
    latent_regularizer: Any = None,
    prior_flag: Any = 0,
    use_prior: Any = False,
) -> str:
    raw = str(latent_regularizer or "").strip().lower()
    if raw in {"none", "diffusion_prior", "sigreg"}:
        return raw
    if bool(use_prior):
        return "diffusion_prior"
    try:
        if int(prior_flag) != 0:
            return "diffusion_prior"
    except Exception:
        pass
    return "none"


def _candidate_score(child: Path) -> tuple[int, int, float]:
    eval_file = child / "eval_results.json"
    score = (
        1 if eval_file.exists() else 0,
        1 if child.name.startswith("run_") else 0,
        eval_file.stat().st_mtime if eval_file.exists() else (child / "args.json").stat().st_mtime,
    )
    return score


def resolve_explicit_run_dir(
    run_dir: Path,
    *,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if (run_dir / "args.json").exists():
        return run_dir

    candidates: list[tuple[tuple[int, int, float], Path]] = []
    if run_dir.exists():
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "args.json").exists():
                continue
            try:
                resolve_run_checkpoint(child, repo_root=repo_root, roots=roots)
            except Exception:
                continue
            candidates.append((_candidate_score(child), child))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].resolve()
    return run_dir


def _infer_scale_label(args_json: dict[str, Any], run_dir: Path) -> str:
    scale = str(args_json.get("scale", "") or "").strip()
    if scale:
        return scale

    for key in ("encoder_multiscale_sigmas", "decoder_multiscale_sigmas"):
        raw = str(args_json.get(key, "") or "").strip()
        if raw:
            try:
                sigmas = tuple(int(round(float(tok.strip()))) for tok in raw.split(",") if tok.strip())
            except Exception:
                sigmas = ()
            if sigmas == (1, 2, 4, 8):
                return "multi_1248"
            if len(sigmas) == 1:
                return f"single_{sigmas[0]}"

    latent_dim = args_json.get("latent_dim")
    try:
        latent_dim_int = int(latent_dim)
    except Exception:
        latent_dim_int = None

    parent_name = run_dir.parent.name.lower()
    run_name = run_dir.name.lower()
    path_blob = f"{parent_name} {run_name}"

    if latent_dim_int is not None and "latent" in path_blob:
        return f"latent{latent_dim_int}"
    if latent_dim_int is not None:
        return f"latent{latent_dim_int}"
    if "single" in path_blob or "sigma1" in path_blob:
        return "single"
    if "multi" in path_blob:
        return "multi"
    return "unspecified"


def registry_run_from_explicit_dir(
    run_dir: Path,
    *,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
    run_role: str = "",
    run_label: str = "",
    matrix_cell_id: str = "",
    track: str = "manual",
    status: str = "complete",
    paper_track: str = "manual",
    notes: str = "explicit_run_dir",
) -> RegistryRun:
    run_dir = resolve_explicit_run_dir(run_dir, repo_root=repo_root, roots=roots)
    args_json = load_json_dict(run_dir / "args.json")
    if not args_json:
        raise FileNotFoundError(f"Missing or invalid args.json in {run_dir}")

    optimizer = str(args_json.get("optimizer", "")).strip().lower() or "unknown"
    loss_type = str(args_json.get("loss_type", "")).strip().lower() or "unknown"
    decoder_type = str(args_json.get("decoder_type", "")).strip().lower() or "unknown"
    regularizer = resolve_regularizer(
        latent_regularizer=args_json.get("latent_regularizer"),
        use_prior=args_json.get("use_prior", False),
    )
    scale = _infer_scale_label(args_json, run_dir)

    return RegistryRun(
        matrix_cell_id=str(matrix_cell_id or run_dir.parent.name or run_dir.name),
        decoder_type=decoder_type,
        optimizer=optimizer,
        loss_type=loss_type,
        scale=scale,
        regularizer=regularizer,
        prior_flag=int(regularizer != "none"),
        track=str(track),
        status=str(status),
        paper_track=str(paper_track),
        best_run_dir=str(run_dir.resolve()),
        notes=str(notes),
        run_role=str(run_role),
        run_label=str(run_label),
    )


def canonical_pair_runs(
    *,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
) -> tuple[RegistryRun, RegistryRun]:
    selected = [
        registry_run_from_explicit_dir(
            repo_root / str(spec["relative_run_dir"]),
            repo_root=repo_root,
            roots=roots,
            matrix_cell_id=str(spec["matrix_cell_id"]),
            track=CANONICAL_PAIR_TRACK,
            status="complete",
            paper_track="",
            run_role=str(spec["run_role"]),
            run_label=str(spec["run_label"]),
            notes=str(spec["notes"]),
        )
        for spec in _CANONICAL_PAIR_SPECS
    ]
    return order_pair_runs(selected, require_roles=True)


def order_pair_runs(
    runs: Iterable[RegistryRun],
    *,
    require_roles: bool = False,
) -> tuple[RegistryRun, RegistryRun]:
    run_list = list(runs)
    if len(run_list) != 2:
        raise RuntimeError(f"Expected exactly 2 runs for pairwise geometry comparison; got {len(run_list)}.")

    baseline: Optional[RegistryRun] = None
    treatment: Optional[RegistryRun] = None
    unassigned: list[RegistryRun] = []

    for run in run_list:
        role = str(run.run_role or "").strip().lower()
        if role == RUN_ROLE_BASELINE:
            if baseline is not None:
                raise RuntimeError("Duplicate baseline run in canonical geometry selection.")
            baseline = run
            continue
        if role == RUN_ROLE_TREATMENT:
            if treatment is not None:
                raise RuntimeError("Duplicate treatment run in canonical geometry selection.")
            treatment = run
            continue
        unassigned.append(run)

    if require_roles and unassigned:
        raise RuntimeError("Canonical geometry pair entries must define run_role as baseline or treatment.")

    if baseline is None and unassigned:
        baseline = unassigned.pop(0)
        baseline.run_role = RUN_ROLE_BASELINE
    if treatment is None and unassigned:
        treatment = unassigned.pop(0)
        treatment.run_role = RUN_ROLE_TREATMENT
    if unassigned:
        raise RuntimeError("Unresolved extra runs remained after pair-role assignment.")
    if baseline is None or treatment is None:
        raise RuntimeError("Could not resolve one baseline and one treatment run for pairwise geometry comparison.")

    if not str(baseline.run_label).strip():
        baseline.run_label = "Baseline"
    if not str(treatment.run_label).strip():
        treatment.run_label = "Treatment"
    return baseline, treatment
