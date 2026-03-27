from __future__ import annotations

from typing import Any


_OWNER = "scripts/csp/fae_transport_spec.py"


def _normalize_transformer_latent_shape(raw_shape: Any) -> list[int] | None:
    if raw_shape in (None, "", []):
        return None
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
        return [int(raw_shape[0]), int(raw_shape[1])]
    raise ValueError(
        "Transformer latent shape must be a length-2 sequence "
        f"of [num_latents, emb_dim]. Got {raw_shape!r}."
    )


def _default_vector_transport_info(latent_dim: int) -> dict[str, Any]:
    return {
        "owner": _OWNER,
        "latent_representation": "vector",
        "transport_latent_format": "vector",
        "transport_latent_dim": int(latent_dim),
        "transformer_latent_shape": None,
    }


def _token_transport_info(
    *,
    latent_dim: int,
    token_shape: list[int],
    transport_latent_format: str,
) -> dict[str, Any]:
    return {
        "owner": _OWNER,
        "latent_representation": "token_sequence",
        "transport_latent_format": str(transport_latent_format),
        "transport_latent_dim": int(latent_dim),
        "transformer_latent_shape": list(token_shape),
    }


def infer_fae_transport_info(
    fae_meta: dict[str, Any] | None,
    *,
    latent_dim: int,
    token_transport: str = "flattened",
) -> dict[str, Any]:
    if not isinstance(fae_meta, dict):
        return _default_vector_transport_info(latent_dim)

    arch = fae_meta.get("architecture", {})
    if not isinstance(arch, dict):
        return _default_vector_transport_info(latent_dim)

    latent_representation = str(arch.get("latent_representation", "vector"))
    latent_dim_value = int(latent_dim)

    if latent_representation == "vector":
        return _default_vector_transport_info(latent_dim_value)

    if latent_representation != "token_sequence":
        raise ValueError(
            "Unsupported FAE latent representation for CSP transport: "
            f"{latent_representation!r}."
        )

    token_shape = _normalize_transformer_latent_shape(arch.get("transformer_latent_shape"))
    if token_shape is None:
        num_latents = arch.get("transformer_num_latents")
        emb_dim = arch.get("transformer_emb_dim")
        if num_latents in (None, "", []) or emb_dim in (None, "", []):
            raise ValueError(
                "Transformer FAE metadata is missing transformer token shape information."
            )
        token_shape = [int(num_latents), int(emb_dim)]

    expected_latent_dim = int(token_shape[0] * token_shape[1])
    if expected_latent_dim != latent_dim_value:
        raise ValueError(
            "Transformer latent metadata does not match the flattened transport dimension: "
            f"token_shape={token_shape}, latent_dim={latent_dim_value}."
        )

    token_transport_mode = str(token_transport)
    if token_transport_mode == "flattened":
        return _token_transport_info(
            latent_dim=latent_dim_value,
            token_shape=token_shape,
            transport_latent_format="flattened_tokens",
        )
    if token_transport_mode == "token_native":
        return _token_transport_info(
            latent_dim=latent_dim_value,
            token_shape=token_shape,
            transport_latent_format="token_native",
        )
    raise ValueError(
        "token_transport must be one of {'flattened', 'token_native'}, "
        f"got {token_transport!r}."
    )


def validate_fae_transport_info(
    transport_info: dict[str, Any] | None,
    *,
    latent_dim: int,
    fae_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_dim = int(latent_dim)
    if transport_info is None:
        return infer_fae_transport_info(fae_meta, latent_dim=expected_dim)
    if not isinstance(transport_info, dict):
        raise ValueError(f"transport_info must be a dict when present; got {type(transport_info)}.")

    normalized = dict(transport_info)
    normalized.setdefault("owner", _OWNER)
    normalized["latent_representation"] = str(normalized.get("latent_representation", "vector"))
    normalized["transport_latent_format"] = str(normalized.get("transport_latent_format", "vector"))
    normalized["transport_latent_dim"] = int(normalized.get("transport_latent_dim", expected_dim))

    if normalized["transport_latent_dim"] != expected_dim:
        raise ValueError(
            "transport_info transport_latent_dim must match the stored archive latent dimension; "
            f"got transport_latent_dim={normalized['transport_latent_dim']} and latent_dim={expected_dim}."
        )

    if normalized["transport_latent_format"] == "vector":
        normalized["latent_representation"] = "vector"
        normalized["transformer_latent_shape"] = None
        return normalized

    if normalized["transport_latent_format"] not in {"flattened_tokens", "token_native"}:
        raise ValueError(
            "Unsupported transport_latent_format in archive: "
            f"{normalized['transport_latent_format']!r}."
        )

    normalized["latent_representation"] = "token_sequence"
    token_shape = _normalize_transformer_latent_shape(normalized.get("transformer_latent_shape"))
    if token_shape is None:
        inferred = infer_fae_transport_info(fae_meta, latent_dim=expected_dim)
        token_shape = inferred["transformer_latent_shape"]
    if token_shape is None:
        raise ValueError(
            "Token-sequence transport info requires transformer_latent_shape metadata."
        )
    if int(token_shape[0] * token_shape[1]) != expected_dim:
        raise ValueError(
            "transformer_latent_shape does not match the stored archive latent dimension; "
            f"got transformer_latent_shape={token_shape}, latent_dim={expected_dim}."
        )
    normalized["transformer_latent_shape"] = token_shape
    return normalized


__all__ = [
    "infer_fae_transport_info",
    "validate_fae_transport_info",
]
