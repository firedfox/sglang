# SPDX-License-Identifier: Apache-2.0
"""Validate the deliberately narrow initial decode DP admission contract."""

from __future__ import annotations

import math

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.environ import envs


def validate_dp_admission(server_args) -> None:
    cfg = resolving_view(server_args)
    if not cfg.enable_dp_admission:
        return

    def require(condition, message):
        if not condition:
            raise ValueError(f"--enable-dp-admission {message}")

    require(cfg.disaggregation_mode == "decode", "requires PD decode mode.")
    require(
        cfg.attn_dp_size > 1 and cfg.dp_size == 1,
        "requires attention data parallelism (--attn-dp-size > 1).",
    )
    require(
        cfg.nnodes == cfg.pp_size == cfg.dcp_size == cfg.attn_cp_size == 1,
        "currently requires one node and PP, DCP, and attention CP sizes of 1.",
    )
    require(
        cfg.load_balance_method == "total_tokens",
        "requires --load-balance-method total_tokens.",
    )
    require(
        cfg.disable_radix_cache
        and not cfg.disaggregation_decode_enable_radix_cache
        and not cfg.enable_session_radix_cache,
        "requires --disable-radix-cache and no decode or session radix cache.",
    )
    require(
        cfg.disaggregation_transfer_backend in ("mooncake", "mooncake_tcp", "fake")
        and cfg.disaggregation_decode_allocation_policy == "early",
        "currently supports Mooncake (or fake transfer) with early allocation.",
    )
    require(
        cfg.disaggregation_decode_host_receive_threshold == 1.0
        and not cfg.enable_hisparse
        and not cfg.enable_hierarchical_cache
        and not cfg.enable_lmcache
        and not cfg.disaggregation_decode_enable_offload_kvcache,
        "does not support decode host receive, HiSparse, HiCache, LMCache, or KV offload.",
    )
    require(
        not cfg.enable_pd_role_switch
        and not cfg.enable_pdmux
        and cfg.elastic_ep_backend is None
        and cfg.ep_join_mode is None
        and not cfg.enable_dsa_cache_layer_split
        and not cfg.enable_kv_cache_sharding,
        "requires fixed DP topology without role switching, elastic EP, LayerSplit, or KV sharding.",
    )
    require(
        cfg.speculative_algorithm in (None, "EAGLE", "EAGLE3", "NEXTN")
        and (cfg.speculative_eagle_topk or 1) == 1
        and not cfg.speculative_adaptive
        and cfg.decoupled_spec_role == "null",
        "currently supports ordinary decode or fixed top-k=1 EAGLE/EAGLE3/NEXTN.",
    )
    require(
        not cfg.is_embedding and cfg.dllm_algorithm is None and not cfg.enable_lora,
        "currently supports text generation without dLLM or LoRA.",
    )
    require(
        not envs.SGLANG_RUST_SERVER.get()
        and not cfg.smg_grpc_mode
        and not cfg.grpc_mode
        and cfg.grpc_port is None,
        "currently requires the Python HTTP server.",
    )
    require(
        cfg.dp_admission_max_queued_requests >= 0,
        "requires a nonnegative --dp-admission-max-queued-requests.",
    )
    require(
        cfg.num_reserved_decode_tokens >= 0,
        "requires nonnegative --num-reserved-decode-tokens.",
    )
    for field in ("dp_admission_queue_timeout", "dp_admission_snapshot_timeout"):
        value = getattr(cfg, field)
        require(
            math.isfinite(value) and value > 0,
            f"requires a finite positive --{field.replace('_', '-')}.",
        )
