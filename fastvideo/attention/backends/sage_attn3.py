# SPDX-License-Identifier: Apache-2.0

import torch
from fastvideo.attention.backends.sageattn import sageattn_blackwell

from fastvideo.attention.backends.abstract import (  # FlashAttentionMetadata,
    AttentionBackend, AttentionImpl, AttentionMetadata)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SageAttention3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_THREE"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SageAttention3Impl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        output = sageattn_blackwell(
            query,
            key,
            value,
            is_causal=self.causal)
        return output
