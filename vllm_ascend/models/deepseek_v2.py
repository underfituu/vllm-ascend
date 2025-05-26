# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# # Adapted from
# # vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
# # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# # vllm-project/vllm/vllm/model_executor/models/deepseek_v2.py
# """Inference-only DeepseekV2/DeepseekV3 model."""

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch_npu
import torchair
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.distributed import (get_dp_group, get_pp_group,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.deepseek_v2 import \
    DeepseekV2ForCausalLM  # ruff: noqa: E501
from vllm.model_executor.models.deepseek_v2 import \
    yarn_get_mscale  # ruff: noqa: E501
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2Attention,
                                                    DeepseekV2DecoderLayer,
                                                    DeepseekV2MLAAttention)
from vllm.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import dispose_tensor
from vllm_ascend.distributed.parallel_state import get_wp_group
from vllm_ascend.model_executor.layers.linear import (
    MergedColumnParallelLinearWp, RowParallelLinearWp)
from torch.nn import functional as F
from vllm.logger import logger
@dataclass
class DPMetadataForPadding:
    cu_tokens_across_dp_cpu: torch.Tensor
    lengths: torch.Tensor
    max_length: int
    pad_size: torch.Tensor
    atten_unpad_mask: torch.Tensor

_dp_metadata_for_padding: Optional[DPMetadataForPadding] = None

def padding_aligned_tp(dp_rank, data: torch.Tensor) -> torch.Tensor:


    pad_size = _dp_metadata_for_padding.pad_size

    if pad_size[dp_rank] == 0:
        return data

    return F.pad(data, (0, 0, 0, pad_size[dp_rank]))

def padding_aligned_wp(data: torch.Tensor, is_prefill, layer_idx) -> torch.Tensor:

    lengths = _dp_metadata_for_padding.lengths
    max_length = _dp_metadata_for_padding.max_length

    merged_data = torch.zeros((max_length*len(lengths), data.shape[1]), 
                            dtype=data.dtype, device=data.device)
    padded_starts = 0
    current_pos = 0
    for dp_rank in range(len(lengths)):
        seq_len = lengths[dp_rank].item()
        
        merged_data[current_pos:current_pos + seq_len] = data[padded_starts:padded_starts + seq_len]
        
        current_pos += max_length
        padded_starts += seq_len
    return merged_data


def unpadding_aligned_tp(padded_data: torch.Tensor) -> torch.Tensor:

    atten_unpad_mask = _dp_metadata_for_padding.atten_unpad_mask
    merged_data = padded_data[atten_unpad_mask, :]
    return merged_data

def unpadding_aligned_wp(dp_rank, padded_data: torch.Tensor) -> torch.Tensor:

    lengths = _dp_metadata_for_padding.lengths
    max_length = _dp_metadata_for_padding.max_length
    seq_len = lengths[dp_rank].item()

    padded_data = padded_data[max_length * dp_rank :max_length * dp_rank + seq_len]
    return padded_data

class CustomDeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

        # NOTE: `torch_npu.npu_dequant_swiglu_quant` can only be enabled in dynamic quant
        self.is_dynamic_quant = not isinstance(
            self.gate_up_proj.quant_method,
            UnquantizedLinearMethod) and isinstance(
                self.gate_up_proj.quant_method.quant_method,
                AscendW8A8DynamicLinearMethod)

    def forward(self, x, is_prefill: bool = False, reduce_results: bool = True) -> torch.Tensor:
        if self.is_dynamic_quant:
            x, dynamic_scale = torch_npu.npu_dynamic_quant(x)
            x = torch_npu.npu_quant_matmul(
                x,
                self.gate_up_proj.weight,
                self.gate_up_proj.weight_scale,
                output_dtype=torch.int32,
            )
            x, dynamic_scale = torch_npu.npu_dequant_swiglu_quant(
                x=x,
                weight_scale=self.gate_up_proj.weight_scale_fp32,
                activation_scale=dynamic_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1)
            x = torch_npu.npu_quant_matmul(
                x,
                self.down_proj.weight,
                self.down_proj.weight_scale,
                pertoken_scale=dynamic_scale,
                output_dtype=torch.bfloat16,
            )
            if reduce_results and self.down_proj.tp_size > 1:
               x = tensor_model_parallel_all_reduce(x)
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class CustomDeepseekV2SharedExpertMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinearWp(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinearWp(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

        # NOTE: `torch_npu.npu_dequant_swiglu_quant` can only be enabled in dynamic quant
        self.is_dynamic_quant = not isinstance(
            self.gate_up_proj.quant_method,
            UnquantizedLinearMethod) and isinstance(
                self.gate_up_proj.quant_method.quant_method,
                AscendW8A8DynamicLinearMethod)

    def forward(self, x, is_prefill: bool = False, reduce_results: bool = True) -> torch.Tensor:
        if self.is_dynamic_quant:
            x, dynamic_scale = torch_npu.npu_dynamic_quant(x)
            x = torch_npu.npu_quant_matmul(
                x,
                self.gate_up_proj.weight,
                self.gate_up_proj.weight_scale,
                output_dtype=torch.int32,
            )
            x, dynamic_scale = torch_npu.npu_dequant_swiglu_quant(
                x=x,
                weight_scale=self.gate_up_proj.weight_scale_fp32,
                activation_scale=dynamic_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1)
            x = torch_npu.npu_quant_matmul(
                x,
                self.down_proj.weight,
                self.down_proj.weight_scale,
                pertoken_scale=dynamic_scale,
                output_dtype=torch.bfloat16,
            )
            if reduce_results and self.down_proj.tp_size > 1:
               x = tensor_model_parallel_all_reduce(x)
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class CustomDeepseekV2MoE(nn.Module):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = AscendFusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)

        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = CustomDeepseekV2SharedExpertMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        CustomDeepseekV2MoE.top_k = config.num_experts_per_tok

        self.params_dtype = torch.get_default_dtype()
        self.tp_rank_in_group = get_tp_group().rank_in_group
        self.tp_group = get_tp_group().device_group
        self.dp_size = get_dp_group().world_size
        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)

    def forward(self,
                hidden_states: torch.Tensor,
                is_prefill: bool = False) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_dim)

        #                       MC2                      no mc2  
        #  prefill_req    allreduce+allreduce      allreduce+allreduce 
        #  decode_req     all_gather+allreduce      allreduce+allreduce

        if envs_ascend.VLLM_ENABLE_MC2 and not is_prefill and self.tp_size > 1:
            chunks = torch.chunk(hidden_states, self.tp_size, dim=0)
            hidden_states = chunks[self.tp_rank_in_group]

        if self.dp_size > 1 and self.enable_graph_mode and not is_prefill:
            stream_ctx = torchair.scope.npu_stream_switch(
                "CustomDeepseekV2MoE_dp_graph_decode")
        else:
            stream_ctx = nullcontext()

        with stream_ctx:
            # router_logits: (num_tokens, n_experts)
            # gating after all_gather
            # router_logits, _ = self.gate(hidden_states)

            hidden_states = self.experts(
                hidden_states=hidden_states,
                is_prefill=is_prefill,
                top_k=CustomDeepseekV2MoE.top_k,
                gate=self.gate) * self.routed_scaling_factor



        
        return hidden_states


class CustomDeepseekV2MLAAttention(DeepseekV2MLAAttention):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        # all_reduce into function post-attention—process
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        reduce_results=False,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])
        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        if self.q_lora_rank is not None:
            ckq = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
        else:
            hidden_states_or_q_c = hidden_states
        if self.enable_graph_mode:
            forward_kwargs = {}
            if envs.VLLM_USE_V1:
                output_shape = hidden_states.shape
                output = torch.empty(output_shape,
                                     dtype=hidden_states_or_q_c.dtype,
                                     device=hidden_states_or_q_c.device)
                forward_kwargs['output'] = output

            output = self.mla_attn.impl.forward(self.mla_attn,
                                                hidden_states_or_q_c,
                                                hidden_states, None, kv_cache,
                                                attn_metadata,
                                                **forward_kwargs)
            if envs.VLLM_USE_V1:
                output = output.view(-1, output_shape[-1])
            return output
        else:
            kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            return self.mla_attn(hidden_states_or_q_c,
                                 kv_c_normed,
                                 k_pe,
                                 output_shape=hidden_states.shape)


class CustomDeepseekV2DecoderLayer(DeepseekV2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        # TODO: enable mla in vllm-ascend
        if model_config.use_mla:
            attn_cls = CustomDeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.is_moe = config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace \
                and layer_idx % config.moe_layer_freq == 0
        if self.is_moe:
            self.mlp = CustomDeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.shared_experts = self.mlp.shared_experts if config.n_shared_experts is not None else None
        else:
            self.mlp = CustomDeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_rank_in_group = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size
        self.dp_rank = (0 if self.dp_size == 1 else get_dp_group().rank_in_group)
        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)

    def post_attention_process(self, hidden_states, residual, is_prefill):
        if is_prefill or not self.enable_graph_mode:
            if self.dp_size <= 1 or self.layer_idx < self.config.first_k_dense_replace:
                hidden_states = get_tp_group().all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
            else:
                # padding hidden_states
                hidden_states = padding_aligned_tp(self.dp_rank, hidden_states)
                # RS hidden_states
                hidden_states = dist._functional_collectives.reduce_scatter_tensor(
                    hidden_states,
                    "sum",
                    scatter_dim=0,
                    group=get_tp_group().device_group)
                if self.layer_idx == self.config.first_k_dense_replace:
                    # padding and slice residual
                    reduce_scatter_tokens = hidden_states.size(0)
                    residual = F.pad(residual, (0, 0, 0, reduce_scatter_tokens * self.tp_size - residual.size(0)))
                    start = self.tp_rank_in_group * reduce_scatter_tokens
                    residual = residual[start:start + reduce_scatter_tokens]
                # post layernorm
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
                # 全局 all_gather
                hidden_states = get_wp_group().all_gather(hidden_states, 0)
                # unpad
                hidden_states = unpadding_aligned_tp(hidden_states)

        else:
            if self.tp_size > 1:
                hidden_states = get_tp_group().all_reduce(hidden_states)
            if self.enable_graph_mode and not envs_ascend.VLLM_ENABLE_MC2:
                hidden_states = get_dp_group().all_gather(hidden_states, 0)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_moe_process(self, shared_output, hidden_states, residual, is_prefill):
        if is_prefill or not self.enable_graph_mode:
            hidden_states = shared_output + hidden_states
            if self.dp_size <= 1:
                hidden_states = get_wp_group().all_reduce(hidden_states)
                return hidden_states, None
            hidden_states = padding_aligned_wp(hidden_states, is_prefill, self.layer_idx)

            # RS hidden_states
            hidden_states = dist._functional_collectives.reduce_scatter_tensor(
                hidden_states,
                "sum",
                scatter_dim=0,  
                group=get_wp_group().device_group)
            # add residual
            hidden_states = hidden_states + residual
            residual = hidden_states
            # 全局 all_gather
            hidden_states = get_wp_group().all_gather(hidden_states, 0)
            # unpad
            hidden_states = unpadding_aligned_wp(self.dp_rank, hidden_states)
            if self.layer_idx == self.num_hidden_layers - 1:
                residual = None
            return hidden_states, residual
        else:
            if envs_ascend.VLLM_ENABLE_MC2:
                shared_output = self.shared_experts(hidden_states, is_prefill = False, reduce_results=True)
                num_tokens, hidden_dim = hidden_states.shape
                final_hidden_states = torch.zeros([num_tokens, hidden_dim],
                                                dtype=self.params_dtype,
                                                device="npu")
                dist.all_gather_into_tensor(final_hidden_states, hidden_states,
                                            self.tp_group)
                hidden_states = final_hidden_states
                hidden_states = shared_output + final_hidden_states
                hidden_states = hidden_states + residual
            else:
                hidden_states = shared_output + hidden_states
                hidden_states = get_wp_group().all_reduce(hidden_states)
                hidden_states = hidden_states + residual
            return hidden_states, None


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_process(hidden_states, residual, is_prefill)
        if self.is_moe:
            shared_output = None
            if self.config.n_shared_experts is not None:
                shared_output = self.shared_experts(hidden_states, is_prefill = is_prefill, reduce_results=False)
            
            hidden_states = self.mlp(hidden_states, is_prefill)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
            
            hidden_states, residual = self.post_moe_process(shared_output, hidden_states, residual, is_prefill)

        else:
            hidden_states = self.mlp(hidden_states, is_prefill)
            hidden_states = hidden_states + residual
            residual = None
        
        if isinstance(
                self.mlp,
                CustomDeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual

class CustomDeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomDeepseekV2DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata, is_prefill)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = self.norm(hidden_states)
        return hidden_states


class CustomDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    # add `packed_modules_mapping` in `DeepseekV2ForCausalLM` to support weight merging
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomDeepseekV2Model(vllm_config=vllm_config,
                                           prefix=maybe_prefix(
                                               prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.dp_size = get_dp_group().world_size
        self.dp_rank = (0 if self.dp_size == 1 else get_dp_group().rank_in_group)
        self.tp_size = get_tp_group().world_size
        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if is_prefill or not self.enable_graph_mode:
            cu_tokens_across_dp_cpu = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
            # get padding data
            lengths = torch.cat([cu_tokens_across_dp_cpu[:1], cu_tokens_across_dp_cpu[1:] - cu_tokens_across_dp_cpu[:-1]])
            max_length = lengths.max().item()
            max_length = ((max_length + self.tp_size - 1) // self.tp_size) * self.tp_size
            pad_size = -(lengths - max_length)

            # 生成索引掩码（核心优化）
            
            group_indices = torch.arange(self.dp_size)
            position_matrix = torch.arange(max_length).expand(self.dp_size, max_length)
            lengths_tensor = lengths.view(-1, 1)
            atten_unpad_mask = (position_matrix < lengths_tensor).view(-1).to("npu", non_blocking=True)

            global _dp_metadata_for_padding
            _dp_metadata_for_padding = DPMetadataForPadding(cu_tokens_across_dp_cpu, lengths, max_length, pad_size, atten_unpad_mask)

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds, is_prefill)
        if is_prefill or not self.enable_graph_mode:
            del atten_unpad_mask
        return hidden_states


class CustomDeepseekV3ForCausalLM(CustomDeepseekV2ForCausalLM):
    pass
