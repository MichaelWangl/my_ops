# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(EXAMPLES_TRANSFORMER_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# 获取 SocName
get_filename_component(_define_py ${OPS_ADV_CMAKE_DIR}/scripts/examples/get_soc_info.py REALPATH)
execute_process(
        COMMAND ${HI_PYTHON} ${_define_py} "-i=soc_name"
        RESULT_VARIABLE _Rst
        OUTPUT_VARIABLE _SocName
        ERROR_QUIET
)
if (_RST)
    message(WARNING "Can't get socName")
else ()
    if ("${_SocName}x" STREQUAL "x")
        set(_SocName "Unknown")
    endif ()
    string(TOLOWER ${_SocName} _SocName)
endif ()
message(STATUS "SocName=${_SocName}")

# 添加示例工程
if (_SocName MATCHES ascend910b*)
    add_subdirectory(dequant_rope_quant_kvcache)
    add_subdirectory(flash_attention_score)
    add_subdirectory(flash_attention_score_grad)
    add_subdirectory(fused_infer_attention_score)
    add_subdirectory(moe_init_routing_v2)
    add_subdirectory(moe_init_routing)
    add_subdirectory(moe_init_routing_quant_v2)
    add_subdirectory(moe_init_routing_quant)
    add_subdirectory(grouped_bias_add_grad)
    add_subdirectory(moe_finalize_routing_v2)
    add_subdirectory(moe_gating_top_k_softmax)
    add_subdirectory(moe_gating_top_k_softmax_v2)
    add_subdirectory(moe_token_unpermute_grad)
    add_subdirectory(moe_finalize_routing)
    add_subdirectory(moe_token_permute)
    add_subdirectory(moe_compute_expert_tokens)
    add_subdirectory(moe_finalize_routing_v2_grad)
    add_subdirectory(sinkhorn)
    
else()
    message(STATUS "Dequant_rope_quant_kvcache don't support SocName(${_SocName}), skip.")
    message(STATUS "FlashAttentionScore don't support SocName(${_SocName}), skip.")
    message(STATUS "FlashAttentionScoreGrad don't support SocName(${_SocName}), skip.")
    message(STATUS "FusedInferAttentionScore don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeInitRoutingV2 don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeInitRouting don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeInitRoutingQuantV2 don't support SocName(${_SocName}), skip.")
    message(STATUS "GroupedBiasAddGrad don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeInitRoutingQuant don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeFinalizeRoutingV2 don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeGatingTopKSoftmax don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeGatingTopKSoftmaxV2 don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeTokenUnpermuteGrad don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeFinalizeRouting don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeTokenPermute don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeComputeExpertTokens don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeFinalizeRoutingV2Grad don't support SocName(${_SocName}), skip.")
    message(STATUS "Sinkhorn don't support SocName(${_SocName}), skip.")
endif()

# ops only support 310p
if (_SocName MATCHES ascend310p*)
    add_subdirectory(swin_transformer_ln_qkv_quant)
    add_subdirectory(swin_attention_score_quant)
else()
    message(STATUS "SwinTransformerLnQkvQuant don't support SocName(${_SocName}), skip.")
    message(STATUS "SwinTransformerLnQkvQuant don't support SocName(${_SocName}), skip.")
endif()
add_subdirectory(apply_rotary_pos_emb)
add_subdirectory(ffn)
add_subdirectory(incre_flash_attention)
add_subdirectory(prompt_flash_attention)