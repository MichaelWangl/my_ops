# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
OP_NAME MoeGatingTopKSoftmaxV2
OPTIONS --cce-auto-sync=off
-Wno-deprecated-declarations
-Werror
)

target_sources(op_host_aclnn PRIVATE
moe_gating_top_k_softmax_v2_def.cpp
)

target_sources(optiling PRIVATE
moe_gating_top_k_softmax_v2_tiling_base.cpp
moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp
moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp
moe_gating_top_k_softmax_v2_tiling_perf.cpp
moe_gating_top_k_softmax_v2_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
target_sources(opmaster_ct PRIVATE
moe_gating_top_k_softmax_v2_tiling_base.cpp
moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp
moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp
moe_gating_top_k_softmax_v2_tiling_perf.cpp
moe_gating_top_k_softmax_v2_tiling.cpp
)
endif ()

target_sources(opsproto PRIVATE
moe_gating_top_k_softmax_v2_proto.cpp
)

target_include_directories(optiling PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}
)