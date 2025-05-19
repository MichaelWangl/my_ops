# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_MOE_GATING_TOP_K_SOFTMAX_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling_perf.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling_k_full_load.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling_e_k_full_load.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling_base.cpp
)

set(_MOE_GATING_TOP_K_SOFTMAX_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_gating_top_k_softmax/ophost/moe_gating_top_k_softmax_tiling_base.h
)

set(_MOE_GATING_TOP_K_SOFTMAX_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_gating_top_k_softmax/ophost
)

set(_MOE_GATING_TOP_K_SOFTMAX_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${OPS_ADV_DIR}/src/transformer/moe_gating_top_k_softmax _MOE_GATING_TOP_K_SOFTMAX_KernelSourcesExt)

OpsTest_Level2_AddOp(
    SUB_SYSTEM                              transformer
    BRIEF                                   MoeGatingTopKSoftmax
    SNAKE                                   moe_gating_top_k_softmax
    KERNEL_TILING_DATA_DEF_H                ${_MOE_GATING_TOP_K_SOFTMAX_KernelTilingDataDefH}
    KERNEL_SOURCES_EXT                      ${_MOE_GATING_TOP_K_SOFTMAX_KernelSourcesExt}
    KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_GATING_TOP_K_SOFTMAX_KernelPrivateCompileDefinitionsExt}
    TILING_SOURCES_EXT                      ${_MOE_GATING_TOP_K_SOFTMAX_TilingSourcesExt}
    TILING_PRIVATE_INCLUDES_EXT             ${_MOE_GATING_TOP_K_SOFTMAX_KernelTilingExtInclude}
    UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MOE_GATING_TOP_K_SOFTMAX_TargetPrivateLinkLibrariesExt}
)