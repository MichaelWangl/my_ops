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

set(_MoeComputeExpertTokens_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_compute_expert_tokens/ophost/moe_compute_expert_tokens_tiling.cc
)

set(_MoeComputeExpertTokens_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_compute_expert_tokens/ophost/moe_compute_expert_tokens_tiling.h
)

set(_MoeComputeExpertTokens_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/moe/moe_compute_expert_tokens/ophost/
)

set(_MoeComputeExpertTokens_TargetPrivateIncludesExt
    ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_compute_expert_tokens _MoeComputeExpertTokens_KernelSourcesExt)
set(_MoeComputeExpertTokens_CompileDefintions
        MoeComputeExpertTokens_UT
)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeComputeExpertTokens
        SNAKE                                   moe_compute_expert_tokens
        KERNEL_TILING_DATA_DEF_H                ${_MoeComputeExpertTokens_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MoeComputeExpertTokens_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MoeComputeExpertTokens_CompileDefintions}
        TILING_SOURCES_EXT                      ${_MoeComputeExpertTokens_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MoeComputeExpertTokens_TargetPrivateIncludesExt}
)
