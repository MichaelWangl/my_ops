# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

set(_MOE_TOKEN_PERMUTE_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_token_permute/ophost/moe_token_permute_tiling.cpp
)

set(_MOE_TOKEN_PERMUTE_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_token_permute/ophost/moe_token_permute_tiling.h
)

set(_MOE_TOKEN_PERMUTE_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_token_permute/ophost/
)

set(_MOE_TOKEN_PERMUTE_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_MOE_TOKEN_PERMUTE_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_token_permute _MOE_TOKEN_PERMUTE_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeTokenPermute
        SNAKE                                   moe_token_permute
        KERNEL_TILING_DATA_DEF_H                ${_MOE_TOKEN_PERMUTE_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MOE_TOKEN_PERMUTE_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_MOE_TOKEN_PERMUTE_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MOE_TOKEN_PERMUTE_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_TOKEN_PERMUTE_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MOE_TOKEN_PERMUTE_TargetPrivateLinkLibrariesExt}
)
