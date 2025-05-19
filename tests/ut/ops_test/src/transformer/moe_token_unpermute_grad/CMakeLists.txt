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

set(_MOE_TOKEN_UNPERMUTE_GRAD_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_token_unpermute_grad/ophost/moe_token_unpermute_grad_tiling.cpp
)

set(_MOE_TOKEN_UNPERMUTE_GRAD_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_token_unpermute_grad/ophost/moe_token_unpermute_grad_tiling.h
)

set(_MOE_TOKEN_UNPERMUTE_GRAD_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_token_unpermute_grad/ophost/
)

set(_MOE_TOKEN_UNPERMUTE_GRAD_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_MOE_TOKEN_UNPERMUTE_GRAD_CompileDefintions
        DTYPE_PERMUTED_TOKENS=float
        DTYPE_PROBS=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_token_unpermute_grad _MOE_TOKEN_UNPERMUTE_GRAD_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeTokenUnPermuteGrad
        SNAKE                                   moe_token_unpermute_grad
        KERNEL_TILING_DATA_DEF_H                ${_MOE_TOKEN_UNPERMUTE_GRAD_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MOE_TOKEN_UNPERMUTE_GRAD_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_MOE_TOKEN_UNPERMUTE_GRAD_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MOE_TOKEN_UNPERMUTE_GRAD_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_TOKEN_UNPERMUTE_GRAD_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MOE_TOKEN_UNPERMUTE_GRAD_TargetPrivateLinkLibrariesExt}
)