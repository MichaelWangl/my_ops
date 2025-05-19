# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

set(_SCALED_MASKED_SOFTMAX_V2_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/scaled_masked_softmax_v2/ophost/scaled_masked_softmax_v2_tiling.cpp
)

set(_SCALED_MASKED_SOFTMAX_V2_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/scaled_masked_softmax_v2/ophost/scaled_masked_softmax_v2_tiling.h
)

set(_SCALED_MASKED_SOFTMAX_V2_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/scaled_masked_softmax_v2/ophost/
)

set(_SCALED_MASKED_SOFTMAX_V2_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_SCALED_MASKED_SOFTMAX_V2_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/scaled_masked_softmax_v2 _SCALED_MASKED_SOFTMAX_V2_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   ScaledMaskedSoftmaxV2
        SNAKE                                   scaled_masked_softmax_v2
        KERNEL_TILING_DATA_DEF_H                ${_SCALED_MASKED_SOFTMAX_V2_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_SCALED_MASKED_SOFTMAX_V2_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_SCALED_MASKED_SOFTMAX_V2_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_SCALED_MASKED_SOFTMAX_V2_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_SCALED_MASKED_SOFTMAX_V2_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_SCALED_MASKED_SOFTMAX_V2_TargetPrivateLinkLibrariesExt}
)