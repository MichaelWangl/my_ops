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

set(_GROUPED_BIAS_ADD_GRAD_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/grouped_bias_add_grad/ophost/grouped_bias_add_grad_tiling.cpp
)

set(_GROUPED_BIAS_ADD_GRAD_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/grouped_bias_add_grad/ophost/grouped_bias_add_grad_tiling_def.h
)

set(_GROUPED_BIAS_ADD_GRAD_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/grouped_bias_add_grad/ophost/
)

set(_GROUPED_BIAS_ADD_GRAD_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_GROUPED_BIAS_ADD_GRAD_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/grouped_bias_add_grad _GROUPED_BIAS_ADD_GRAD_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   GroupedBiasAddGrad
        SNAKE                                   grouped_bias_add_grad
        KERNEL_TILING_DATA_DEF_H                ${_GROUPED_BIAS_ADD_GRAD_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_GROUPED_BIAS_ADD_GRAD_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_GROUPED_BIAS_ADD_GRAD_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_GROUPED_BIAS_ADD_GRAD_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_GROUPED_BIAS_ADD_GRAD_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_GROUPED_BIAS_ADD_GRAD_TargetPrivateLinkLibrariesExt}
)
