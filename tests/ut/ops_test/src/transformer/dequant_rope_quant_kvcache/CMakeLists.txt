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

set(_DEQUANT_ROPE_QUANT_KVCACHE_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/dequant_rope_quant_kvcache/ophost/dequant_rope_quant_kvcache_tiling.cpp
)

set(_DEQUANT_ROPE_QUANT_KVCACHE_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/dequant_rope_quant_kvcache/ophost/dequant_rope_quant_kvcache_tiling.h
)

set(_DEQUANT_ROPE_QUANT_KVCACHE_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/dequant_rope_quant_kvcache/ophost
)

set(_DEQUANT_ROPE_QUANT_KVCACHE_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

set(_DEQUANT_ROPE_QUANT_KVCACHE_CompileDefintions
        DTYPE_X=half
        DTYPE_COS=half
)
aux_source_directory(${OPS_ADV_DIR}/src/transformer/dequant_rope_quant_kvcache _DEQUANT_ROPE_QUANT_KVCACHE_KernelSourcesExt)

OpsTest_Level2_AddOp(
    SUB_SYSTEM                              transformer
    BRIEF                                   DequantRopeQuantKvcache
    SNAKE                                   dequant_rope_quant_kvcache
    KERNEL_TILING_DATA_DEF_H                ${_DEQUANT_ROPE_QUANT_KVCACHE_KernelTilingDataDefH}
    KERNEL_SOURCES_EXT                      ${_DEQUANT_ROPE_QUANT_KVCACHE_KernelSourcesExt}
    KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_DEQUANT_ROPE_QUANT_KVCACHE_KernelPrivateCompileDefinitionsExt}
    TILING_SOURCES_EXT                      ${_DEQUANT_ROPE_QUANT_KVCACHE_TilingSourcesExt}
    TILING_PRIVATE_INCLUDES_EXT             ${_DEQUANT_ROPE_QUANT_KVCACHE_KernelTilingExtInclude}
    KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_DEQUANT_ROPE_QUANT_KVCACHE_CompileDefintions}
    UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_DEQUANT_ROPE_QUANT_KVCACHE_TargetPrivateLinkLibrariesExt}
)