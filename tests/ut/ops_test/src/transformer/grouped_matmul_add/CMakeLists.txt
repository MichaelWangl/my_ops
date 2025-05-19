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

set(_GMMADD_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/grouped_matmul_add/ophost/grouped_matmul_add_tiling.cpp
)

set(_GMMADD_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/grouped_matmul_add/ophost/grouped_matmul_add_tiling.h
)

set(_GMMADD_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/grouped_matmul_add/ophost/
)

set(_GMMADD_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/grouped_matmul_add/ophost/grouped_matmul_add_ops.cpp
)

set(_GMMADD_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/grouped_matmul_add _GMMADD_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   GMMADD
        SNAKE                                   grouped_matmul_add
        PROTO_SOURCES_EXT                       ${_GMMADD_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_GMMADD_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_GMMADD_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_GMMADD_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_GMMADD_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_GMMADD_TargetPrivateLinkLibrariesExt}
)
