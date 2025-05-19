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

set(_SINKHORN_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/sinkhorn/ophost/sinkhorn_tiling.cpp
)

set(_SINKHORN_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/sinkhorn/ophost/sinkhorn_tiling.h
)

set(_SINKHORN_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/sinkhorn/ophost/
)

set(_SINKHORN_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/sinkhorn/ophost/sinkhorn_proto.cpp
)

set(_SINKHORN_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/sinkhorn _SINKHORN_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Sinkhorn
        SNAKE                                   sinkhorn
        PROTO_SOURCES_EXT                       ${_SINKHORN_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_SINKHORN_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_SINKHORN_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_SINKHORN_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_SINKHORN_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_SINKHORN_TargetPrivateLinkLibrariesExt}
)
