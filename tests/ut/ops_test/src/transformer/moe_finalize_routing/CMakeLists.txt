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

set(_MoeFinalizeRouting_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_finalize_routing/ophost/moe_finalize_routing_tiling.cc
)

set(_MoeFinalizeRouting_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_finalize_routing/ophost/moe_finalize_routing_tiling.h
)

set(_MoeFinalizeRouting_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_finalize_routing/ophost/
)

set(_MoeFinalizeRouting_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/moe_finalize_routing/ophost/moe_finalize_routing_ops.cc
)

set(_MoeFinalizeRouting_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_finalize_routing _MoeFinalizeRouting_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeFinalizeRouting
        SNAKE                                   moe_finalize_routing
        PROTO_SOURCES_EXT                       ${_MoeFinalizeRouting_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_MoeFinalizeRouting_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MoeFinalizeRouting_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_MoeFinalizeRouting_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_MoeFinalizeRouting_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MoeFinalizeRouting_TargetPrivateLinkLibrariesExt}
)
