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

set(_MOE_INIT_ROUTING_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing/ophost/moe_init_routing_tiling.cpp
)

set(_MOE_INIT_ROUTING_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing/ophost/moe_init_routing_tiling.h
)

set(_MOE_INIT_ROUTING_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing/ophost/
)

set(_MOE_INIT_ROUTING_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_MOE_INIT_ROUTING_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing _MOE_INIT_ROUTING_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeInitRouting
        SNAKE                                   moe_init_routing
        KERNEL_TILING_DATA_DEF_H                ${_MOE_INIT_ROUTING_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MOE_INIT_ROUTING_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_MOE_INIT_ROUTING_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MOE_INIT_ROUTING_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_INIT_ROUTING_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MOE_INIT_ROUTING_TargetPrivateLinkLibrariesExt}
)
