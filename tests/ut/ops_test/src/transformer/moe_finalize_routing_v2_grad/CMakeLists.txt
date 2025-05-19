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

set(_MoeFinalizeRoutingV2Grad_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_finalize_routing_v2_grad/ophost/moe_finalize_routing_v2_grad_tiling.cc
)

set(_MoeFinalizeRoutingV2Grad_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_finalize_routing_v2_grad/ophost/moe_finalize_routing_v2_grad_tiling.h
)

set(_MoeFinalizeRoutingV2Grad_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/moe/moe_finalize_routing_v2_grad/ophost/
)

set(_MoeFinalizeRoutingV2Grad_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/moe_finalize_routing_v2_grad/ophost/moe_finalize_routing_v2_grad_infer.cc
)

set(_MoeFinalizeRoutingV2Grad_TargetPrivateIncludesExt
    ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_finalize_routing_v2_grad _MoeFinalizeRoutingV2Grad_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeFinalizeRoutingV2Grad
        SNAKE                                   moe_finalize_routing_v2_grad
        PROTO_SOURCES_EXT                       ${_MoeFinalizeRoutingV2Grad_OpProtoSourceExt}
        KERNEL_TILING_DATA_DEF_H                ${_MoeFinalizeRoutingV2Grad_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MoeFinalizeRoutingV2Grad_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MoeFinalizeRoutingV2Grad_CompileDefintions}
        TILING_SOURCES_EXT                      ${_MoeFinalizeRoutingV2Grad_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MoeFinalizeRoutingV2Grad_TargetPrivateIncludesExt}
)
