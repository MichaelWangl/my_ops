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

set(_ROPE_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/rotary_position_embedding/ophost/rotary_position_embedding_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/rotary_position_embedding/ophost/rope_rotate_half_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/rotary_position_embedding/ophost/rope_interleaved_tiling.cpp
)

set(_ROPE_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/rotary_position_embedding/ophost/rotary_position_embedding_tiling.h
)

set(_ROPE_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/rotary_position_embedding/ophost/
)

set(_ROPE_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/rotary_position_embedding/ophost/rotary_position_embedding_proto.cpp
)

set(_ROPE_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/rotary_position_embedding _ROPE_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   RoPE
        SNAKE                                   rotary_position_embedding
        PROTO_SOURCES_EXT                       ${_ROPE_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_ROPE_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_ROPE_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_ROPE_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_ROPE_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_ROPE_TargetPrivateLinkLibrariesExt}
)

