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

set(_AROPE_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/apply_rotary_pos_emb/ophost/apply_rotary_pos_emb_tiling.cpp
        
)

set(_AROPE_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/apply_rotary_pos_emb/ophost/apply_rotary_pos_emb_tiling.h
)

set(_AROPE_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/apply_rotary_pos_emb/ophost/
)

set(_AROPE_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/apply_rotary_pos_emb/ophost/apply_rotary_pos_emb_proto.cpp
)

set(_AROPE_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

set(_AROPE_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam apply_rotary_pos_emb fp32 ORIG_DTYPE_QUERY=DT_FLOAT32 DTYPE_QUERY=float
        KernelCtrlParam apply_rotary_pos_emb fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_QUERY=half
        KernelCtrlParam apply_rotary_pos_emb bf16 ORIG_DTYPE_QUERY=DT_BF16 DTYPE_QUERY=bfloat16_t
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/apply_rotary_pos_emb _AROPE_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   ARoPE
        SNAKE                                   apply_rotary_pos_emb
        PROTO_SOURCES_EXT                       ${_AROPE_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_AROPE_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_AROPE_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_AROPE_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_AROPE_KernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_AROPE_KernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_AROPE_TargetPrivateLinkLibrariesExt}
)

