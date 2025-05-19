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

set(_SWIN_TRANSFORMER_LN_QKV_QUANT_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/swin_transformer_ln_qkv_quant/ophost/swin_transformer_ln_qkv_quant_tiling.cpp
)

set(_SWIN_TRANSFORMER_LN_QKV_QUANT_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/swin_transformer_ln_qkv_quant/ophost/swin_transformer_ln_qkv_quant_tiling.h
)

set(_SWIN_TRANSFORMER_LN_QKV_QUANT_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/swin_transformer_ln_qkv_quant/ophost/
)

set(_SWIN_TRANSFORMER_LN_QKV_QUANT_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/swin_transformer_ln_qkv_quant/ophost/swin_transformer_ln_qkv_quant_proto.cpp
)

set(_SWIN_TRANSFORMER_LN_QKV_QUANT_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/swin_transformer_ln_qkv_quant _SWIN_TRANSFORMER_LN_QKV_QUANT_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   SwinTransformerLnQkvQuant
        SNAKE                                   swin_transformer_ln_qkv_quant
        PROTO_SOURCES_EXT                       ${_SWIN_TRANSFORMER_LN_QKV_QUANT_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_SWIN_TRANSFORMER_LN_QKV_QUANT_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_SWIN_TRANSFORMER_LN_QKV_QUANT_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_SWIN_TRANSFORMER_LN_QKV_QUANT_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_SWIN_TRANSFORMER_LN_QKV_QUANT_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_SWIN_TRANSFORMER_LN_QKV_QUANT_TargetPrivateLinkLibrariesExt}
)
