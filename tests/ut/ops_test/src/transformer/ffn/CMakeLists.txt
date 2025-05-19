# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
set(_FFN_TilingSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost/ffn_tiling.cpp
)

set(_FFN_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost
)

set(_FFN_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost/ffn_tiling.h
)

set(_FFN_OpApiSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost/ffn.cpp
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost/aclnn_ffn.cpp
)

set(_FFN_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost/ffn_proto.cpp
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/ffn _FFN_KernelSourcesExt)
set(_FFN_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam ffn fp16 DTYPE_X=half DTYPE_WEIGHT1=half DTYPE_WEIGHT2=half DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
        KernelCtrlParam ffn bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=bfloat16_t DTYPE_WEIGHT2=bfloat16_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
        KernelCtrlParam ffn quant_fp16 DTYPE_X=int8_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=int32_t DTYPE_BIAS2=int32_t DTYPE_Y=half
        KernelCtrlParam ffn quant_bf16 DTYPE_X=int8_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=int32_t DTYPE_BIAS2=int32_t DTYPE_Y=bfloat16_t
        KernelCtrlParam ffn a16w8_fp16 DTYPE_X=half DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
        KernelCtrlParam ffn a16w8_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
        KernelCtrlParam ffn a16w4_fp16 DTYPE_X=half DTYPE_WEIGHT1=int4b_t DTYPE_WEIGHT2=int4b_t DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
        KernelCtrlParam ffn a16w4_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=int4b_t DTYPE_WEIGHT2=int4b_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
)

set(_FFN_TargetPrivateIncludeExt
        ${CMAKE_SOURCE_DIR}/src/transformer/ffn/ophost
)

set(_FFN_TillingPrivateLinkLibrariesExt
)

set(_FFN_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   FFN
        SNAKE                                   ffn
        OPAPI_SOURCES_EXT                       ${_FFN_OpApiSourcesExt}
        PROTO_SOURCES_EXT                       ${_FFN_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_FFN_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FFN_TilingPrivateIncludesExt}
        TILING_PRIVATE_LINK_LIBRARIES_EXT       ${_FFN_TillingPrivateLinkLibrariesExt}
        KERNEL_SOURCES_EXT                      ${_FFN_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_FFN_KernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FFN_KernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_FFN_TargetPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FFN_TargetPrivateLinkLibrariesExt}
)
