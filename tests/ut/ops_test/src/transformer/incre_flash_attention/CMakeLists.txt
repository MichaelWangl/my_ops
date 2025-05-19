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

set(_IFA_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.cc
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling_check.cc
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling_register.cc
)

set(_IFA_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.h
)

set(_IFA_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention _IFA_KernelSourcesExt)

set(_IFA_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS fp16_fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 ORIG_DTYPE_ATTENTION_OUT=DT_FLOAT16
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS fp16_int8 ORIG_DTYPE_QUERY=DT_FLOAT16 ORIG_DTYPE_ATTENTION_OUT=DT_INT8
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS bf16_bf16 ORIG_DTYPE_QUERY=DT_BF16 ORIG_DTYPE_ATTENTION_OUT=DT_BF16
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS bf16_int8 ORIG_DTYPE_QUERY=DT_BF16 ORIG_DTYPE_ATTENTION_OUT=DT_INT8
)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Ifa
        SNAKE                                   incre_flash_attention
        KERNEL_TILING_DATA_DEF_H                ${_IFA_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_IFA_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_IFA_KernelPrivateCompileDefinitionsExt}
        TILING_SOURCES_EXT                      ${_IFA_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_IFA_TargetPrivateLinkLibrariesExt}
)
