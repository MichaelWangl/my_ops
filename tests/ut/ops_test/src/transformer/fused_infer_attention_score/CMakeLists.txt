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

set(_FIA_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.cc
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling_check.cc
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling_register.cc
        ${OPS_ADV_DIR}/src/transformer/prompt_flash_attention/ophost/prompt_flash_attention_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/prompt_flash_attention/ophost/prompt_flash_attention_tiling_register.cc
        ${OPS_ADV_DIR}/src/transformer/fused_infer_attention_score/ophost/fused_infer_attention_score_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/fused_infer_attention_score/ophost/fused_infer_attention_score_tiling_register.cpp
)

set(_FIA_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.h
        ${CMAKE_SOURCE_DIR}/src/transformer/prompt_flash_attention/ophost/prompt_flash_attention_tiling.h
)
set(_FIA_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention/ophost/
        ${CMAKE_SOURCE_DIR}/src/transformer/prompt_flash_attention/ophost/
)

set(_FIA_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/fused_infer_attention_score _FIA_KernelSourcesExt)
set(_FIA_CompileDefintions
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS,prompt_flash_attention,prompt_flash_attention_FIAS fia PFA_UT
)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Fia
        SNAKE                                   fused_infer_attention_score
        KERNEL_TILING_DATA_DEF_H                ${_FIA_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_FIA_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_FIA_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FIA_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FIA_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FIA_TargetPrivateLinkLibrariesExt}
)

