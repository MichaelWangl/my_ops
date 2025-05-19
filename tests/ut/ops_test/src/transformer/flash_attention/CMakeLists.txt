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

# 自动反向(FA & FAG)
set(_FA_OpApiSourcesExt
        # FAS
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score.cpp
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/aclnn_flash_attention_score.cpp
        # FAG
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad.cpp
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/aclnn_flash_attention_score_grad.cpp
)
set(_FA_OpProtoSourcesExt
        # FAS
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_proto.cpp
        # FAG
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_proto.cpp
)
file(GLOB _FAS_TilingSourcesExt_cc  "${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling*.cc")
file(GLOB _FAS_TilingSourcesExt_cpp "${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling*.cpp")
file(GLOB _FASG_TilingSourcesExt_cc  "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*.cc")
file(GLOB _FASG_TilingSourcesExt_cpp "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*.cpp")
set(_FA_OpTilingSourcesExt
        # FAS
        ${_FAS_TilingSourcesExt_cc}
        ${_FAS_TilingSourcesExt_cpp}
        # FAG
        ${_FASG_TilingSourcesExt_cc}
        ${_FASG_TilingSourcesExt_cpp}
)
set(_FA_OpTilingPrivateIncludesExt
        # FAS
        # FAG
)
set(_FA_OpKernelSourcesExt
        # FAS
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/flash_attention_score.cpp
        # FAG
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/flash_attention_score_grad.cpp
)
file(GLOB _FASG_OpKernelTilingDataDefH_def "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*_def.h")
set(_FA_OpKernelTilingDataDefH
        # 公共
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        # FAS
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling.h
        # FAG
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling.h
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling_common.h
        ${_FASG_OpKernelTilingDataDefH_def}
)
set(_FA_OpKernelPrivateCompileDefinitionsExt
        KernelCtrlParam flash_attention_score,flash_attention_score_grad fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half
        KernelCtrlParam flash_attention_score,flash_attention_score_grad bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t
        KernelCtrlParam flash_attention_score,flash_attention_score_grad fp32 ORIG_DTYPE_QUERY=DT_FLOAT   DTYPE_DQ=float
)
set(_FA_UTestCommonPrivateIncludeExt
        # FAS
        # FAG
)
set(_FA_UTestCommonPrivateLinkLibrariesExt
        # FAS
        # FAG
        ops_utils_tiling_headers        # 用于 FAG 指定模板优先级相关实现
        ${UTest_NamePrefix}_OpTiling    # 用于 FAG 指定模板优先级相关实现
        error_manager                   # 用于 FAG 指定模板优先级相关实现
)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Fa
        SNAKE                                   flash_attention
        OPAPI_SOURCES_EXT                       ${_FA_OpApiSourcesExt}
        PROTO_SOURCES_EXT                       ${_FA_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_FA_OpTilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FA_OpTilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_FA_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_FA_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FA_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_FA_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FA_UTestCommonPrivateLinkLibrariesExt}
)
