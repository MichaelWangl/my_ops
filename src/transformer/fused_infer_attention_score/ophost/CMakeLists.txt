# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME FusedInferAttentionScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
)

set(fused_infer_attention_score_depends transformer/incre_flash_attention transformer/prompt_flash_attention 
    PARENT_SCOPE)

target_sources(op_host_aclnnInner PRIVATE
        fused_infer_attention_score_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_fused_infer_attention_score.cpp
        aclnn_fused_infer_attention_score_v2.cpp
)

target_sources(optiling PRIVATE
        fused_infer_attention_score_tiling.cpp
        fused_infer_attention_score_tiling_register.cpp
        fallback_fused_infer_attention_score.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        fused_infer_attention_score_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_fused_infer_attention_score.h
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_fused_infer_attention_score_v2.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
