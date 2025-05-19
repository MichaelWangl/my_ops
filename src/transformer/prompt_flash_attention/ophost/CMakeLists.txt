# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME PromptFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
)

add_ops_compile_options(
        OP_NAME PromptFlashAttention
        COMPUTE_UNIT ascend310p1
        OPTIONS -mllvm
                -cce-aicore-jump-expand=true
)
target_sources(op_host_aclnnExc PRIVATE
        prompt_flash_attention_def.cpp
)

target_sources(opapi PRIVATE
        prompt_flash_attention_base_aclnn.cpp
        aclnn_prompt_flash_attention_inner.cpp
        aclnn_prompt_flash_attention.cpp
        aclnn_prompt_flash_attention_v2.cpp
        aclnn_prompt_flash_attention_v3.cpp
)

target_sources(optiling PRIVATE
        prompt_flash_attention_tiling.cpp
        fallback_prompt_flash_attention.cpp
        prompt_flash_attention_tiling_register.cc
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        prompt_flash_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        prompt_flash_attention_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../prompt_flash_attention/ophost
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/prompt_flash_attention.h
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_prompt_flash_attention_inner.h
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_prompt_flash_attention.h
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_prompt_flash_attention_v2.h
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_prompt_flash_attention_v3.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
