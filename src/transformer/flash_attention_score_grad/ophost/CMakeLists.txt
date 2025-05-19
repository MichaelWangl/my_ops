# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME FlashAttentionScoreGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        flash_attention_score_grad_def.cpp
)

target_sources(opapi PRIVATE
        flash_attention_score_grad.cpp
        aclnn_flash_attention_score_grad.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            flash_attention_score_grad.cpp
            aclnn_flash_attention_score_grad.cpp
    )
endif ()

target_sources(optiling PRIVATE
        flash_attention_score_grad_tiling.cpp
        flash_attention_score_grad_tiling_bngs1s2_b.cpp
        flash_attention_score_grad_tiling_common.cpp
        flash_attention_score_grad_tiling_ngs1s2_bn.cpp
        flash_attention_score_grad_tiling_s1s2_bn2.cpp
        flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp
        flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp
        flash_attention_score_grad_tiling_unpadded_attension.cpp
)

target_sources(opsproto PRIVATE
        flash_attention_score_grad_proto.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_flash_attention_score_grad.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
