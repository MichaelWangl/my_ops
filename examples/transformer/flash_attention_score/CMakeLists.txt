# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_flash_attention_score
        test_flash_attention_score.cpp
)
add_execute_example(
        TARGET_NAME test_flash_attention_score
        SCRIPT ${EXAMPLES_TRANSFORMER_SCRIPTS}/run_fa_case.sh
        TEST_CASE fa_case
        ACLNN_FUNC "aclnnFlashAttentionScore"
)

add_executable(test_flash_attention_varLen_score
        test_flash_attention_varLen_score.cpp
)
add_execute_example(
        TARGET_NAME test_flash_attention_varLen_score
        SCRIPT ${EXAMPLES_TRANSFORMER_SCRIPTS}/run_fa_case.sh
        TEST_CASE fa_varlen_case
        ACLNN_FUNC "aclnnFlashAttentionVarLenScore"
)

install(TARGETS test_flash_attention_score test_flash_attention_varLen_score
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)
