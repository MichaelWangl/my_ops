# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_moe_gating_top_k_softmax
        test_moe_gating_top_k_softmax.cpp
)
add_execute_example(
        TARGET_NAME test_moe_gating_top_k_softmax
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_moe_gating_top_k_softmax_case.sh
        TEST_CASE moe_gating_top_k_softmax_case
        ACLNN_FUNC "aclnnMoeGatingTopKSoftmax"
)

install(TARGETS test_moe_gating_top_k_softmax
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)
target_link_libraries(test_moe_gating_top_k_softmax PRIVATE
    -lc_sec
)