# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_incre_flash_attention
        test_incre_flash_attention.cpp
)
add_execute_example(
        TARGET_NAME test_incre_flash_attention
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ifa_case.sh
        TEST_CASE ifa_case
        ACLNN_FUNC "aclnnIncreFlashAttention"
)

add_executable(test_incre_flash_attention_v2
        test_incre_flash_attention_v2.cpp
)
add_execute_example(
        TARGET_NAME test_incre_flash_attention_v2
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ifa_case.sh
        TEST_CASE ifa_case_v2
        ACLNN_FUNC "aclnnIncreFlashAttentionV2"
)

add_executable(test_incre_flash_attention_v3
        test_incre_flash_attention_v3.cpp
)
add_execute_example(
        TARGET_NAME test_incre_flash_attention_v3
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ifa_case.sh
        TEST_CASE ifa_case_v3
        ACLNN_FUNC "aclnnIncreFlashAttentionV3"
)

add_executable(test_incre_flash_attention_v4
        test_incre_flash_attention_v4.cpp
)
add_execute_example(
        TARGET_NAME test_incre_flash_attention_v4
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ifa_case.sh
        TEST_CASE ifa_case_v4
        ACLNN_FUNC "aclnnIncreFlashAttentionV4"
)

install(TARGETS test_incre_flash_attention test_incre_flash_attention_v2 test_incre_flash_attention_v3 test_incre_flash_attention_v4
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)
target_link_libraries(test_incre_flash_attention PRIVATE
    -lc_sec
)
target_link_libraries(test_incre_flash_attention_v2 PRIVATE
    -lc_sec
)
target_link_libraries(test_incre_flash_attention_v3 PRIVATE
    -lc_sec
)
target_link_libraries(test_incre_flash_attention_v4 PRIVATE
    -lc_sec
)