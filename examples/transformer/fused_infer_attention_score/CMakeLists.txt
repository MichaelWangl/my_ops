# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


add_executable(test_fused_infer_attention_score
        test_fused_infer_attention_score.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case
        ACLNN_FUNC "aclnnFusedInferAttentionScore"
)

add_executable(test_fused_infer_attention_score_v2
        test_fused_infer_attention_score_v2.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_gqa
        test_fused_infer_attention_score_v2_pfa_gqa.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_gqa
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_gqa
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_sparse2
        test_fused_infer_attention_score_v2_pfa_sparse2.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_sparse2
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_sparse2
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_pse
        test_fused_infer_attention_score_v2_pfa_pse.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_pse
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_pse
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_lse
        test_fused_infer_attention_score_v2_pfa_lse.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_lse
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_lse
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_innerprecise2
        test_fused_infer_attention_score_v2_pfa_innerprecise2.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_innerprecise2
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_innerprecise2
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_left_padding
        test_fused_infer_attention_score_v2_pfa_left_padding.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_left_padding
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_left_padding
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_page_attention
        test_fused_infer_attention_score_v2_pfa_page_attention.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_page_attention
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_page_attention
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_system_prefix
        test_fused_infer_attention_score_v2_pfa_system_prefix.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_system_prefix
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_system_prefix
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_msd
        test_fused_infer_attention_score_v2_pfa_msd.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_msd
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_msd
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_pfa_tensorlist
        test_fused_infer_attention_score_v2_pfa_tensorlist.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_pfa_tensorlist
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_pfa_tensorlist
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_antiquant
test_fused_infer_attention_score_v2_ifa_antiquant.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_antiquant
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_antiquant
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_leftpad
test_fused_infer_attention_score_v2_ifa_leftpad.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_leftpad
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_leftpad
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_Lse
test_fused_infer_attention_score_v2_ifa_Lse.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_Lse
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_Lse
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_PA
test_fused_infer_attention_score_v2_ifa_PA.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_PA
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_PA
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_postquant
test_fused_infer_attention_score_v2_ifa_postquant.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_postquant
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_postquant
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)

add_executable(test_fused_infer_attention_score_v2_ifa_system_prefix
test_fused_infer_attention_score_v2_ifa_system_prefix.cpp
)
add_execute_example(
        TARGET_NAME test_fused_infer_attention_score_v2_ifa_system_prefix
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_fia_case.sh
        TEST_CASE fia_case_v2_ifa_system_prefix
        ACLNN_FUNC "aclnnFusedInferAttentionScoreV2"
)


install(TARGETS test_fused_infer_attention_score
        test_fused_infer_attention_score_v2
        test_fused_infer_attention_score_v2_ifa_antiquant
        test_fused_infer_attention_score_v2_ifa_leftpad
        test_fused_infer_attention_score_v2_ifa_Lse
        test_fused_infer_attention_score_v2_ifa_PA
        test_fused_infer_attention_score_v2_ifa_postquant
        test_fused_infer_attention_score_v2_ifa_system_prefix
        test_fused_infer_attention_score_v2_pfa_gqa
        test_fused_infer_attention_score_v2_pfa_sparse2
        test_fused_infer_attention_score_v2_pfa_pse
        test_fused_infer_attention_score_v2_pfa_lse
        test_fused_infer_attention_score_v2_pfa_innerprecise2
        test_fused_infer_attention_score_v2_pfa_left_padding
        test_fused_infer_attention_score_v2_pfa_page_attention
        test_fused_infer_attention_score_v2_pfa_system_prefix
        test_fused_infer_attention_score_v2_pfa_msd
        test_fused_infer_attention_score_v2_pfa_tensorlist
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)


target_link_libraries(test_fused_infer_attention_score PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2 PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_gqa PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_sparse2 PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_pse PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_lse PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_innerprecise2 PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_left_padding PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_page_attention PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_system_prefix PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_msd PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_pfa_tensorlist PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_antiquant PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_leftpad PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_Lse PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_PA PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_postquant PRIVATE
    -lc_sec
)
target_link_libraries(test_fused_infer_attention_score_v2_ifa_system_prefix PRIVATE
    -lc_sec
)