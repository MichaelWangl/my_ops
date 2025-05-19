# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


if (_SocName MATCHES ascend910b*)
        add_executable(ffn_v2_antiquant
                test_ffn_v2.cpp
        )
        add_execute_example(
                TARGET_NAME ffn_v2_antiquant
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_v2_antiquant
                ACLNN_FUNC "aclnnFFNV2"
        )

        add_executable(ffn_v3_quant_token_index_flag
                test_ffn_v3_quant.cpp
        )
        add_execute_example(
                TARGET_NAME ffn_v3_quant_token_index_flag
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_v3_quant_token_index_flag
                ACLNN_FUNC "aclnnFFNV3"
        )

        add_executable(ffn_v3_float16_high_precision
                test_ffn_v3.cpp
        )
        add_execute_example(
                TARGET_NAME ffn_v3_float16_high_precision
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_v3_float16_high_precision
                ACLNN_FUNC "aclnnFFNV3"
        )
        install(TARGETS ffn_v2_antiquant ffn_v3_quant_token_index_flag ffn_v3_float16_high_precision
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib
                RUNTIME DESTINATION bin
                OPTIONAL
        )
else()
        message(STATUS "FFN has no example for SocName(${_SocName}), skip.")
endif()

