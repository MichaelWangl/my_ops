# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

target_sources(opapi PRIVATE
        matmul_util.cpp
)

target_sources(optiling PRIVATE
        hccl_formulaic_tiling.cpp
        hccl_performance.cpp
        matmul_formulaic_tiling.cpp
        matmul_performance.cpp
        mc2_tiling_utils.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        hccl_formulaic_tiling.cpp
        hccl_performance.cpp
        matmul_formulaic_tiling.cpp
        matmul_performance.cpp
        mc2_tiling_utils.cpp
    )
endif()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)