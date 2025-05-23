# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME AllGatherMatmul
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

set(all_gather_matmul_depends mc2/common 
    PARENT_SCOPE)

target_sources(op_host_aclnnInner PRIVATE
        all_gather_matmul_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_all_gather_matmul.cpp
)

target_sources(optiling PRIVATE
        all_gather_matmul_tiling.cpp
        all_gather_formulaic_tiling.cpp
        fallback_all_gather_matmul.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        all_gather_matmul_tiling.cpp
        all_gather_formulaic_tiling.cpp
        fallback_all_gather_matmul.cpp
    )
endif()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        all_gather_matmul_proto.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_all_gather_matmul.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
