# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
    OP_NAME MatmulReduceScatter
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

set(matmul_reduce_scatter_depends mc2/common
    PARENT_SCOPE)

target_sources(op_host_aclnnExc PRIVATE
        matmul_reduce_scatter_def.cpp
)

target_sources(opapi PRIVATE
    aclnn_matmul_reduce_scatter.cpp
)

target_include_directories(opapi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/inc
)


if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        aclnn_matmul_reduce_scatter.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        aclnn_matmul_reduce_scatter.cpp
    )
endif ()

target_sources(optiling PRIVATE
    matmul_reduce_scatter_tiling.cpp
    reduce_scatter_formulaic_tiling.cpp
    fallback_matmul_reduce_scatter.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/inc
)

target_sources(opsproto PRIVATE
        matmul_reduce_scatter_proto.cpp
)

file(GLOB _matmul_reduce_scatter_aclnn_header "aclnn*matmul_reduce_scatter*.h")
install(FILES ${_matmul_reduce_scatter_aclnn_header}
    DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)