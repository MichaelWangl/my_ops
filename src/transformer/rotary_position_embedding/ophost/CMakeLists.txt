# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME RotaryPositionEmbedding
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(optiling PRIVATE
        rotary_position_embedding_tiling.cpp
        rope_rotate_half_tiling.cpp
        rope_interleaved_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
        target_sources(opmaster_ct PRIVATE
                rotary_position_embedding_tiling.cpp
                rope_rotate_half_tiling.cpp
                rope_interleaved_tiling.cpp
        )
endif()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        rotary_position_embedding_proto.cpp
)