# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME MoeInitRoutingV2Grad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        moe_init_routing_v2_grad_def.cpp
)


target_sources(optiling PRIVATE
        moe_init_routing_v2_grad_tiling.cpp
        moe_init_routing_v2_grad_regbase_full_load.cpp
        moe_init_routing_v2_grad_regbase_split_h.cpp
        moe_init_routing_v2_grad_regbase.cpp
        moe_init_routing_v2_grad_tiling_base.cpp
)

if (NOT BUILD_OPEN_PROJECT)
        target_sources(opmaster_ct PRIVATE
                moe_init_routing_v2_grad_tiling.cpp
                moe_init_routing_v2_grad_regbase_full_load.cpp
                moe_init_routing_v2_grad_regbase_split_h.cpp
                moe_init_routing_v2_grad_regbase.cpp
                moe_init_routing_v2_grad_tiling_base.cpp
        )
endif()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../moe_init_routing_v2_grad/ophost
)

target_sources(opsproto PRIVATE
        moe_init_routing_v2_grad_proto.cpp
)