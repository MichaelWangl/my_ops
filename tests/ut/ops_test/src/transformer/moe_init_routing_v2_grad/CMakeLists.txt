# # Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
# # This file is a part of the CANN Open Software.
# # Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# # Please refer to the License for details. You may not use this file except in compliance with the License.
# # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# # See LICENSE in the root of the software repository for the full text of the License.
# # ======================================================================================================================


# ########################################################################################################################
# # 调用编译方法, 生成对应编译目标
# ########################################################################################################################

# set(_MOE_INIT_ROUTING_V2_GRAD_TilingSourcesExt
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling.cpp
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling_base.cpp
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase_full_load.cpp
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase_split_h.cpp
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase.cpp
# )

# set(_MOE_INIT_ROUTING_V2_GRAD_KernelTilingDataDefH
#         ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
#         ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling.h
# )

# set(_MOE_INIT_ROUTING_V2_GRAD_KernelTilingExtInclude
#         ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/
# )

# set(_MOE_INIT_ROUTING_V2_GRAD_OpProtoSourceExt
#         ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_proto.cpp
# )

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_MIRG_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase_full_load.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase_split_h.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_regbase.cpp
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling_base.cpp
)

set(_MIRG_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_tiling.h
)

set(_MIRG_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/
)

set(_MIRG_OpProtoSourceExt
        ${OPS_ADV_DIR}/src/transformer/moe_init_routing_v2_grad/ophost/moe_init_routing_v2_grad_proto.cpp
)

set(_MIRG_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
set(_MOE_INIT_ROUTING_V2_GRAD_CompileDefintions
        DTYPE_GRAD_EXPANDED_X=float
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/moe_init_routing_v2_grad _MIRG_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MIRG
        SNAKE                                   moe_init_routing_v2_grad
        PROTO_SOURCES_EXT                       ${_MIRG_OpProtoSourceExt}
        TILING_SOURCES_EXT                      ${_MIRG_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MIRG_KernelTilingExtInclude}
        KERNEL_SOURCES_EXT                      ${_MIRG_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_MIRG_KernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_INIT_ROUTING_V2_GRAD_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MIRG_TargetPrivateLinkLibrariesExt}
)