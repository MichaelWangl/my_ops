# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(EXAMPLES_TRANSFORMER_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# 获取 SocName
get_filename_component(_define_py ${OPS_ADV_CMAKE_DIR}/scripts/examples/get_soc_info.py REALPATH)
execute_process(
        COMMAND ${HI_PYTHON} ${_define_py} "-i=soc_name"
        RESULT_VARIABLE _Rst
        OUTPUT_VARIABLE _SocName
        ERROR_QUIET
)
if (_RST)
    message(WARNING "Can't get socName")
else ()
    if ("${_SocName}x" STREQUAL "x")
        set(_SocName "Unknown")
    endif ()
    string(TOLOWER ${_SocName} _SocName)
endif ()
message(STATUS "SocName=${_SocName}")

# 添加示例工程
if (_SocName MATCHES ascend910b*)
    add_subdirectory(all_gather_matmul)
    add_subdirectory(matmul_reduce_scatter)
    add_subdirectory(moe_distribute_dispatch)
else()
    message(STATUS "AllGatherMatmul don't support SocName(${_SocName}), skip.")
    message(STATUS "MatmulReduceScatter don't support SocName(${_SocName}), skip.")
    message(STATUS "MoeDistributeDispatch don't support SocName(${_SocName}), skip.")
endif()