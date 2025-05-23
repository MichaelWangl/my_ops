# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_Target ${UTest_NamePrefix}_Stubs)

add_library(${_Target}_Wno INTERFACE)
target_compile_options(${_Target}_Wno
        INTERFACE
            -Wno-unused-parameter
            -Werror
)

set(_Sources
        acl/aclrt_stubs.cpp
        platform/platform_stubs.cpp
        runtime/runtime_stubs.cpp
)
add_library(${_Target} SHARED)
target_sources(${_Target} PRIVATE ${_Sources})
target_link_libraries(${_Target}
        PRIVATE
            $<BUILD_INTERFACE:intf_pub_utest>
            $<BUILD_INTERFACE:${_Target}_Wno>
            ${UTest_NamePrefix}_Utils
)
