/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_GROUPED_BIAS_ADD_GRAD_H_
#define OP_API_INC_LEVEL0_GROUPED_BIAS_ADD_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* GroupedBiasAddGrad(const aclTensor *gradY, const aclTensor *groupIdxOptional,
                                    int64_t groupIdxType, aclOpExecutor *executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_GROUPED_BIAS_ADD_GRAD_H_