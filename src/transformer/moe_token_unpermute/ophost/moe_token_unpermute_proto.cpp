/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_token_unpermute_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMoeTokenUnpermute(gert::InferShapeContext *context)
{
    const gert::Shape *permuted_inputs_shape = context->GetInputShape(0);
    const gert::Shape *probs_shape = context->GetInputShape(2);
    int64_t tokens_num;
    if (probs_shape == nullptr) {
        const gert::Shape *indices_shape = context->GetInputShape(1);
        tokens_num = indices_shape->GetDim(0);
    } else {
        tokens_num = probs_shape->GetDim(0);
    }

    gert::Shape *out_shape = context->GetOutputShape(0);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenUnpermute(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermute)
    .InferShape(InferShapeForMoeTokenUnpermute)
    .InferDataType(InferDataTypeForMoeTokenUnpermute);
} // namespace ops
