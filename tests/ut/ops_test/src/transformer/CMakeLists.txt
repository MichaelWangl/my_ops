# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(flash_attention_score_alias flash_attention)
set(flash_attention_score_grad_alias flash_attention)
set(ffn_alias ffn)
set(grouped_matmul_alias grouped_matmul)
set(incre_flash_attention incre_flash_attention)
set(prompt_flash_attention prompt_flash_attention)
set(fused_infer_attention_score fused_infer_attention_score)
set(moe_init_routing_v2_grad moe_init_routing_v2_grad)
set(moe_compute_expert_tokens moe_compute_expert_tokens)
set(moe_finalize_routing_v2_grad moe_finalize_routing_v2_grad)
OpsTestUt_AddSubdirectory()
