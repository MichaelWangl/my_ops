/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_regbase_full_load.cpp
 * \brief
 */

 #include "moe_init_routing_v2_grad_tiling.h"
 #include "register/op_def_registry.h"
 #include "tiling/tiling_templates_registry.h"
 
 namespace optiling {
 constexpr uint64_t TILINGKEY_FULL_LOAD_DROPLESS = 300001;
 constexpr uint64_t  TILINGKEY_FULL_LOAD_DROP_PAD = 300002;
 constexpr uint64_t  TILINGKEY_FULL_LOAD_ACTIVE = 300003;
 const static int64_t MAX_FULL_LOAD_CNT = 10;
 const static int64_t DOUBLE_BUFFER = 2;
 const static int64_t FP16_BF16_SIZE = 2;
 const static int64_t FP32_SIZE = 4;
 const static int64_t UINT16_MAX_VALUE = 65535;
 
 class MoeInitRoutingV2GradRegbaseFullLoad : public MoeInitRoutingV2GradTilingBaseClass {
  public:
   explicit MoeInitRoutingV2GradRegbaseFullLoad(gert::TilingContext* context)
       : MoeInitRoutingV2GradTilingBaseClass(context) { Reset(); }
   ~MoeInitRoutingV2GradRegbaseFullLoad() override = default;
   void Reset(gert::TilingContext* context) override { MoeInitRoutingV2GradTilingBaseClass::Reset(context); Reset(); }
  protected:
   bool IsCapable() override { return true; }
   ge::graphStatus DoOpTiling() override;
   uint64_t GetTilingKey() const override;
   ge::graphStatus PostTiling() override;
   void Reset();
  private:
   void SetTilingShapeInfo();
   void SetTilingSplitCore();
   ge::graphStatus SetTilingFactor();
   int64_t blockDim = 0;
   MoeInitRoutingV2GradRegbaseFullLoadTilingData moeInitRoutingV2GradTilingData;
 };
 
 void MoeInitRoutingV2GradRegbaseFullLoad::Reset() { opName = nullptr; }
 
 ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::DoOpTiling() {
   SetTilingShapeInfo();
   SetTilingSplitCore();
   return SetTilingFactor();
 }
 
 void MoeInitRoutingV2GradRegbaseFullLoad::SetTilingShapeInfo() {
   moeInitRoutingV2GradTilingData.set_h(hiddenSize);
   moeInitRoutingV2GradTilingData.set_n(N);
   moeInitRoutingV2GradTilingData.set_k(topK);
   moeInitRoutingV2GradTilingData.set_activeNum(activeNum);
 }
 
 void MoeInitRoutingV2GradRegbaseFullLoad::SetTilingSplitCore() {
   int64_t nBlockFactor = CeilDiv(N, aivNum);  // 单核处理最大token数
   blockDim = CeilDiv(N, nBlockFactor);        // 实际使用核数
   moeInitRoutingV2GradTilingData.set_blockDim(blockDim);
   moeInitRoutingV2GradTilingData.set_nBlockFactor(nBlockFactor);
 }
 
 ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::SetTilingFactor() {
   int64_t blockSize = GetUbBlockSize(context_);
   int64_t regSize = GetVRegSize(context_) / sizeof(float);
   bool notFloat = (inDtype != ge::DT_FLOAT) ? true : false;
   int64_t typeSize = notFloat ? FP16_BF16_SIZE : FP32_SIZE;
   int64_t ubSizeCanUse = aicoreParams_.ubSize;
   int64_t hiddenSizeAlign = AlignUp(hiddenSize * typeSize, blockSize) / typeSize;
   int64_t rowsSizeForOneToken = (topK * DOUBLE_BUFFER + 1) * typeSize;
   bool canHiddenSizeFullLoad = (rowsSizeForOneToken * hiddenSizeAlign) <= ubSizeCanUse;
   if (canHiddenSizeFullLoad) {
     int64_t tokensCanInUb = ubSizeCanUse / (rowsSizeForOneToken * hiddenSizeAlign);
     hiddenSizeAlign = ClipMax(hiddenSizeAlign, UINT16_MAX_VALUE * regSize);
     tokensCanInUb = ClipMax(tokensCanInUb, UINT16_MAX_VALUE);
     moeInitRoutingV2GradTilingData.set_nUbFactor(tokensCanInUb);
     moeInitRoutingV2GradTilingData.set_hUbFactor(hiddenSizeAlign);
   } else {
     if (topK > MAX_FULL_LOAD_CNT) { return ge::GRAPH_PARAM_INVALID; }
     int64_t tokenHiddenSizeCanInUb = ubSizeCanUse / rowsSizeForOneToken;
     tokenHiddenSizeCanInUb = AlignDown(tokenHiddenSizeCanInUb * typeSize, blockSize) / typeSize;
     tokenHiddenSizeCanInUb = ClipMax(tokenHiddenSizeCanInUb, UINT16_MAX_VALUE * regSize);
     moeInitRoutingV2GradTilingData.set_nUbFactor(1);
     moeInitRoutingV2GradTilingData.set_hUbFactor(tokenHiddenSizeCanInUb);
   }
   return ge::GRAPH_SUCCESS;
 }
 
 uint64_t MoeInitRoutingV2GradRegbaseFullLoad::GetTilingKey() const {
   uint64_t tilingKey = TILINGKEY_FULL_LOAD_DROPLESS;
   if (dropPadMode == 1) { tilingKey = TILINGKEY_FULL_LOAD_DROP_PAD; }
   else if (activeNum > 0) { tilingKey = TILINGKEY_FULL_LOAD_ACTIVE; }
   return tilingKey;
 }
 
 ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::PostTiling() {
   context_->SetBlockDim(blockDim);
   size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
   currentWorkspace[0] = workspaceSize_;
   auto tilingData = context_->GetRawTilingData();
   OPS_LOG_E_IF_NULL(context_, tilingData, return ge::GRAPH_FAILED);
   moeInitRoutingV2GradTilingData.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
   tilingData->SetDataSize(moeInitRoutingV2GradTilingData.GetDataSize());
   return ge::GRAPH_SUCCESS;
 }
 
 REGISTER_TILING_TEMPLATE("MoeInitRoutingV2Grad", MoeInitRoutingV2GradRegbaseFullLoad, 30000);
 }  // namespace optiling