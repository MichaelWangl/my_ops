声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# PromptFlashAttention


## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：全量推理场景的FlashAttention算子。

-   计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。
    $$
     Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个n\*d的矩阵。

## 实现原理

图1 计算流程图

![FA图](./fig/PromptFlashAttention.png)

按照flashAttention正向计算流程实现，整体计算流程如下：

1. query与转置后的key做matmul计算后得到最初步的attention_score，然后与位置编码pse相加后再乘以缩放系数scale_value。此时的结果通过atten_mask进行select操作，将atten_mask中为true的位置进行遮蔽，得到结果masked_attention_score，即atten_mask中为true的位置在select后结果为负的极小值，经过softmax计算之后变成0从而达到遮蔽效果。

2. 为了实现FlashAttention加速算法，使用FlashSoftmax操作对masked_attention_score进行运算，用以代替原公式中的softmax运算，而后将结果与value做matmul运算。由于FlashSoftmax操作对masked_attention_score的Skv(输入key、value的sequence length)方向进行了切分，故实现过程中存在一个刷新流程，具体如下：

   1. 每次FlashSoftmax计算只对切分后的一个SkvSplit（SkvSplit是针对Skv轴进行切分之后的序列长度的简称）进行操作，并从第二次循环开始记录exp，其中 i 表示Skv切分后的循环变量，针对exp的i是从1开始 ，exp的计算公式如下：
      $$
      exp[i] = e^{max_{i - 1} - max_{i}}
      $$

   2. 当i = 0时，计算出的MM[PV]结果直接保存到ub_attention_out[0]的ub中。

   3. 从i = 1开始，需要增加Mul和Add操作，即将上一次的MM[PV]的结果和当前exp相乘，相乘完的结果和本次MM[PV]的结果相加得到的结果保存到ub_attention_out[1]的ub中。以此类推，遍历Skv计算完成。

   4. 由于FlashSoftmax计算中的除sum被后移到输出attention_out之前，因此最后需要将ub中的ub_attention_out按行除以softmax_sum并将最终完整的结果保存到输出内存attention_out(Final)上。


## 算子执行接口

算子执行接口为[两段式接口](common/两段式接口.md)，必须先调用“aclnnPromptFlashAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPromptFlashAttention”接口执行计算。

* `aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *pseShift, const aclTensor *attenMask, const aclIntArray *actualSeqLengths, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens, char* inputLayout, int64_t numKeyValueHeads, const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnPromptFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明：**
- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnPromptFlashAttentionGetWorkspaceSize

- **参数说明：**
  -   query（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入Q，数据类型与key的数据类型需满足数据类型推导规则，即保持与key、value的数据类型一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16
      - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  -   key（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入K，数据类型与query的数据类型需满足数据类型推导规则，即保持与query、value的数据类型一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16
      - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  -   value（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入V，数据类型与query的数据类型需满足数据类型推导规则，即保持与query、key的数据类型一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16
      - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  -   pseShift（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型与query的数据类型需满足数据类型推导规则。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。**预留参数，暂未使用**，目前该参数会被强制设置为nullptr。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16
      - Atlas 推理系列加速卡产品：仅支持nullptr

  -   attenMask（aclTensor\*，计算输入）：Device侧的aclTensor，代表下三角全为0上三角全为负无穷的倒三角mask矩阵，不支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。如果不使用该功能可传入nullptr。通常建议shape输入Q_S,KV_S;B,Q_S,KV_S;1,Q_S,KV_S;B,1,Q_S,KV_S;1,1,Q_S,KV_S，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S，对于attenMask的KV_S为非32字节对齐的场景，建议padding到32字节对齐来提高性能，多余部分填充成1。综合约束请见[约束与限制](#约束与限制)。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持BOOL、INT8、UINT8
      - Atlas 推理系列加速卡产品：数据类型仅支持BOOL

  -   actualSeqLengths（aclIntArray\*，计算输入）：Host侧的aclIntArray，代表不同Batch中query的有效Sequence Length。如果不指定seqlen可以传入nullptr，表示和query的shape的S长度相同。限制：该入参中每个batch中的有效Sequence Length应该不大于query中对应batch的Sequence Length。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持INT64
      - Atlas 推理系列加速卡产品：仅支持nullptr
  -   numHeads（int64\_t，计算输入）：Host侧的int，代表query的head个数，数据类型支持INT64。限制：在BNSD/NSD场景下，需要与shape中的query的N轴shape值相同。
  -   scaleValue（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。数据类型与query的数据类型需满足数据类型推导规则。用户不特意指定时可传入默认值1.0。  
  -   preTokens（int64\_t，计算输入）：Host侧的int，用于稀疏计算，表示attention需要和前几个Token计算关联。用户不特意指定时可传入默认值2147483647，支持负数。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持INT64
      - Atlas 推理系列加速卡产品：仅支持默认值2147483647

  -   nextTokens（int64\_t，计算输入）：Host侧的int，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持INT64。用户不特意指定时可传入默认值0，支持负数。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持INT64
      - Atlas 推理系列加速卡产品：仅支持默认值0和2147483647
  -   inputLayout（char\*，计算输入）：Host侧的字符指针CHAR\*，用于标识输入query、key、value的数据排布格式，当前支持BSH、BSND、BNSD、BNSD_BSND(输入为BNSD时，输出格式为BSND)。用户不特意指定时可传入默认值"BSH"。

     **说明：**
     query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

  -   numKeyValueHeads（int64\_t，计算输入）：Host侧的int，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景。用户不特意指定时可传入默认值0，表示key/value和query的head个数相等。限制：需要满足numHeads整除numKeyValueHeads，numHeads与numKeyValueHeads的比值不能大于64，且在BSND、BNSD、BNSD_BSND场景下，需要与shape中的key/value的N轴shape值相同，否则报错。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持INT64
      - Atlas 推理系列加速卡产品：仅支持默认值0

  -   attentionOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出，[数据格式](common/数据格式.md)支持ND。限制：当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出shape为BSND；其余情况该入参的shape需要与入参query的shape保持一致。
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16、INT8
      - Atlas 推理系列加速卡产品：数据类型支持FLOAT16
  -   workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  -  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  -  返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。
  -  返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
  ```

### aclnnPromptFlashAttention

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnPromptFlashAttentionGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了"可传入nullptr"的入参为空指针时，不进行处理。
-   query，key，value输入，功能使用限制如下：
    -   Atlas A2 训练系列产品/Atlas 800I A2 推理产品：
        -   支持B轴小于等于65536(64k)，输入类型包含INT8时D轴非32对齐或输入类型为FLOAT16或BFLOAT16时D轴非16对齐时，B轴仅支持到128；
        -   支持N轴小于等于256；
        -   S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于： 
              -   B=1,Q_N=20,Q_S=2097152,D=256,KV_N=1,KV_S=2097152;
              -   B=1,Q_N=2,Q_S=20971520,D=256,KV_N=2,KV_S=20971520;
              -   B=20,Q_N=1,Q_S=2097152,D=256,KV_N=1,KV_S=2097152;
              -   B=1,Q_N=10,Q_S=2097152,D=512,KV_N=1,KV_S=2097152。
        -   支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。
    -   Atlas 推理系列加速卡产品： 
        -   支持B轴小于等于128；
        -   支持N轴小于等于256；   
        -   支持S轴小于等于65535(64k), Q_S或KV_S非128对齐，Q_S和KV_S不等长的场景不支持配置atten_mask；
        -   支持D轴小于等于512。

## 算子原型

```c++
REG_OP(PromptFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention)
```
参数解释请参见**算子执行接口**。

## 调用示例

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_prompt_flash_attention](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  示例代码如下（以Atlas A2 训练系列产品/Atlas 800I A2 推理产品为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++

#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"
 
using namespace std;
 
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
 
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}
 
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
 
  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
 
int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
 
  // 创建query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  
  int64_t numKeyValueHeads = numHeads;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用第一段接口
  ret = aclnnPromptFlashAttentionGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, numHeads, scaleValue, 
    preTokens, nextTokens, layerOut, numKeyValueHeads, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用第二段接口
  ret = aclnnPromptFlashAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttention failed. ERROR: %d\n", ret); return ret);
 
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
 
  // 6. 释放资源
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

```
