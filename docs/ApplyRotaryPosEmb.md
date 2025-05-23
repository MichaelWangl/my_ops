声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ApplyRotaryPosEmb

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能描述

-  **算子功能**：推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。

-  **计算公式**：

  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$
  
  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$
  
  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnApplyRotaryPosEmbGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnApplyRotaryPosEmb”接口执行计算。

* `aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(aclTensor *queryRef, aclTensor *keyRef, const aclTensor *cos, const aclTensor *sin, int64_t layout, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnApplyRotaryPosEmb(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明：**
- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnApplyRotaryPosEmbGetWorkspaceSize

- **参数说明：**

  - queryRef（aclTensor\*，计算输入）：表示要执行旋转位置编码的第一个张量，公式中的`query`，Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，维度为4维。计算结果原地更新。
  - keyRef（aclTensor\*，计算输入）：表示要执行旋转位置编码的第二个张量，公式中的`key`，Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，维度为4维。
  - cos（aclTensor*，计算输入）：表示参与计算的位置编码张量，公式中的`cos`，Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，维度为4维。
  - sin（aclTensor*，计算输入）：表示参与计算的位置编码张量，公式中的`sin`，Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，维度为4维。
  - layout（int64_t，计算输入）：表示输入Tensor的布局格式，数据类型int64，目前只支持1，代表格式为BSND的4维Tensor。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1.传入的queryRef、keyRef、cos或sin是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1.传入的queryRef、keyRef、cos、sin的数据类型、数据格式不在支持的范围内或shape不匹配。
                                        2.传入的layout参数不在支持范围内。
  ```

## aclnnApplyRotaryPosEmb

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnApplyRotaryPosEmbGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

  - 输入张量queryRef、keyRef、cos、sin只支持4维的shape，layoutOptional只支持1
  - 输入张量queryRef、keyRef、cos、sin 的dtype必须相同，且4个输入shape的前2维和最后一维必须相等，cos和sin的shape第3维必须等于1，输入shape最后一维必须等于128
  - 输入queryRef的shape用（q_b, q_s, q_n, q_d）表示，keyRef shape用（q_b, q_s, k_n, q_d）表示，cos和sin shape用（q_b, q_s, 1, q_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。

    - 当输入是BFLOAT16时，cast表示为1，castSize为4，DtypeSize为2
    - 当输入是FLOAT16或FLOAT32时，cast表示为0，castSize = DtypeSize（FLOAT16时为2，FLOAT32时为4）

    需要使用的UB空间大小计算方式：`ub_required = (q_n + k_n) * 128 * castSize * 2 + 128 * DtypeSize * 4 + (q_n + k_n) * 128 * castSize + (q_n + k_n) * 128 * castSize * 2 + cast * (128 * 4 * 2)`，
    当计算出`ub_required`的大小超过当前AI处理器的UB空间总大小时，不支持使用该融合算子
  - 不支持空tensor场景

## 算子原型

```c++
REG_OP(ApplyRotaryPosEmb)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(layout, Int, 1)
    .OUTPUT(query,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(key,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(ApplyRotaryPosEmb)
```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> queryShape = {1, 1, 1, 128};
    std::vector<int64_t> keyShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    int64_t layout = 1;

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    std::vector<float> queryHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                        119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                        24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                        34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                        39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                        26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                        33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> keyHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42,
                                      40, 17, 98, 117, 53, 75, 68, 75, 38, 41, 115, 76, 67, 22, 76, 10, 24, 46, 85, 54, 61, 114,
                                      10, 59, 6, 123, 58, 10, 115, 9, 13, 58, 66, 120, 23, 30, 83, 13, 11, 76, 18, 82, 57, 4,
                                      117, 105, 8, 73, 127, 5, 91, 56, 12, 125, 20, 3, 104, 40, 46, 18, 89, 63, 99, 104};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};

    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnApplyRotaryPosEmb第一段接口
    ret = aclnnApplyRotaryPosEmbGetWorkspaceSize(query, key, cos, sin, layout, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnApplyRotaryPosEmb第二段接口
    ret = aclnnApplyRotaryPosEmb(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmb failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %hhu\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(keyShape);
    std::vector<float> resultData1(size, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), keyDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %hhu\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);

    // 7. 释放device 资源
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
