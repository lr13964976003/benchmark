# Ascend NPU LLM Trace Replay Generator

这个工具用于读取：
- `execution trace`（PyTorch execution trace JSON）
- `kineto trace`（Kineto profiler JSON）

并自动生成 **可在昇腾 NPU 环境运行** 的算子流重放程序（Python）。

## 功能

- 解析 execution trace 中的节点信息（算子名、输入输出 shape/type/value）。
- 解析 kineto trace 中的 CPU op 时间（`dur`）。
- 按算子名与顺序对齐 execution/kineto 事件。
- 生成 replay 脚本：
  - 在 `npu:0` 上按算子流顺序执行。
  - 支持常见算子：`view/to/flatten/lt/ge/bitwise_or/__or__/clone/copy_`。
  - 未实现算子自动降级为小矩阵乘负载（保证能持续施压 NPU）。
  - 可按 kineto 的 `dur` 做近似忙等时延模拟。

## 文件

- `trace2replay.py`: 生成器主程序。
- 生成产物示例：`generated_replay.py`（运行时自动产生）。

## 使用方式

```bash
python trace2replay.py \
  --execution-trace execution_trace.json \
  --kineto-trace kineto_trace.json \
  --output generated_replay.py
```

然后在 Ascend 环境执行：

```bash
python generated_replay.py
```

## 环境要求

- Python 3.9+
- PyTorch
- torch-npu（必须）
- Ascend 驱动与 CANN 运行时可用

## 说明

1. execution trace 里的 tensor 元数据并不总是完整，本工具会在缺失时自动构造输入张量。
2. 若算子未被显式支持，重放程序会执行 matmul fallback 来模拟 NPU 负载。
3. 该重放更偏向 **算子流与负载重建**，不是数值正确性回放。
