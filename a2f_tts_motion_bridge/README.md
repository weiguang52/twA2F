# MotorStreamService

一个面向机器人口型/表情驱动的流式服务项目。

它做三件事：
1. 接收上游 TTS/大模型输出的流式音频块，以及情绪和强度
2. 在云端完成降采样、A2F/ONNX 增量推理、重定向、电机映射
3. 以 30 Hz 输出电机动作，并可选保存 `.npy` 调试产物

## 目录结构

- `proto/motorstream.proto`：gRPC 接口定义
- `motorstream_service/`：核心算法与会话管理
- `server.py`：gRPC 服务入口
- `demo_from_wav.py`：不用 gRPC，直接拿 wav 模拟 TTS 流输入，生成 `.npy`
- `client_stream_wav.py`：示例 gRPC 客户端，用 wav 模拟上游流式音频
- `tools/build_proto.py`：生成 Python gRPC stub

## 先回答一个关键问题

### 能不能“直接复用 NVIDIA 的双向流方案”？

可以**直接复用它的交互模式**，但不建议**原封不动复用它的 proto/消息结构**。

更准确地说：

- 你可以直接复用 **“单条双向流连接、输入端持续送音频 chunk、输出端持续收时序结果”** 这个方案。
- 但 NVIDIA 官方 `ProcessAudioStream` 的输出是 **动画数据 / blendshape data with time code**，而你的服务输出是 **机器人电机动作**，所以对外接口应该定义你自己的 proto，更贴近业务。
- 如果你的系统内部还要再调用 NVIDIA A2F-3D，则可以在你的服务内部把上游音频转接给 NVIDIA 的双向流接口；但你暴露给 TTS/通信同事的 API，最好仍然是你自己的 `MotorFrame` 语义。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/build_proto.py
```

## 1. 本地直接跑算法链路（不用 gRPC）

```bash
python demo_from_wav.py \
  --input_wav /home/cxm2/data/a2f/stream.wav \
  --model /path/to/your_model.onnx \
  --out_npy /home/cxm2/data/a2f/robot_stream_output.npy \
  --emotion neutral \
  --intensity 1.0
```

这会做：
- 用现有 wav 模拟上游 TTS 连续分片输出
- 每片做 `16k mono` 预处理
- 增量推理
- 重定向到电机动作
- 输出原生节拍 + 30fps 对齐版 `.npy`

## 2. 启动 gRPC 服务

```bash
python server.py \
  --model /path/to/your_model.onnx \
  --host 0.0.0.0 \
  --port 50051 \
  --save_dir /tmp/motorstream_npy
```

## 3. 用示例客户端从 wav 流式推音频

```bash
python client_stream_wav.py \
  --server localhost:50051 \
  --input_wav /home/cxm2/data/a2f/stream.wav \
  --emotion neutral \
  --intensity 1.0
```

## 输入协议建议

建议上游给你的字段：
- `session_id`
- `sample_rate`
- `channels`
- `bits_per_sample`
- `pts_ms`
- `audio_bytes`（PCM16）
- `emotion`
- `intensity`

chunk 长度建议：
- 80 ms 到 200 ms 都可以
- 不需要上游理解你的 `WINDOW/HOP`

## 输出协议建议

建议下游收到：
- `session_id`
- `frame_id`
- `time_code_ms`
- `motor_values[]`

推荐网络打包方式：
- 每个 `MotorFrameBatch` 放 1 到 3 帧
- 端侧按 30 Hz 播放

## `.npy` 内容

保存的 `.npy` 是一个 `dict`，包括：
- `frame_times_native`
- `weights_native`
- `features_native`
- `motor_values_native`
- `frame_times_30fps`
- `weights_30fps`
- `features_30fps`
- `motor_values_30fps`
- `audio_16k_mono`

## 什么时候选“直接复用 NVIDIA 双向流”？

### 适合直接复用“模式”
- 你的上游是流式 TTS
- 你希望一条连接里持续进音频、持续出结果
- 你需要低延时链路

### 不适合原样复用“官方 proto”
- 你的输出不是 blendshape，而是电机动作
- 你还需要 `.npy` 调试落盘
- 你要把情绪、强度、保存路径等业务参数纳入会话

## 工程建议

### 第一阶段
先用本项目自己的 gRPC API 跑通三方联调：
- TTS 同事负责流式推 `AudioChunk`
- 你负责服务端推理和动作生成
- 通信同事负责消费 `MotorFrameBatch`

### 第二阶段
如果最终决定把 NVIDIA A2F-3D 微服务作为内部依赖：
- 仍然保留本项目对外的 gRPC API
- 在服务内部把音频转接给 NVIDIA 的 bidirectional endpoint
- 然后把返回的动画进一步重定向成电机动作

这样外部协作接口不会因为底层实现替换而变动。

PYTHONUNBUFFERED=1 python -u -m a2f_tts_motion_bridge.entrypoints.tts_audio_consumer_web_interactive_test   --host 36.103.236.220   --port 50051   --model /home/cxm2/data/a2f/audio2face-3d-model/network.onnx   --tts_sample_rate 24000   --tts_channels 1   --robot_id test_robot   --web_host 0.0.0.0   --web_port 8765