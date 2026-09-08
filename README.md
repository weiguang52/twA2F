# twA2F · Audio2Face → 机器人表情桥接服务

2026-09-08：P06.1/P06.2 已实现 26 维正确编码、十情绪重定向辅助偏置和批量回归/盲评工具，详见 [情绪协议与回归说明](a2f_tts_motion_bridge/docs/EMOTION_PROTOCOL_P06.md)。

一个面向机器人头部表情/口型驱动的**流式服务**：接收上游 TTS / 大模型推来的 PCM16 音频分片（附带情绪与强度），在服务端用 NVIDIA Audio2Face-3D（Claire v2.3）ONNX 模型做增量推理，把 169 维人脸动画权重逐级重定向为 **13 个归一化的头部机构自由度**（眉毛×4、眼睑×4、嘴角×4、下颌×1），以 **30Hz** 电机帧流式输出，同时落盘 `.npy` 调试产物并提供网页实时预览。

| 关键指标 | 数值 | 说明 |
|---|---|---|
| 模型输出 | 169 维 | 140 皮肤 + 10 舌 + 15 颌 + 4 眼 |
| 推理原生帧率 | 10 fps | 窗口 8320 / 步长 1600 @ 16kHz |
| 电机帧输出 | 30 Hz | 零阶保持上采样 |
| 机构自由度 | 13 | 8 个四连杆垂直模块 + 2×2 嘴角五连杆 + 直接下颌 |
| 模型情绪输入 | 26 维 | 16 隐式隐空间 + 10 显式情绪权重 |

> 📖 详细实现原理、情绪链路实测证据与「表情多元化」改进路线图（P0–P3），见仓库内报告：[`a2f_实现原理与表情多元化改进.html`](./a2f_实现原理与表情多元化改进.html)

---

## 总体架构

```
① TTS / LLM 上游          ② 音频预处理           ③ A2F 流式推理            ④ 重定向映射                ⑤ 输出与消费
VoiceAgent gRPC     →     PCM16→float32     →    8320窗口/1600步长    →    169→ARKit52→13 DOF    →    MotorFrame @30Hz
PCM16分片+meta            混单→重采样16kHz        ONNX→169维权重+RMS        几何+语义混合/门控/平滑       640ms .npy 分段
(emotion, intensity)                                                                                 MJPEG 网页预览
```

架构刻意做了**两层解耦**：上游按自己的节奏推音频（80–200ms 一片，不必理解内部窗口机制）；下游机器人只消费归一化的 13 维 `MotorFrame`。中间换模型、换重定向策略，对外协议都不变。

## 目录结构

| 模块 | 职责 |
|---|---|
| `a2f_tts_motion_bridge/motion_core/` | 核心算法：流式推理引擎（`a2f_stream_infer.py`）、音频预处理、169→ARKit52 混合映射（`a2f169_to_arkit52.py`）、ARKit52→13DOF 重定向（`arkit52_to_motor.py`）、会话管理、录制器、全局参数 |
| `a2f_tts_motion_bridge/entrypoints/` | 服务入口：`tts_action_npy_server.py`（gRPC 服务端，产出 .npy 分段）、`tts_audio_consumer*.py`（消费远端 VoiceAgent 音频本地推理）、`demo_motion_from_wav.py`（免 gRPC 直接跑链路） |
| `a2f_tts_motion_bridge/outputs/` | 滚动 .npy 分段导出器（640ms 一段）与分段时间轴工具 |
| `a2f_tts_motion_bridge/preview/` | OpenCV 绘制的 2D 机器人脸渲染器 + MJPEG 网页实时预览（含字幕交互版），离线渲染 mp4 |
| `a2f_tts_motion_bridge/protocol/` | VoiceAgent gRPC 协议桩（AudioRequest / AudioResponse） |
| `audio2face-3d-model/` | 模型资产：network.onnx、bs_skin.npz（52 个 ARKit pose 基）、model_data.npz、implicit_emo_db.npz（隐式情绪库）、各配置 JSON |
| `emotion_test_npy/` `emotion_test_90hz/` | 5 组情绪样例动作（smile/laugh/angry/sad/surprise × 强度）的 .npy / mp4，及网页查看器 `a2f-npy-viewer.html` |

## 环境与安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r a2f_tts_motion_bridge/requirements.txt
```

依赖要点：`onnxruntime`（自动选择 CUDA/CPU provider）、`grpcio`、`numpy`、`soundfile`、`opencv-python`。

### ⚠️ 大文件说明（需自行获取）

以下两个文件超过 GitHub 100MB 单文件限制，**未包含在仓库中**，需从 NVIDIA NGC 或内网获取后放到 `audio2face-3d-model/`：

- `network.onnx`（约 152MB）—— Audio2Face-3D 回归网络
- `model_data.npz`（约 102MB）—— 模型结构数据

缺失时推理引擎无法启动；重定向层的 ARKit52 映射只依赖 `bs_skin.npz` / `bs_tongue.npz` / `implicit_emo_db.npz`（均已通过 Git LFS 包含）。

## 快速开始

### 1. 本地直接跑算法链路（不用 gRPC）

```bash
python -m a2f_tts_motion_bridge.entrypoints.demo_motion_from_wav \
  --input_wav /path/to/stream.wav \
  --model /path/to/audio2face-3d-model/network.onnx \
  --out_npy /tmp/robot_stream_output.npy \
  --emotion happy --intensity 1.0
```

流程：用现有 wav 模拟上游 TTS 连续分片输出 → 每片做 16k mono 预处理 → 增量推理 → 重定向到电机动作 → 输出原生节拍 + 30fps 对齐版 `.npy`。

### 2. 启动 gRPC 服务（VoiceAgent 协议）

```bash
python -m a2f_tts_motion_bridge.entrypoints.tts_action_npy_server \
  --model /path/to/audio2face-3d-model/network.onnx \
  --host 0.0.0.0 --port 50061 \
  --save_dir ./artifacts
```

### 3. 消费远端 TTS 流并本地推理

```bash
python -m a2f_tts_motion_bridge.entrypoints.tts_audio_consumer \
  --host <tts_host> --port 50051 \
  --model /path/to/audio2face-3d-model/network.onnx \
  --text "你好，很高兴见到你" \
  --emotion happy --intensity 1.2 \
  --tts_sample_rate 24000 --tts_channels 1
```

### 4. 网页实时预览（MJPEG + 字幕交互）

```bash
PYTHONUNBUFFERED=1 python -u -m a2f_tts_motion_bridge.entrypoints.tts_audio_consumer_web_interactive \
  --host <tts_host> --port 50051 \
  --model /path/to/audio2face-3d-model/network.onnx \
  --tts_sample_rate 24000 --tts_channels 1 \
  --robot_id test_robot --web_host 0.0.0.0 --web_port 8765
```

浏览器打开 `http://<host>:8765`：2D 机器人脸实时渲染（`/stream.mjpg`）、状态 JSON（`/state`）、文本输入框触发 TTS 全流程。

## 输入 / 输出协议

### 上游 → 本服务（AudioRequest）

- `session_id` / `request_id` / `robot_id`
- `codec=PCM16`、`sample_rate`、`channels`、`audio_chunk`（PCM16 bytes）、`end_of_stream`
- `meta`：

```python
meta = {
    "emotion": "happy",       # 当前 chunk 的目标情绪，可逐 chunk 切换
    "intensity": "1.2",       # 情绪强度
    "segment_ms": "640",      # 输出分段时长
    "speaker_id": "spk_01",
}
```

- chunk 长度建议 80–200ms，上游**不需要**理解内部窗口机制。

### 本服务 → 下游（AudioResponse / MotorFrame）

- `ACTION` 响应：每 640ms 一段 `.npy bytes`（首段 320ms），内含 30fps 对齐的 169 维权重、ARKit52、7 特征、电机值、audio_rms、speech_gate
- `MotorFrame`：`frame_id` / `time_code_ms` / `motor_values[13]` / `debug_features[7]`，端侧按 30Hz 播放

## 算法链路要点

| 环节 | 位置 | 关键机制 |
|---|---|---|
| 流式推理 | `motion_core/a2f_stream_infer.py` | 滑窗 8320（0.52s）/ 步长 1600（0.1s）；time_code 取窗口中心；窗口中心 ±80ms RMS 作语音门控信号；`finalize()` 补零冲刷尾部 |
| 169→ARKit52 | `motion_core/a2f169_to_arkit52.py` | 双路混合：官方资产投影梯度 NNLS（预计算 G/步长，每帧 20 次矩阵乘）+ 语义兜底（自适应量程归一 → 7 抽象特征 → 规则表）；`max()` 融合 + 分类别 EMA |
| ARKit52→13DOF | `motion_core/arkit52_to_motor.py` | SpeechGate 滞回门控压静音泄漏；线性加权和映射；情绪偏置 `_emotion_bias()`；AsymmetricEMA + RateLimiter 类肌肉平滑（下颌最快、眉毛最慢） |
| 30Hz 重采样 | `motion_core/motion_session.py` | 零阶保持，每原生帧 ≤3 帧 |

## 情绪 / 表情链路现状

情绪通过两条独立路径进入最终表情：

1. **模型条件路径**：业务别名映射到十种原生情绪，标签 one-hot × intensity 写入 ONNX 26 维 `emotion` 的显式块 `[16:26]`；也支持显式十维权重和隐式十六维向量透传；
2. **重定向偏置路径**：`_emotion_bias()` 覆盖全部十种原生情绪；默认辅助幅度为 0.35，尊重模型位移方向和剩余行程，可通过 `emotion_bias_scale=0` 关闭。

已通过直接对 `network.onnx` 做探针实验确认的关键事实（详见报告）：

- 26 维情绪输入 = **前 16 维隐式隐空间 + 后 10 维显式情绪**（amazement / anger / cheekiness / disgust / fear / grief / joy / outofbreath / pain / sadness），全部维度对输出有显著影响；
- 旧版七标签落入隐式块 dim0–6 的错误已在 P06 修复：例如 happy→joy(dim22)、sad→sadness(dim25)，neutral 为零向量；
- 模型自带的 `implicit_emo_db.npz` 自动选样尚未启用；P06 已支持业务侧直接透传十六维隐向量。

## 改进路线图（表情多元化）

| 优先级 | 内容 | 预期效果 |
|---|---|---|
| **P0 / P06** | 已实现：26 维正确编码、十情绪辅助偏置、10×5 视频和 DOF 回归工具 | 十种模型条件可用；实际可辨识率待人工盲评 |
| **P1** | 启用 `implicit_emo_db`：mute_* 质心插值 → 连续/混合情绪空间；情绪时间学（attack/sustain/release 包络、300–500ms 交叉淡入）；混合情绪协议 | 连续情绪、自然过渡 |
| **P2** | 风格/节奏层（笑 4–5Hz 振荡、叹气、抽泣）；生命体征层（Poisson 眨眼、呼吸、聆听姿态）；不对称微表情；数据驱动标定 | 静场不死机、会笑出声 |
| **P3** | LLM 编排情绪输出、韵律感知强度、评测闭环（可辨识度分类器 / 静音泄漏率 / golden 回归集） | 规模化与长期质量 |

**P0 是一切的地基**——只有情绪编码变成显式 10 维 + 隐式 16 维的正确结构，隐向量插值和加权混合才有意义。

## 风险与注意事项

- **硬件上限**：13 DOF 决定表情上限，鼻区/脸颊/唇滚动无机构自由度，disgust/cheekiness 等情绪只能靠现有 DOF 组合近似；
- **延迟**：窗口中心对齐带来固有 ~260ms 滞后 + 首帧需攒满 0.52s 音频；
- **30Hz 零阶保持**：快速表情（惊讶、笑）会丢中间帧，可改线性插值；
- **重采样**：24k→16k 线性插值可能有高频混叠，正式化建议换 `scipy.signal.resample_poly`。

## 参考资料

- [NVIDIA/Audio2Face-3D-Samples](https://github.com/NVIDIA/Audio2Face-3D-Samples)
- [nvidia/Audio2Face-3D-v3.0 Model Card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)
- [NVIDIA NGC: Audio2Face-3D](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/audio2face-3d)
- [Audio2Face-3D — Speech-Driven Avatar Motion (Soniqo)](https://soniqo.audio/guides/avatar-motion)
