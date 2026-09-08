# VoiceAgent 联调说明

P06 情绪参数更新：支持 10 种原生情绪、10 维显式权重和 16 维隐向量；meta 字段、覆盖/清空规则及回归流程见 [EMOTION_PROTOCOL_P06.md](EMOTION_PROTOCOL_P06.md)。

## 你现在有两条并行接口

### 1. 原生实时 motor stream（保留原能力）
- 服务：`server.py`
- 协议：`motorstream.v1.MotorStreamService/ProcessAudioToMotorStream`
- 输入：`SessionHeader + AudioChunk + EmotionUpdate + EndOfAudio`
- 输出：实时 `MotorFrameBatch`
- 适合：机器人控制链路 / 实时驱动

### 2. VoiceAgent 兼容接入（新加联调能力）
- 服务：`motorstream_service.voiceagent_compat_server`
- 协议：`robotic.voice.VoiceAgent/StreamSession`
- 输入：PCM16 `audio_chunk`
- 输出：每 640 ms 一个 `ACTION.action_npz`，实际内容是 `.npy bytes`
- 适合：先和 TTS/LLM 侧打通

## 服务端 / 客户端分布

### 服务端（你这边）
建议同时跑两个服务：

```bash
# 1) 保留原生实时帧服务
cd /home/cxm2/data/a2f/motorstream_project
python server.py \
  --model /home/cxm2/data/a2f/audio2face-3d-model/network.onnx \
  --host 0.0.0.0 \
  --port 50051 \
  --save_dir ./artifacts
```

```bash
# 2) 新增 VoiceAgent 兼容服务
cd /home/cxm2/data/a2f/motorstream_project
python -m motorstream_service.voiceagent_compat_server \
  --model /home/cxm2/data/a2f/audio2face-3d-model/network.onnx \
  --host 0.0.0.0 \
  --port 50061 \
  --segment_ms 640 \
  --save_dir ./artifacts
```

### 客户端（TTS / LLM / 总控那边）
- 如果他们要**先快速联调**：连 `50061`，走 `VoiceAgent.StreamSession`
- 如果他们要**真正实时驱动 motor**：连 `50051`，走 `MotorStreamService`

## 数据如何传

### TTS 传给你的内容
TTS 侧必须发 **PCM16 音频 chunk**，不要只发 text。

`AudioRequest` 关键字段：
- `session_id`
- `request_id`
- `codec=PCM16`
- `sample_rate`
- `channels`
- `audio_chunk`
- `end_of_stream`
- `meta`

### meta 里建议这样发

```python
meta = {
    "emotion": "happy",
    "intensity": "1.2",
    "segment_ms": "640",
    "speaker_id": "spk_01",
}
```

说明：
- `emotion`：当前 chunk 对应的目标情绪
- `intensity`：当前 chunk 对应的情绪强度
- 默认首段为 0–320 ms，第二段为 320–640 ms，此后按 640 ms 分段。服务端 `--segment_ms` 配置常规周期，`--first_segment_ms` 配置首段；请求 meta 的 `segment_ms` 不覆盖服务端周期，正整数 `first_segment_ms` 可覆盖首段（最大不超过常规周期）。
- 如果情绪中途切换，后续 chunk 改这个 meta 即可
- 兼容服务会按 chunk 读取最新 emotion/intensity

### 你回给他们的内容
你回的是 `AudioResponse(type=ACTION)`：
- `action_npz`: **实际是 `.npy bytes`**
- `action_description`: 这一段的起止时间、segment index、是否 final
- `model_version`: 附带 segment 元信息

客户端收到后直接：

```python
with open("seg_000.npy", "wb") as f:
    f.write(resp.action_npz)
```

## 640 ms 打包规则

- 每累计 640 ms 输入音频，服务端产出 1 个 `.npy bytes`
- 结尾不足 640 ms 的尾段，也会再发 1 个 final segment
- 这样总控侧可以按 segment 连续消费

## 立刻测试

### 用 wav 模拟 TTS 推流

```bash
cd /home/cxm2/data/a2f/motorstream_project
python -m motorstream_service.voiceagent_audio_push_client \
  /path/to/test_pcm16.wav \
  --host 127.0.0.1 \
  --port 50061 \
  --emotion happy \
  --intensity 1.2
```

## 重要约束

1. `VoiceAgent` 兼容服务当前只支持 `PCM16`
2. 当前 `ACTION.action_npz` 字段名虽然叫 `npz`，但里面实际放的是 `.npy bytes`
3. 原生实时 `MotorFrameBatch` 能力没有删除，也没有被替换
4. 如果对方只有 text、没有音频，这个兼容服务无法直接推理，必须让 TTS 结果音频继续往下转发给你
