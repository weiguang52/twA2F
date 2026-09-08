# P06：Claire 26 维情绪接口及回归流程

模型依据：本项目 `audio2face-3d-model/network_info.json` 的 `implicit_emotion_len=16` 和 `explicit_emotions`。中性为全零，不再把 neutral 或其他业务标签误写入隐式块。

## 显式情绪顺序

| 显式数组索引 | 模型绝对维度 | 原生标签 | 业务别名 |
|---|---|---|---|
| 0 | 16 | amazement | surprise, surprised |
| 1 | 17 | anger | angry |
| 2 | 18 | cheekiness | — |
| 3 | 19 | disgust | — |
| 4 | 20 | fear | — |
| 5 | 21 | grief | — |
| 6 | 22 | joy | happy, smile, laugh |
| 7 | 23 | outofbreath | out_of_breath, out-of-breath |
| 8 | 24 | pain | — |
| 9 | 25 | sadness | sad, sorrow |

`neutral` 是额外的零条件，不占这十个显式维度。标签忽略大小写和首尾空白；未知标签返回错误，不再静默当成 neutral。

## VoiceAgent 的 meta（字符串映射，无需改 proto）

```python
meta = {"emotion": "happy", "intensity": "1.25"}
# 实际模型 emotion[0,0,22] = 1.25，其余 25 维为零。
```

支持三个新增键：

- `emotion_explicit_weights`：JSON 数组，长度恰好 10；按上表顺序。非负有限数，乘以 intensity 后写入 `[16:26]`，不归一化、不截断到 1。
- `emotion_implicit_vector`：JSON 数组，长度恰好 16；允许有符号有限数，原样写入 `[:16]`，不乘 intensity。
- `emotion_bias_scale`：非负有限数，默认 `0.35`；`0` 关闭手调 DOF 偏置，保留模型及重定向。所有值还须可表示为 float32。

```python
import json
meta = {
    "emotion": "neutral",
    "intensity": "0.8",
    "emotion_explicit_weights": json.dumps([0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0.3]),
    "emotion_bias_scale": "0.35",
}
# joy=0.56, sadness=0.24；偏置由这两个显式分量混合，neutral 标签不覆盖它们。

meta = {
    "emotion_implicit_vector": json.dumps([0.0] * 16),
    "emotion_explicit_weights": "null",
    "emotion_bias_scale": "0",
}
# 示例零隐向量；实际业务应传经过验证的演员库隐向量。
```

覆盖/持续规则：

1. 未提供向量时，业务标签生成十维 one-hot；neutral 生成全零。
2. 显式数组优先于标签；显式数组和隐向量可同时传入，分别写入两个块。
3. 仅提供隐向量时，显式块全零，并且没有标签手调偏置；intensity=0 只清零显式块，不清除隐向量。
4. 同一会话中，省略字段保持上次设置；字段传 JSON `null` 清除该向量。清除后按剩余控制重新选择：没有隐向量/显式数组时回到标签模式。
5. meta 明确包含 `emotion` 表示重新选择标签，先清除旧向量，再应用同一 meta 中的新向量。仅改变 intensity 不清除向量。
6. `AudioChunk.emotion_meta` 和 `SessionConfig.extra` 支持相同的键。直接 Python 调用中，相同 `AudioChunk.emotion` 重复出现不清除向量；不同标签会清除旧向量。需要显式重置同名标签时使用 `emotion_meta={"emotion": ...}`。
7. 错误长度、非数值、NaN/Inf、负 intensity、负显式权重都会抛出 ValueError；gRPC 返回 ERROR 并结束该流，不使用错误数值继续推理。

参数作用于接收它的音频块所触发的推理窗口，沿用现有窗口时序；本次未加入时间码级情绪调度、过渡包络或隐式库自动选样。

## 两条情绪路径

`_infer_input_array` 将最终 26 维向量送入 ONNX，`OnlineRetargeter` 根据同一控制对象补充 DOF 偏置。新增 disgust/fear/cheekiness/grief/pain/outofbreath 偏置，已有四类支持原生名称及别名。

偏置使用显式向量各分量（已乘 intensity）混合；每个分量的偏置强度上限仍为 1.5，模型显式输入不采用这个上限。默认再乘 `emotion_bias_scale=0.35`。偏置与模型相对中性位置的位移方向相反时不施加该偏置；方向一致时按距边界的剩余空间缩放，再进入原有平滑器。这样不覆盖模型输出，并减少饱和。

13 DOF 没有独立鼻区/脸颊机构，disgust 等使用现有眉、眼睑、嘴角近似；其可辨识性需要盲评验证，不能仅凭编码正确保证。

录制 NPY 和分段 NPY 新增 `emotion_vectors_native`（N×26）及 `emotion_bias_scales_native`（N），记录各原生帧的实际控制。meta 中的 `emotion_vector` 是最新控制快照，`emotion_encoding=claire26_implicit16_explicit10_v1`；分段时应以逐帧记录为准。旧字段和 13 DOF 顺序不变。

## 远端测试与生成基准

```bash
cd /home/cxm2/data/a2f
conda activate /home/cxm2/data/conda_envs/a2f_env
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s a2f_tts_motion_bridge/tests -v
A2F_TEST_MODEL="$PWD/audio2face-3d-model/network.onnx" \
  python -m unittest a2f_tts_motion_bridge.tests.test_emotion_model_integration -v

python -m a2f_tts_motion_bridge.entrypoints.build_emotion_regression \
  --input_wav a2f_offline_motion_pipeline/audio_prep/stream_10s.wav \
  --model audio2face-3d-model/network.onnx \
  --output artifacts/p06_next --workers 3
```

默认十个原生情绪 × `0.25/0.50/0.75/1.00/1.25`，相同 10 秒音频、固定 200 ms 分片，每组使用独立会话状态，共享只读模型运行时/资产。输出目录必须全新，防止覆盖基准。

- `index.html`：带标签的 10×5 视频矩阵；`videos/p06_<emotion>_<intensity>.mp4`。
- `emotion_test_npy/`：原生帧和 30 Hz 插值数据；`emotion_test_90hz/`：90 Hz 版本。兼容旧 key `*_30fps`，实际帧率查看 `meta.export_fps`。
- `profiles.json` / `dof_means.csv`：每个 DOF 的均值、标准差、极值、饱和率；分别保存合成输出、仅模型路径输出及各自去除前 2 秒后的画像。模型路径对照重用同一 A2F 权重，并独立重置重定向状态。
- `manifest.json`：音频、模型资产、代码哈希、provider、强度等复现信息；`video_checks.json` 记录视频帧数、时长、音轨及哈希验证结果。
- 下次传 `--baseline /旧目录/profiles.json --max_mean_delta 0.02` 会写 `comparison.json`，报告合成 DOF 均值差及阈值结果；比较前应核对 manifest 的音频、模型、参数是否一致。

## 人工盲评

只向评审者提供 `blind/` 目录，勿同时提供带标签矩阵或 `evaluation/answer_key.json`。视频内只出现匿名 trial ID，不显示源文件名、情绪或强度；所有组使用同一音频。

1. 打开 `blind/index.html`，每段选一个情绪，填写 `scores_template.csv` 的 `predicted_emotion`（可选 confidence）。
2. 管理者持答案表汇总，每份评分表单独调用：

```bash
python -m a2f_tts_motion_bridge.entrypoints.build_emotion_regression \
  --output artifacts/p06_20260908 --scores /path/to/filled_scores.csv
```

生成 `evaluation/human_scores.json`（总体/逐情绪准确率、混淆矩阵、完成率）。未填写评分时不计算可辨识率；部分评分会明确标记 incomplete。数值画像不是人工可辨识率的替代指标。
