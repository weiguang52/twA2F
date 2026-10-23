# 文本眉眼动作：KeyframeFace + A2F 分区混合

## 选择与边界

使用 [KeyframeFace 官方代码/说明](https://github.com/wjc12345123/KeyframeFace)
及 [官方 LoRA](https://modelscope.cn/models/chaochaooooooo/KeyframeFace)，不是用规则词典冒充模型。
模型为 Qwen3-14B + 面部关键帧 LoRA，输出 52 个 ARKit blendshape 与 9 个头/眼姿态参数。
本接入验证全部 52 通道，只将眉毛、眼睑投影到现有 13 DOF 中的 8 个上脸通道。
头姿、眼球朝向没有对应执行器，不发送；嘴部与下颌始终沿用 A2F/已有行为层。

为适配单张 4090 与磁盘空间，远端使用兼容的
[unsloth/Qwen3-14B-bnb-4bit](https://modelscope.cn/models/unsloth/Qwen3-14B-bnb-4bit)
量化基座，搭配官方 LoRA，**不是原精度论文复现**。
Express4D 也进行了核查，但其官方采样依赖需申请的数据集归一化统计，权重下载也未连通；
因此没有用伪造均值/方差或未训练网络接入。

这是“文本 → 语义姿态关键帧 → 平滑插值”，不是逐帧演员微表情生成器。
单条提示产生一个姿态；多个动作的顺序/时间由调用方提供关键帧时间戳。
不保证任意中文/复杂指令准确执行，建议先用明确英文描述检查结果。
ARKit 单一 browInnerUp 不能表达独立左右内眉抬升；13 DOF 和现有投影也会损失细节。
本版使用默认中性、无情绪偏置的 ARKit 投影，不自动套用个体机器人标定文件。
生产部署前必须人工验收动作和真机限位，不能凭 JSON 合法就宣称动作正确。

## 安装与编译（全部在远端 Linux）

依赖与音频服务隔离。沿用已有 PyTorch/CUDA 环境创建独立 venv，然后安装
`requirements-text-face.txt`。普通音频服务不需要 transformers/peft，也不会加载文本模型。
下载官方适配器和兼容量化基座至仓库外，不提交权重。运行时只读本地 safetensors，
禁用远程代码，不会在音频 chunk 回调中下载模型或调用 LLM。

在项目根目录执行，例如：

```bash
TEXT_PY=/home/cxm2/data/a2f-textface-work/venv/bin/python
BASE=/home/cxm2/data/a2f-textface-work/models/base
ADAPTER=/home/cxm2/data/a2f-textface-work/models/adapter
export A2F_TEXT_FACE_DIR=/home/cxm2/data/a2f/artifacts/text_face/clips
"$TEXT_PY" a2f_tts_motion_bridge/entrypoints/compile_text_face.py \
  --base "$BASE" --adapter "$ADAPTER" --device cuda:1 \
  --prompt 'The person frowns, pulling both eyebrows down and inward.' \
  --duration 5 --output "$A2F_TEXT_FACE_DIR/frown_v1.json"
```

输出文件不覆盖已有文件。每个片段包含完整 52 通道、模型/量化信息、适配器 SHA256、
原提示与原响应、生成耗时。模型缺失、生成截断、缺少通道、数值非有限或越界均报错，
不会静默补零或退回手工动作。

`--script script.json` 替代 `--prompt` 支持显式时间编排：

```json
{
  "duration_s": 6,
  "keyframes": [
    {"time_s": 0, "description": "The person frowns, lowering both eyebrows."},
    {"time_s": 3, "description": "The person raises both eyebrows in surprise."}
  ]
}
```

最多 8 个提示关键帧、60 秒，首帧必须为 0，时间严格递增。
当前编译为离线操作，14B 自回归输出完整参数有显著延迟，不承诺实时自然语言响应。

## 流式 meta

服务进程也需设置 `A2F_TEXT_FACE_DIR` 为操作方受控目录。
客户端只能给简短资产 ID，不能提交任意文件路径或执行提示词：

```json
{
  "text_face_clip": "frown_v1",
  "text_face_region": "brows",
  "text_face_weight": "1.0",
  "text_face_fade_ms": "250",
  "auto_blink": "true"
}
```

字段：

| 字段 | 取值/默认 | 含义 |
|---|---|---|
| text_face_clip | 操作方资产 ID / `none` | 从本 chunk 起播放或淡出取消 |
| text_face_region | `brows` / `eyes` / `upper_face`（默认） | 仅控制选定区域 |
| text_face_weight | 0–1，默认 1 | 0 为原路径，1 为文本姿态完全接管选定区域 |
| text_face_fade_ms | 50–2000，默认 250 | 切换、取消、片段末尾平滑淡化 |

传入权重/区域等选项时必须同时携带 clip ID。参数完全相同的重复包不重启片段，
即使原片段已经结束；若要重新播放，先发 `none` 再启动，或使用新资产 ID。
改变参数视为新片段，从头播放并交叉淡入。片段到期自动回到原路径，不循环、不复活旧片段。
资产按会话缓存，修改磁盘同名文件不会改变已运行会话；更新动作应使用新 ID。
会话最多 128 次不同调度，单资产最多 1 MiB/64 个数值关键帧。
未知 text_face_* 参数（包括 text_face_prompt）报错，不会被静默忽略。

处理优先级：A2F/情绪重定向 → 原风格/眉毛节奏 → 文本选定区域混合 → 眨眼/闭眼事件。
因此 weight=1 时眉毛情绪偏置不会压过文字指令，眨眼也不会被文本睁眼姿态取消。
需要保留 A2F 眼神时用 `brows`；需要自动眨眼仍须明确开启 `auto_blink=true` 或 listening/idle。
自然眨眼的现有节奏和默认行为均未更改。

实时与 30/90Hz 导出共用会话绝对时间轴计算，不按调用次数积累。
导出元数据 `text_face_schedule` 保存 ID、权重、区域、时间、SHA256 和来源，便于追溯。
重放需保留原资产文件，只有 metadata 不包含全部关键帧。

## 验证

```bash
/home/cxm2/data/conda_envs/a2f_env/bin/python -m unittest \
  a2f_tts_motion_bridge.tests.test_text_face -v
```

合成单元测试与真实模型生成分开：合成样本只验证投影/时钟/混合，不能证明语言理解效果。
测试覆盖完整通道校验、路径限制、原子更新、重复包、取消/切换、30/90Hz 一致性、
嘴部/下颌逐值不变、眨眼最终优先。真实模型运行结果另存 artifacts，不提交大文件。

真实 A2F 对照评估（需要已编译的 clip）：

```bash
/home/cxm2/data/conda_envs/a2f_env/bin/python \
  a2f_tts_motion_bridge/preview/evaluate_text_face.py \
  --model /home/cxm2/data/a2f/audio2face-3d-model/network.onnx \
  --clip-dir "$A2F_TEXT_FACE_DIR" --clip brow_sequence_v1 \
  --wav /home/cxm2/data/a2f/artifacts/p16_20260909/evaluation_audio.wav \
  --output /home/cxm2/data/a2f/artifacts/text_face/evaluation_v1 --render
```

会输出 baseline/baseline_repeat/text 三组 30/90Hz NPY、两个 MP4 与 evaluation.json；
NPY 可加载现有 `preview/a2f_visualizer.html`。
评估脚本对同一份 A2F 输出做配对导出，检查混合后嘴部下颌逐值不变；
另做不带文本的重复完整推理，记录 GPU/模型重复运行数值波动，
独立文本推理的下脸差异不得超过对照误差的两倍加 1e-7（最低容差 1e-6、硬上限 1e-4）。
同时验证真实 gRPC handler 的重复包与取消，
但没有进行人工盲评或真机测试。

### 2026-10-23 远端实测（实验功能，不代表泛化成功率）

- 代码修改前基线与新 fetch 的 GitHub main 均为 `50a6f10ee11d0f9bc9f3e7ce13e77fa689d52d5c`。
- 全套单元测试 90 项：82 通过、8 按环境条件跳过；原有真实 ONNX/协议 3 项另行通过。
- 真实 KeyframeFace 三帧：皱眉 `browDownLeft/Right=0.6`；双眉抬起
  `browInnerUp/browOuterUpLeft/Right=0.2`；单侧挑眉输出全零眉参数，**未执行成功**。
  另用独立、明确的不对称提示复测，仍生成双侧抬眉，不应将单侧控制作为可靠能力。
- 每个关键帧约 73 秒（RTX 4090、当前 Transformers/4bit 配置），适合离线动作库，
  尚不能在低延迟对话中即时生成。当前只检查少量英文指令，未评估中文理解成功率。
- 初次不限定通道名的输出出现 `jawWidth`，被校验拒绝；当前明确列出合法 61 通道。
  合法 JSON 依旧不代表指令执行正确，没有通过手工改系数掩盖模型失败。
- 8 秒同音频对照：30Hz 240 帧、90Hz 720 帧；同输入配对导出的下脸最大差为 0。
  独立完整推理下脸最大差约 5.1e-6，不带文本的重复推理对照约 8.9e-6。
  上脸最大变化约 0.349；真实 gRPC 14 个分段，重复 clip 未重启，取消后仅保留预期两次调度。
- 实测文件位于远端 `/home/cxm2/data/a2f/artifacts/text_face/evaluation_v2/`：
  `baseline.mp4`、`text.mp4`、六个 NPY、`evaluation.json`。
  模型生成的片段在相邻 `clips/`，包含原响应与来源信息。文件不是公开下载链接。
  `evaluation_v1/` 保留首次严格独立推理比较未通过的诊断产物，没有覆盖它。

上游项目采用 MIT；其系统提示采用官方推理格式，见 `KEYFRAMEFACE_NOTICE.txt`。
基座和模型权重的使用还须遵守各自模型卡条款。
