# 独立提示词表情（v1）

此路径与 A2F 并行，不要求音频、不加载 A2F/ONNX、不建立 MotorStreamSession。
提示词 → 本地 Qwen3（不加载 KeyframeFace LoRA）→ 严格动作计划 → 连续13DOF → 现有 NPY 分段导出器。
动作轨迹是程序化软件姿态，不是从文本生成的演员动作，不保证任意提示词都能正确理解。

## 启动

在仓库根目录，使用已安装 transformers、量化依赖和 grpc 的环境：

```bash
python -m a2f_tts_motion_bridge.entrypoints.prompt_face_server \
  --base /home/cxm2/data/a2f-textface-work/models/base \
  --device cuda:1 --listen 127.0.0.1:50062
```

省略 `--base` 启动 CPU 计划执行模式，无需 torch、模型权重或音频。
默认仅监听回环地址；没有鉴权/TLS，不应直接暴露公网。
Qwen模型在启动时加载一次，解析请求串行；并非实时动作生成器。

## 输入

使用现有 VoiceAgent.StreamSession，但连接独立端口。
每次调用仅发送一个 AudioRequest：request_id 使用1–64位ASCII字母/数字/下划线/短横线，audio_chunk为空，end_of_stream=true。
meta中 `prompt` 或 `face_plan` 二选一；可选 `export_fps`（30/90）、`segment_ms`（默认640）。meta值全部是字符串。
不支持逐chunk重新编排；若需要中途停止，应取消 RPC 并由下游停止已缓冲动作、平滑回中，不能仅依赖断开网络停止硬件。

```json
{"actions":[{"action":"smile","side":"both","strength":0.25,"tone":"warm","attack_ms":500,"hold_ms":1000,"release_ms":700,"offset_ms":0}],"unsupported":false}
```

支持 smile、raise_brow、frown、sad_brow、blink、wink、close_eye、open_eye、squint。
左右为机器人自身方向。wink必须单侧。单计划最多8动作，同通道后开始动作优先，不恢复已经被覆盖的旧动作。
轻微/稍微默认0.25，普通0.6，明显/很0.85；眨眼默认1。明确数值优先。默认眨眼80ms闭合、40ms保持、140ms打开。
tone枚举 neutral/warm/sad/angry/playful：温柔/伤感影响解析时默认起落时间；微笑的warm增加轻微眯眼，sad增加内高外低眉形，playful左右嘴角不对称；伤感皱眉改为内高外低，愤怒保留下压。其余组合当前没有额外姿态偏置，不是完整情绪模型。
无自动背景眨眼/呼吸，避免污染明确的局部动作；眨眼需在指令/计划中明确安排。
不支持头部、视线、语音、任意新动作；不能以静默预设替代失败模型结果。

## 输出与执行边界

直接复用 ActionSegment、build_payload_from_recorder、payload_to_npy_bytes 和 AudioResponse.action_npz（名字虽是npz，内容仍是现有NPY）。ACTION响应的model_version沿用 motorstream 分段字段；最后单独发送 end_of_response=true。
motor_names严格使用 MOTOR_CFG顺序，13维float32、0..1、同中性位；frame_times_30fps/motor_values_30fps命名保持兼容，实际帧率读取meta.export_fps，90Hz仍沿用原字段名。分段帧数/边界沿用原导出器规则。
释放后追加100ms中性尾部，避免半开时间轴遗漏最终回中姿态。
音频为空，ARKit/模型权重/语音门控等辅助字段为零，不可误作模型推理结果；meta.motion_source=standalone_prompt_face、synthetic_auxiliary_channels=true标记来源。
只兼容DOF消费者，不声称合成的辅助字段与A2F语义等价。共享下游机器人标定/安全限幅；此模块不把软件值转换为硬件角度。
上游必须互斥选择A2F或提示词表情源，不允许两个源同时直接控制同一电机。当前没有跨服务仲裁器、机器人实测或真人感盲评结论。

## 验证

```bash
python -m unittest a2f_tts_motion_bridge.tests.test_standalone_face -v
python -m a2f_tts_motion_bridge.preview.evaluate_prompt_face --base /path/to/base --output /new/artifact/directory
```

单测覆盖无需推理依赖、9动作/5风格/左右组合、强度、单眼闭合、回中、30/90Hz、协议序列化和无效请求。
真实模型评估保留原始输出，语义匹配与schema合法性分开计数；少量固定用例不代表开放提示词准确率。

### 2026-10-23 验证记录

- 全套回归165项：155通过、10按环境条件跳过；含9项独立路径测试及真实回环gRPC传输。
- 首轮真实Qwen3测试10条：schema 10/10，预期语义9/10；“愤怒地皱眉”被错误提高强度至0.85。
- 补充“情感与程度独立”规则后，同10条复测10/10。此为已知用例回归，不是独立泛化评估。
- 简单动作解析约4–5秒，不能宣称实时理解。原始计划和分段NPY位于远端 `artifacts/standalone_prompt_face/smoke_v1`、`smoke_v2`。
- 尚未做机器人实测、视觉盲评、开放提示词大样本测试或双源仲裁验证。
