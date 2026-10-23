# 结构化局部动作 v1

目的：把明确的左右眉眼指令与泛化表情生成分开。自然语言模型只生成结构化计划；
程序执行器负责确定的左右侧、强度与时间，不将程序化动作冒充 KeyframeFace 输出。
默认不开启，不改原有 A2F 音频模型、口型、眨眼节奏或运动平滑参数。

## 架构与优先级

文字 → 可选离线 Qwen3 解析 → 严格校验的动作计划 → local_actions meta → 会话动作轨道。
上游也可直接生成符合协议的 JSON，完全跳过解析模型。

从低到高：A2F/情绪/风格/眉毛节奏 → KeyframeFace 选定区域 → 结构化局部动作 → 眨眼/闭眼事件。
局部动作只改变自身目标通道，不改另一侧、不改嘴角和下颌。
例如 open_eye 不阻止自然眨眼；close_eye 可保持闭眼。
`auto_blink` 默认值未更改，需要业务显式开启。

本版支持 `raise_brow`、`frown`、`close_eye`、`open_eye`、`squint`；
`side` 为 `left/right/both`，**以机器人自身解剖方向为准**。
不支持头部倾斜、眼球朝向或嘴部控制，模型解析遇到这些请求应拒绝整条计划。
不支持的自然语言也可能被模型错误解析为支持动作，因此仍需人工审核或受限指令入口。

## 流式协议

`meta.local_actions` 接受 JSON 数组（Python 内部也接受 list），每包最多 32 项：

```json
{
  "local_actions": [
    {
      "id": "left-brow-001",
      "action": "raise_brow",
      "side": "left",
      "strength": 0.6,
      "attack_ms": 200,
      "hold_ms": 1000,
      "release_ms": 300,
      "offset_ms": 0
    }
  ]
}
```

gRPC 的 map<string,string> 中需对数组做 `json.dumps`，不是把整个 meta 再编码一次。
`id` 必填，1–64 位 ASCII 字母、数字、下划线或连字符。其余数值默认与示例相同；side 默认 both。
字段范围：strength 0–1，attack 50–2000ms，hold 0–10000ms，release 50–3000ms，offset 0–60000ms。
hold 从 attack 完成后开始，片段总长 = attack + hold + release。
所有时间沿用会话音频采样时间轴；offset 相对收到该命令时的时间，不以墙钟或回调次数计时。
`tick_idle(time_s)` 可按主机的会话时钟无音频采样；不是新建后台计时线程。

每个动作是相对于中性姿态的**绝对目标**，不是在当前姿态上不断累加。
strength=0 是无操作，不用于恢复中性；需要结束动作请取消。
强度值为软件归一化控制量，不是电机角度或经过真机标定的生理强度。

### 同区域冲突和取消

- 同一通道由最新开始的动作接管；相同开始时间按到达顺序决定。
- 新动作 attack 从旧动作当时姿态平滑过渡。新动作结束后回到下层路径，旧动作不复活。
- 不同区域可并行。取消尚未开始的动作不会让它在未来抢占区域。
- 相同 id + 相同参数的重复包不重新开始；相同 id + 不同参数报错。
- 改变动作或重新播放必须使用新 id。每会话最多 256 个动作和 256 个取消请求。
- 修改包先完整验证，再一起提交到行为层；错误的局部动作不会顺带触发 style 等其它改变。

```json
{
  "local_action_cancels": [
    {"id": "cancel-001", "targets": ["left-brow-001"], "release_ms": 200}
  ]
}
```

targets 也可为字符串 `all`，只取消该请求首次处理时已经登记的动作（含等待执行的动作）。
重复发送同一个取消 id 不会取消后来新加入的动作；若要再次全取消，使用新取消 id。
同一动作第一次取消生效，后续取消不重置它的释放计时。取消未知动作 id 报错。
取消在动作进行中时，从当时姿态平滑回到下层路径；提前取消则不开始动作。

## 自然语言解析（可选、离线）

使用已有独立文本环境与本地 Qwen3 基座，**不加载 KeyframeFace LoRA**，不使用关键词字典充当模型理解。
复用 `requirements-text-face.txt`。模型解析不进入实时音频 chunk 回调，不调用外部服务。

```bash
cd /home/cxm2/data/a2f
/home/cxm2/data/a2f-textface-work/venv/bin/python \
  a2f_tts_motion_bridge/entrypoints/compile_local_actions.py \
  --base /home/cxm2/data/a2f-textface-work/models/base --device cuda:1 \
  --text '机器人左眉抬高，强度0.6，从现在开始，起势200毫秒，保持1秒，释放300毫秒。' \
  --request-id demo-left-brow \
  --output /home/cxm2/data/a2f/artifacts/local_actions/left-brow-plan.json
```

生成文件含原指令、模型原始响应、验证后的计划、可直接传输的 meta、耗时与模型来源。
输出存在时拒绝覆盖；模型拒绝请求或生成非法结构时不生成可执行 meta，不静默修复。
文本长度最多 2000 字符，每个计划最多 8 个动作。使用新 request-id 重播计划。

解析约定：模型只输出 actions/cancel_all/unsupported；校验确保结构和范围合法，
**不保证语义正确**。同一条格式正确的指令仍可能被模型理解错。

## 50 条验收

`preview/local_action_cases.py` 存放 50 条固定中文指令及预先定义的期望计划：
5 动作 × 3 侧 × 3 强度 = 45 条，加定时双动作、全取消、头部拒绝、嘴部拒绝、指定保持时长 5 条。
这是一套明确模板的验收集，不是未知中文表达、否定句、歧义或多轮对话的泛化评测。

两个结果必须分开报告：

1. **神经模型解析**：真实 Qwen3 每条输出与期望计划的所有字段完全比较；保留原响应，不纠正失败。
2. **确定性执行**：直接给期望计划，以独立公式按 90Hz 验证目标、包络和非目标区域。
   额外回归覆盖接管、取消、原子更新、幂等、优先级及 30/90Hz 一致性。

```bash
/home/cxm2/data/conda_envs/a2f_env/bin/python -m unittest \
  a2f_tts_motion_bridge.tests.test_local_actions -v

A2F_TEST_MODEL=/home/cxm2/data/a2f/audio2face-3d-model/network.onnx \
  /home/cxm2/data/conda_envs/a2f_env/bin/python -m unittest \
  a2f_tts_motion_bridge.tests.test_local_actions_integration -v
```

模型评测：`preview/benchmark_local_action_planner.py --base ... --output 新目录`。
执行归档：`preview/evaluate_local_actions.py --output 新目录 --render --planner-results 模型结果/cases.jsonl`。
会生成 47 个动作 NPY、7 个代表性程序化视频与 HTML 索引；取消和不支持请求另作断言，不伪造对应模型视频。
真实 A2F 集成测试对同一记录做有/无局部轨道的配对导出，验证非目标 DOF 逐值不变；另测 gRPC 重复与取消。

2026-10-23 实测：本地 4bit Qwen3-14B、无 KeyframeFace LoRA，50/50 结构合法，50/50 期望计划完全匹配；
平均 4.5128 秒、P95 4.8282 秒。仍为离线预生成，不承诺实时自由提示词。
原始结果：`artifacts/local_actions/planner_50_v1/`。未做真机标定或人工盲评。

下一阶段应增加自然改写、否定/纠正、混合左右、边界时间和超范围拒绝的独立测试集，再讨论模型微调。
