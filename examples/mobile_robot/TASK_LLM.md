# 移动小车自然语言与 LLM 接入任务

> 上游输入：用户自然语言。
>
> 下游接口：视觉/控制团队提供的高层技能、语义对象和任务事件。
>
> 视觉与小车控制主任务见 [TASK.md](TASK.md)。

## 1. 职责边界

LLM 模块负责：

```text
用户自然语言
  ↓
意图、目标属性和约束解析
  ↓
结构化 TaskSpec
  ↓
Schema 校验
  ↓
调用视觉控制侧高层技能
  ↓
把结构化状态转换为用户反馈
```

LLM 模块不负责：

- 读取原始 LiDAR 数组并逐帧驾驶；
- 输出左右轮速度或 `linear_velocity/angular_velocity`；
- 生成 A* 路径节点或绕障轨迹；
- 猜测目标世界坐标；
- 替代 YOLO、深度融合、定位或安全停车；
- 绕过视觉控制侧的速度和安全限制。

因此，LLM API 故障不会阻塞视觉控制团队先完成“固定目的地行驶过程中持续识别物体”的当前交付。

## 2. 最终用户故事

用户输入：

```text
行驶到黄色小车附近
```

LLM 模块应输出经校验的高层任务：

```json
{
  "schema_version": "1.0",
  "intent": "navigate_near_object",
  "target": {
    "category": "car",
    "attributes": {
      "color": "yellow"
    },
    "spatial_qualifier": null
  },
  "constraints": {
    "near_distance_m": 0.8,
    "face_target": true,
    "max_speed_mps": 0.5
  },
  "on_ambiguity": "ask_user"
}
```

视觉控制模块负责查找黄色车、定位目标、规划、驾驶和停车。LLM 只消费状态事件并向用户反馈。

## 3. 输入输出协议

### 3.1 `TaskSpec`

建议使用 JSON Schema 与 Pydantic 校验：

- `intent` 首期允许 `navigate_near_object`、`stop`、`cancel`、`status`；
- `category` 使用受控词表，如“小车/汽车/车辆”归一为 `car`；
- 属性首期支持 `color`，关系词可后续增加；
- `near_distance_m` 必须限制在视觉控制侧公布的安全范围；
- 未提及的字段使用系统默认值；
- 未知动作、未知字段、非有限数值和附带代码一律拒绝；
- “停止/取消”由本地快速通道处理，不等待远程 LLM。

### 3.2 视觉控制侧能力描述

LLM 模块只依赖固定契约：

```python
class RobotSkills(Protocol):
    def navigate_near_object(self, task: TaskSpec) -> TaskHandle: ...
    def stop(self) -> None: ...
    def cancel(self, task_id: str) -> None: ...
    def get_task_status(self, task_id: str) -> TaskStatus: ...
    def list_visible_objects(self) -> list[SemanticObject]: ...
```

不应直接 import YOLO、Genesis Scene、控制器或车辆实体。

### 3.3 状态事件

双方至少约定：

```text
TASK_ACCEPTED
TARGET_NOT_VISIBLE
SEARCHING_TARGET
TARGET_ACQUIRED
PATH_PLANNED
EXECUTING
REPLANNING
ARRIVED
AMBIGUOUS_TARGET
TARGET_NOT_FOUND
NO_SAFE_GOAL_POSE
PATH_NOT_FOUND
COLLISION_RISK
USER_CANCELLED
FAILED
```

每个事件带 `task_id`、时间、状态码和可选的安全消息；LLM 不根据自由文本猜测状态。

## 4. LLM 客户端设计

- 使用 `TaskInterpreter` 协议隔离具体厂商；
- API Key 只从环境变量或密钥系统读取；
- 设置连接超时、总超时、有限重试和退避；
- 优先使用模型原生结构化输出或 tool calling；
- 固定模型名、Schema 版本和采样参数；
- 记录 request ID、模型、耗时、token 用量和校验结果；
- 用户内容和错误日志脱敏；
- API 失败时不调用车辆技能，返回明确错误；
- 测试使用 Fake/Replay 后端，避免依赖网络和计费服务。

### 4.1 建议接口

```python
class TaskInterpreter(Protocol):
    def parse(self, user_text: str, context: TaskContext) -> TaskSpec: ...
```

`TaskContext` 只包含必要的文本摘要：

- 支持的技能和参数；
- 支持的类别、颜色和关系词；
- 安全距离和速度范围；
- 可选的可见语义对象摘要；
- 当前任务状态和结构化失败码。

不要发送 API 密钥、完整传感器数组、无限长度日志或每一帧图像。

## 5. 歧义与错误处理

| 情况                         | 处理                               |
| ---------------------------- | ---------------------------------- |
| “去小车附近”，场景有多台车 | 请求用户补充颜色/位置              |
| 两台黄色车                   | 返回候选摘要并澄清，不随机选择     |
| “附近”未给距离             | 使用双方约定默认值                 |
| 距离超出安全范围             | 裁剪或拒绝并解释                   |
| 不支持的动作                 | 明确告知当前支持能力               |
| LLM 超时/无网络              | 车辆保持静止，允许重试             |
| 非法 JSON/Schema             | 有限修复重试，仍失败则停止         |
| 目标未找到                   | 使用控制侧失败码反馈，不编造已到达 |
| 路径不可达                   | 反馈不可达，不要求模型生成轮速绕行 |
| 用户说“停止”               | 本地立即调用 stop                  |

“小车”可能指本车或目标车。系统词表应明确：

- “本车/机器人/停下”指 ego robot；
- 带颜色或方位修饰的“小车”默认指场景目标；
- 仍有歧义时请求澄清。

## 6. LLM 团队 TODO

### L0：冻结接口和供应商

- [ ] `[P0]` 确定 LLM 厂商、模型名、API 兼容方式和网络条件。
- [ ] `[P0]` 与视觉控制同事冻结 `TaskSpec` Schema v1。
- [ ] `[P0]` 冻结高层技能名称、参数和任务事件码。
- [ ] 确定 API Key 注入、日志脱敏和费用限制。
- [ ] 确定多目标歧义时的产品交互方式。

**验收：** 双方可以只依赖 Schema 和 mock 独立开发。

### L1：本地数据结构与 Fake 后端

- [ ] `[P0]` 实现 `TaskSpec/TargetQuery/NavConstraints/TaskStatus`。
- [ ] `[P0]` 实现 JSON Schema/Pydantic 校验。
- [ ] `[P0]` 实现 `TaskInterpreter` 协议。
- [ ] `[P0]` 实现 `FakeTaskInterpreter`，覆盖黄色小车指令。
- [ ] 实现同义词规范化和默认距离。
- [ ] 实现 stop/cancel/status 本地快速通道。

**验收：** 不调用真实 API，即可用中文样例生成合法 TaskSpec 并通过 mock 调用高层技能。

### L2：真实 LLM API

- [ ] `[P0]` 实现具体 API 适配器。
- [ ] `[P0]` 使用结构化输出/tool calling，不依赖自由文本正则提取。
- [ ] `[P0]` 实现超时、有限重试和错误映射。
- [ ] `[P0]` 非法输出在调用车辆前被拒绝。
- [ ] 记录脱敏请求、响应元数据和 token/耗时。
- [ ] 支持回放已保存结果进行离线回归。

**验收：** API 正常、超时、非法结构和断网时行为可预测，任何错误都不会产生车辆动作。

### L3：语言测试集

- [ ] `[P0]` 建立中文指令集和期望 TaskSpec。
- [ ] 覆盖“前往/开到/靠近/停在附近”等同义表达。
- [ ] 覆盖颜色、类别、方位和距离修饰。
- [ ] 覆盖未指定目标、多目标歧义和不支持动作。
- [ ] 覆盖停止、取消、查询状态。
- [ ] 覆盖越权指令和提示注入式输入。
- [ ] 固定模型参数执行回归并保存结果。

**验收：** 约定测试集解析正确率达到冻结指标，安全关键命令无错误执行。

### L4：与视觉控制模块集成

- [ ] 使用 Fake RobotSkills 完成 LLM 侧集成测试。
- [ ] 使用 Fake TaskInterpreter 完成控制侧集成测试。
- [ ] `[P0]` 将真实 TaskSpec 传给 `navigate_near_object()`。
- [ ] `[P0]` 消费任务状态事件并输出中文状态。
- [ ] `[P0]` 实现歧义澄清后的同一任务续接。
- [ ] 确保 LLM 不进入实时控制循环。
- [ ] 记录 `user_text → TaskSpec → task_id → final_status` 完整链路。

**验收：** 用户输入“行驶到黄色小车附近”后，LLM 只调用一次高层技能；车辆执行期间不需要逐帧调用 LLM。

## 7. 测试矩阵

| 编号 | 输入/故障            | 预期结果                     |
| ---- | -------------------- | ---------------------------- |
| L01  | 行驶到黄色小车附近   | 合法`navigate_near_object` |
| L02  | 去那辆黄车旁边       | 同义词正确归一               |
| L03  | 离黄色小车一米停下   | 输出约 1 m 约束              |
| L04  | 去小车附近，存在多车 | 请求澄清                     |
| L05  | 两台黄色车           | 请求进一步区分               |
| L06  | 停止                 | 不调用远程模型，立即 stop    |
| L07  | API 超时             | 不产生车辆任务               |
| L08  | 模型返回非法 JSON    | 校验失败并安全停止           |
| L09  | 模型输出轮速/代码    | Schema 拒绝                  |
| L10  | 控制侧返回目标不存在 | 如实反馈`TARGET_NOT_FOUND` |
| L11  | 控制侧返回路径不可达 | 如实反馈`PATH_NOT_FOUND`   |
| L12  | 提示注入式用户输入   | 不能扩展允许的技能集合       |

## 8. 建议验收指标

| 指标                       |       建议门槛 |
| -------------------------- | -------------: |
| 约定中文指令解析正确率     |     `>= 95%` |
| TaskSpec Schema 校验覆盖率 |       `100%` |
| 非法/越权动作进入控制侧    |       `0 次` |
| 正常单任务 LLM 调用次数    |       `1 次` |
| stop 本地响应              | 不依赖远程 API |
| API 故障时车辆动作         |          `0` |
| 日志中明文 API Key         |          `0` |

## 9. 需要确认

1. 使用哪个 LLM API 和模型？
2. 是否必须支持完全离线模型？
3. 首期只支持中文还是中英双语？
4. 多目标时采用文本澄清还是界面点击？
5. “附近”的默认距离和允许范围由视觉控制团队提供什么值？
6. 是否允许把语义对象摘要发送到云端？默认只发送文本摘要，不发送图像。

## 10. Definition of Done

- 自然语言只被转换为受约束的高层 TaskSpec；
- 所有 LLM 输出在进入视觉控制模块前完成 Schema 和安全校验；
- LLM API 供应商可替换，控制模块不依赖厂商 SDK；
- 歧义、超时、断网、非法输出和不支持任务均有明确行为；
- 停止命令不依赖远程服务；
- 与视觉控制团队可以通过 Fake 实现独立测试；
- 最终集成时，LLM 不生成路径和轮速、不进入实时驾驶循环；
- 用户输入、TaskSpec、任务状态和最终结果可以完整追踪。
