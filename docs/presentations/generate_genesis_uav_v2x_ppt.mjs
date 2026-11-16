// Rebuild the academic Genesis World deck with @oai/artifact-tool.
// Run a copy of this file from a build directory whose node_modules points to
// the Codex bundled runtime, as documented by the Presentations skill.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(process.cwd());
const SKILL = "/Users/vamos/.codex/plugins/cache/openai-primary-runtime/presentations/26.1110.11613/skills/presentations";
const PYTHON = "/Users/vamos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const BUILD = path.join(ROOT, "docs/presentations/.build");
const CANDIDATE = path.join(BUILD, "candidate.pptx");
const FINAL = path.join(ROOT, "docs/presentations/output/genesis_uav_v2x_research_overview_v2.pptx");
const W = 1280, H = 720;
const C = {
  paper: "#FBFCFD", ink: "#1B2733", muted: "#536476", blue: "#266083",
  pale: "#EAF1F5", line: "#CAD5DE", green: "#2E735E", orange: "#A45C2A",
};
const FONT = "PingFang SC";
const BLOG = "https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward";
const GITHUB = "https://github.com/Genesis-Embodied-AI/genesis-world";

function box(slide, x, y, w, h, fill = "none", stroke = "none", sw = 0) {
  return slide.shapes.add({
    geometry: "rect", position: {left:x, top:y, width:w, height:h},
    fill, line: {style:"solid", fill:stroke, width:sw},
  });
}
function txt(slide, value, x, y, w, h, size=23, color=C.ink, bold=false) {
  const s = slide.shapes.add({
    geometry:"textbox", position:{left:x,top:y,width:w,height:h},
    fill:"none", line:{style:"solid",fill:"none",width:0},
  });
  s.text = value;
  s.text.style = {typeface:FONT,fontSize:size,color,bold,verticalAlignment:"top",
    wrap:true,marginLeft:0,marginRight:0,marginTop:0,marginBottom:0};
  return s;
}
function header(slide, title, n, section) {
  slide.background.fill = C.paper;
  box(slide, 58, 56, 5, 50, C.blue);
  txt(slide,title,78,55,1130,56,32,C.ink,true);
  box(slide,58,124,1164,1,C.line);
  txt(slide,section,58,667,900,22,14,C.muted);
  txt(slide,String(n).padStart(2,"0"),1170,665,45,24,14,C.muted);
}
function note(slide, body, refs=[]) {
  slide.speakerNotes.text = body + "\n\n资料：" + refs.join("；");
}
function bullets(slide, arr, x, y, w, size=22, gap=70, color=C.ink) {
  arr.forEach((s,i)=>{
    box(slide,x,y+i*gap+9,7,7,C.blue);
    txt(slide,s,x+22,y+i*gap,w-22,gap-6,size,color);
  });
}
function label(slide, t, x,y,w,color=C.blue) {
  txt(slide,t,x,y,w,26,17,color,true);
  box(slide,x,y+31,w,1,C.line);
}
function cols(slide, leftTitle, left, rightTitle, right, opt={}) {
  label(slide,leftTitle,58,160,550);
  label(slide,rightTitle,674,160,548);
  bullets(slide,left,58,210,550,opt.size||21,opt.gap||74);
  bullets(slide,right,674,210,548,opt.size||21,opt.gap||74);
}
function flow(slide, items, y=360, height=86) {
  const width=(1130-(items.length-1)*22)/items.length;
  items.forEach((item,i)=>{
    const x=74+i*(width+22);
    box(slide,x,y,width,height,C.pale,C.line,1);
    txt(slide,item,x+12,y+16,width-24,height-20,20,C.ink,true);
    if(i<items.length-1) txt(slide,"→",x+width+1,y+28,24,32,26,C.blue,true);
  });
}
function line(slide, t, y, color=C.ink) {txt(slide,t,58,y,1160,44,21,color);}
const deck=Presentation.create({slideSize:{width:W,height:H}});
let n=0;
function add(title,section){const s=deck.slides.add();header(s,title,++n,section);return s;}

// 1. Cover
{
  const s=deck.slides.add(); s.background.fill=C.paper; n++;
  box(s,70,85,8,440,C.blue);
  txt(s,"Genesis World",104,94,1020,72,52,C.ink,true);
  txt(s,"从仿真内核到无人机智能决策",104,192,1050,62,38,C.blue,true);
  txt(s,"深度强化学习路径规划 · 任务卸载 · 频谱分配",104,302,1040,55,25,C.ink);
  box(s,104,395,1045,1,C.line);
  txt(s,"课题组技术分享  |  本地源码基线 genesis-world 1.3.3",104,424,1000,34,19,C.muted);
  txt(s,"核心问题：如何把物理仿真变成可复现的闭环实验系统",104,521,1000,60,25,C.ink,true);
  note(s,"本次讲述从 Genesis 的四层架构出发，落到我们课题组的三个具体研究问题。区分原生能力、本地扩展和拟建设系统。",[BLOG,GITHUB]);
}
// 2
{
  const s=add("研究动机：闭环评价决定迭代速度","研究定位");
  bullets(s,[
    "离线指标只评估一次输入到输出；闭环评价检验动作如何改变下一时刻观测。",
    "仿真让场景、干扰、初始条件与失败轨迹可重复，适合系统比较策略。",
    "Genesis 提供物理与感知环境；课题组贡献是规划、卸载、频谱策略和评价协议。",
    "论文应报告分布外失败模式，而不只展示一段成功视频。"
  ],72,172,1110,22,108);
  note(s,"官方博客以评估瓶颈为主线：可扩展、可复现的闭环系统覆盖任务和条件的组合空间。此处转译为本组算法实验要求。",[BLOG]);
}
// 3
{
  const s=add("四层技术栈与研究代码的位置","Genesis 架构");
  const rows=[
    ["研究层","PPO / MAPPO、A* / MPC、卸载、无线资源决策"],
    ["Simulation Interface","Scene、资产/实体、控制、传感器、并行环境"],
    ["Physics + Render","刚体/多物理求解；Nyx / Luisa / Pyrender"],
    ["Quadrants","Python kernel → CUDA / ROCm / Metal / Vulkan / CPU"],
    ["硬件","CPU、单卡 GPU 或集群资源"]
  ];
  rows.forEach((r,i)=>{
    const y=157+i*91;
    box(s,74,y,1130,74,i===0?"#E3EDF3":"none",C.line,1);
    txt(s,r[0],93,y+15,240,40,23,i===0?C.blue:C.ink,true);
    txt(s,r[1],341,y+16,850,42,22,C.ink);
  });
  note(s,"官方 README 把 Genesis 定义为应用与计算后端之间的四层。研究层由我们提供，不应把网络协议或强化学习算法写成平台内置。",[GITHUB]);
}
// 4
{
  const s=add("一次 scene.step() 内发生什么","源码执行链");
  flow(s,["动作写入实体","求解器子步","传感器缓存更新","奖励与终止","局部重置"],245,90);
  line(s,"scene.build(n_envs=B)：资产解析、状态分配、批量环境和内核准备",158,C.blue);
  line(s,"rigid-only / 无梯度路径有专门优化；视频渲染会额外消耗时间。",388);
  line(s,"源码：genesis/engine/scene.py、genesis/engine/simulator.py",454,C.muted);
  note(s,"Simulator.step 在刚体-only 且无梯度时直接推进 rigid_solver；一般路径处理输入、求解器子步和耦合，之后调用传感器管理器 step。相机读取按需发生。",["本地 genesis/engine/scene.py","本地 genesis/engine/simulator.py"]);
}
// 5
{
  const s=add("多物理统一场景：与无人机研究的关系","物理求解");
  cols(s,"当前主要使用",[
    "Rigid：四旋翼、车辆、建筑物与障碍的动力学和接触。",
    "固定几何与移动实体共享场景状态；传感器从场景读取观测。",
    "控制研究先验证质量、惯量、旋翼参数和碰撞判据。"
  ],"按问题扩展",[
    "FEM / MPM / PBD / SPH：柔性结构、颗粒、流体等扩展场景。",
    "多求解器耦合是平台能力；不应为了课题展示而增加无关物理。",
    "气动下洗和真实风场并非默认四旋翼模型，需要单独建模。"
  ],{size:20,gap:112});
  note(s,"官方 README 列出多物理求解器与耦合。无人机现有示例主要使用刚体，复杂材料能力与本组路径规划任务的相关性需按场景判断。",[GITHUB,"本地 examples/drone/hover_env.py"]);
}
// 6
{
  const s=add("并行环境的收益来自批量状态与计算","并行训练");
  line(s,"scene.build(n_envs=B) → 状态、动作、观测均带批量维 B",160,C.blue);
  flow(s,["B 组初始条件","一次批量物理推进","B 条轨迹经验","PPO 更新"],270,86);
  bullets(s,[
    "训练吞吐报告 env-steps/s，同时记录显存、渲染状态、传感器和硬件。",
    "避免每步逐环境 Python 循环与 GPU→CPU 同步；只对已终止环境重置。",
    "PPO 来自 rsl-rl，Genesis 提供环境接口与批量状态。"
  ],74,406,1120,20,67);
  note(s,"并行环境文档说明 n_envs 的前导批量维与 envs_idx 局部控制。GPU 收益依赖批量规模与访存，不应直接写成固定倍数。",[GITHUB,"https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/parallel_simulation.html"]);
}
// 7
{
  const s=add("传感器与渲染应按训练目标配置","观测管线");
  cols(s,"用于控制和规划",[
    "位姿/速度、IMU、LiDAR、深度：低成本状态或局部几何观测。",
    "观测延迟、抖动和历史缓冲可模拟部分传感链路不完美。",
    "每个策略周期定义清楚采样、保持和读取时刻。"
  ],"用于视觉和展示",[
    "RGB / 深度视图支持语义目标与可视化；Nyx 面向高保真渲染。",
    "高分辨率相机可能成为训练吞吐瓶颈；须做有/无相机消融。",
    "渲染真实感不自动保证动力学或无线链路真实感。"
  ],{size:20,gap:112});
  note(s,"传感器文档列出 IMU、LiDAR、深度、接触等类别和 measured/ground-truth 读取。当前版本具体相机接口与批量渲染需以本地代码核对。",[GITHUB,"https://genesis-world.readthedocs.io/en/latest/user_guide/sensing/index.html"]);
}
// 8
{
  const s=add("四旋翼从 RPM 产生力与三轴力矩","无人机动力学");
  line(s,"Fᵢ = [0, 0, K_F·RPMᵢ²]ᵀ",172,C.blue);
  line(s,"Mᵢ = [0, 0, sᵢ·K_M·RPMᵢ²]ᵀ     τ = Σ(rᵢ × Fᵢ + Mᵢ)",229,C.blue);
  box(s,72,308,1136,1,C.line);
  bullets(s,[
    "Crazyflie 2.X：m≈0.027 kg，K_F=3.16×10⁻¹⁰，K_M=7.94×10⁻¹²。",
    "悬停估算 RPM≈√(mg/(4K_F))≈14,469；动作可按批量 (B,4) 输入。",
    "集中力/力矩模型不含 CFD、下洗或真实风场。"
  ],74,342,1130,22,86);
  note(s,"参数来自本地 cf2x.urdf；公式来自 drone_entity.py。每个仿真步调用 set_propellers_rpm 一次。风与空气动力需要额外模型。",["本地 genesis/assets/urdf/drones/cf2x.urdf","本地 genesis/engine/entities/rigid_entity/drone_entity.py"]);
}
// 9
{
  const s=add("仓库现成无人机代码只覆盖基线任务","可运行示例");
  const rows=[
    ["fly_route.py","级联 PID 航点飞行","无三维避障"],
    ["hover_env.py","100 Hz，17 维观测，4 维动作","无复杂任务规划"],
    ["hover_train.py","外接 rsl-rl PPO，批量环境","算法库不在 Genesis 内"],
    ["hover_eval.py","评估与录制","不代表实机迁移"]
  ];
  txt(s,"代码",75,167,250,28,18,C.blue,true);
  txt(s,"已实现",350,167,420,28,18,C.blue,true);
  txt(s,"研究前需补",870,167,300,28,18,C.blue,true);
  rows.forEach((r,i)=>{const y=213+i*92;box(s,73,y,1135,1,C.line);txt(s,r[0],77,y+20,250,50,20,C.ink,true);txt(s,r[1],350,y+20,470,50,20);txt(s,r[2],870,y+20,320,50,20,C.orange);});
  note(s,"hover_env 以 0.01 秒步长和 0.005 秒刚体子步运行；17 维观测为目标相对位置、姿态、速度、角速度和上一动作。",[GITHUB,"本地 examples/drone/hover_env.py","本地 examples/drone/hover_train.py"]);
}
// 10
{
  const s=add("路径规划：保留稳定内环，学习局部决策","研究任务 A");
  cols(s,"MDP 设计",[
    "状态：相对目标、速度、航向、局部障碍、上一动作。",
    "动作：期望速度 (vₓ,vᵧ,v_z,ω_z) 或下一局部航点。",
    "每 0.1 s 决策一次，物理按 0.01 s 推进。"
  ],"奖励与约束",[
    "距离进展 + 到达奖励 − 碰撞/近失惩罚。",
    "再惩罚能耗代理、jerk 与姿态越界。",
    "先验证碰撞判据；必要时用解析几何安全距离。"
  ],{size:19,gap:80});
  note(s,"这是拟建设的分层控制环境，不是仓库现有功能。Drone morph 文档对碰撞检查有提示，实施前须做墙面碰撞与接触测试。",["本地 genesis/options/morphs.py","本地 examples/drone/fly_route.py"]);
}
// 11
{
  const s=add("路径规划实验必须包含强基线与分布外测试","研究任务 A");
  cols(s,"对照算法",[
    "3D voxel A* / RRT* + PID：已知地图规划基线。",
    "局部 MPC：约束动力学和安全距离。",
    "PPO/SAC：局部观测策略；多机再考虑 MAPPO。"
  ],"测试切面",[
    "静态 → 动态障碍；全局地图 → 局部传感。",
    "随机质量、旋翼系数、噪声和风扰代理。",
    "成功率、碰撞/近失、时长、路径长度、能耗代理。"
  ],{size:20,gap:112});
  note(s,"训练与 OOD 测试按障碍密度、传感器噪声和动力偏差分轴。不能把 RPM 平方和直接称为真实电池能耗。",[BLOG,"本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 12
{
  const s=add("任务卸载需要补上队列、算力与链路模型","研究任务 B");
  line(s,"T_local = Q_local + Cⱼ / f_local",170,C.blue);
  line(s,"T_edge = Q_tx + Dⱼ / R_uplink + Q_edge + Cⱼ / f_edge + T_return",235,C.blue);
  box(s,74,307,1132,1,C.line);
  bullets(s,[
    "Genesis 输出无人机/用户位置与遮挡；R_uplink 由独立网络模型计算。",
    "策略同时选择下一航点、卸载比例/目的节点，位置会反过来改变未来速率。",
    "奖励：任务收益 − 截止违约 − 飞行/计算/传输代价 − 队列积压。"
  ],74,350,1125,21,84);
  note(s,"D_j 为输入大小、C_j 为计算周期、T_j 为截止时间。该式是研究模型示例，并非 Genesis 内置边缘计算模块。",["本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 13
{
  const s=add("卸载实验的关键是联合效应和消融","研究任务 B");
  cols(s,"实验变量",[
    "任务到达率、输入大小、截止时间、边缘算力。",
    "带宽、遮挡、移动速度和基站位置。",
    "相同物理/任务随机种子比较全部方法。"
  ],"对照与指标",[
    "固定轨迹 + 贪心卸载；启发式轨迹 + 最短队列。",
    "联合 RL；去掉位置、队列、遮挡信息的消融。",
    "完成率、95 分位时延、能耗、队列稳定性。"
  ],{size:20,gap:112});
  note(s,"固定速率表会切断位置与链路的反馈，不能用于证明轨迹—卸载联合优化。",["本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 14
{
  const s=add("频谱分配由几何状态驱动，但链路需外建","研究任务 C");
  line(s,"SINRᵢ,ₖ = Pᵢgᵢ,ₖ / (N₀Bₖ + Σₘ∈Sₖ,ₘ≠ᵢ Pₘgₘ,ₖ)",169,C.blue);
  line(s,"Rᵢ,ₖ = Bₖ log₂(1 + SINRᵢ,ₖ) · η_MCS",231,C.blue);
  box(s,74,307,1132,1,C.line);
  bullets(s,[
    "输入：位置、遮挡、队列、上次 CSI 与信息时效 AoI。",
    "动作：信道/时隙、功率、用户关联；约束干扰与功率预算。",
    "标准协议与 BLER/MCS 细节应由 ns-3/OMNeT++ 或标定模型提供。"
  ],74,350,1125,21,84);
  note(s,"公式是简化的研究链路预算。阴影、快衰落、波束、重传和协议开销需要明确假设，Genesis 不原生给出 SINR 或吞吐。",[GITHUB]);
}
// 15
{
  const s=add("物理与网络通过离散时钟同步","联合仿真设计");
  line(s,"建议初值：dt_phys = 0.01 s，dt_policy = dt_net = 0.1 s",160,C.blue);
  flow(s,["读 t 时刻已到达消息","决策轨迹/卸载/频谱","物理推进 10 步","网络处理发包/队列","产生 t+Δt 观测"],269,88);
  bullets(s,[
    "消息携带 sim_time、agent_id、pose、visibility、queue、channel、delivered_time。",
    "决策不能读取未来 ACK/CSI；物理、网络、任务随机数分别记录。"
  ],74,422,1125,20,86);
  note(s,"此时钟是建议实验设计而非平台默认参数。若网络仿真采用事件驱动，桥接时需屏障同步；不能用墙钟速度改变结果。",["本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 16
{
  const s=add("频谱实验应报告效率与用户尾部表现","研究任务 C");
  cols(s,"基线与消融",[
    "随机/轮询频谱，最大 SINR，比例公平，优化器。",
    "单智能体与多智能体 RL。",
    "去掉位置、遮挡和 AoI，检验物理仿真的增量。"
  ],"评价指标",[
    "总吞吐与频谱效率。",
    "5 分位用户速率、任务违约率和 Jain 公平指数。",
    "信令开销、训练样本效率和跨场景泛化。"
  ],{size:20,gap:112});
  note(s,"所有算法共享同一时钟、场景和任务到达样本。尾部速率与公平性避免总吞吐掩盖少数用户被牺牲。",["本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 17
{
  const s=add("本地移动机器人 demo 展示可迁移的闭环结构","本地扩展");
  flow(s,["中文指令","视觉候选 + RGB-D","二维 A* 路线","航点控制 + LiDAR"],234,98);
  bullets(s,[
    "room_navigation_vision.py：目标定位、静态路径、运行时安全与日志。",
    "视觉只提供目标；几何规划和低层控制仍分别处理可达性与安全。",
    "迁移无人机需 3D 占据体素、安全包络、动力学可达性与 RPM 内环。"
  ],74,382,1120,20,82);
  note(s,"该目录是本地扩展，不是 Genesis 上游案例。其工程组织可借鉴，不能声称已经完成无人机或频谱任务。",["本地 examples/mobile_robot/README.md","本地 examples/mobile_robot/room_navigation_vision.py"]);
}
// 18 Demo video reserved
{
  const s=add("Demo 视频：移动机器人语义目标导航","本地演示");
  box(s,72,158,800,450,"#F1F5F7",C.line,2);
  txt(s,"视频位置（16:9）",284,329,390,55,34,C.blue,true);
  txt(s,"录制后在 PowerPoint 中插入视频",294,381,370,40,18,C.muted);
  label(s,"讲解顺序",914,163,292);
  bullets(s,[
    "任务指令与场景",
    "目标检测、RGB-D 定位",
    "A* 路线与 LiDAR 安全层",
    "到达判据和运行日志"
  ],914,222,290,18,87);
  note(s,"当前 examples/mobile_robot 内无可嵌入的录制视频。建议录 60–90 秒，并并排显示俯视、机器人视角与时间戳。演示运行命令见 examples/mobile_robot/README.md。",["本地 examples/mobile_robot/README.md"]);
}
// 19
{
  const s=add("落地顺序：先建立可信基线，再做联合决策","实施路线");
  const rows=[
    ["0","复现 fly_route / hover，验证撞墙与接触判据"],
    ["1","构建三维导航环境与 A*/MPC/RL 对照"],
    ["2","加入任务、边缘计算与可手算校验的队列"],
    ["3","接入信道/频谱模块，完成统一时钟与消融"],
    ["4","参数标定、真实日志回放与 sim-to-real 差距"]
  ];
  rows.forEach((r,i)=>{let y=153+i*94;box(s,77,y,46,46,C.pale,C.line,1);txt(s,r[0],91,y+7,30,32,24,C.blue,true);txt(s,r[1],145,y+6,1044,57,22);box(s,144,y+69,1040,1,C.line);});
  note(s,"每阶段通过可复现的测试门槛再进入下一阶段。重点防止碰撞链路未验证、无线结果未来信息泄漏及只在训练分布内评价。",["本地 docs/presentations/genesis_uav_v2x_research_overview.md"]);
}
// 20
{
  const s=add("最终评价体系：性能、鲁棒性与真实性","实验规范");
  cols(s,"每项实验固定记录",[
    "版本/提交、随机种子、硬件、时间步、传感器设置。",
    "训练与测试场景划分、任务到达和信道假设。",
    "轨迹、观测、动作、奖励、队列与网络事件日志。"
  ],"必须回答",[
    "能否复现？失败主要来自哪一层？",
    "物理/网络扰动下性能如何变化？",
    "算法增益能否在强基线和消融后保留？"
  ],{size:20,gap:112});
  note(s,"官方博客强调按不同扰动轴评价鲁棒性。课题组应把物理、网络、感知和任务分布分别扫参。",[BLOG]);
}
// 21
{
  const s=add("资料与代码入口","参考文献");
  bullets(s,[
    "Genesis AI, The Role of Simulation in Scalable Robotics, Genesis World 1.0, 2026.",
    "Genesis-Embodied-AI / genesis-world：README、Scene/Simulator、传感器与无人机示例。",
    "本地文档：docs/presentations/genesis_uav_v2x_research_overview.md。",
    "本地 demo：examples/mobile_robot/README.md 与 room_navigation_vision.py。"
  ],72,173,1130,21,104);
  note(s,"官方博客与 GitHub 链接写在本文档顶部。源码详细路径见配套 Markdown。",[BLOG,GITHUB]);
}

await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.dirname(FINAL),{recursive:true});
for(let i=0;i<deck.slides.items.length;i++){
  const png=await deck.export({slide:deck.slides.items[i],format:"png",scale:1});
  await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,"0")}.png`),new Uint8Array(await png.arrayBuffer()));
}
await (await PresentationFile.exportPptx(deck)).save(CANDIDATE);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,"container_tools/artifact_tool_utils.mjs")).href);
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:CANDIDATE,finalPath:FINAL,pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],
  requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:"design",families:[FONT]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(BUILD,"validation.json"),
});
console.log(JSON.stringify({slides:deck.slides.items.length,final:FINAL,result},null,2));
