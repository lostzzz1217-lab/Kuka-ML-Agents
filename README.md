# Unity ML-Agents KUKA 机械臂项目复现说明

GitHub 仓库地址：

```text
<待填写：新 GitHub 仓库 URL>
```

本仓库根目录就是一个 Unity 工程：

```text
Kuka_ML-Agents/
```

用 Unity Hub 打开时，直接选择 `Kuka_ML-Agents` 这个文件夹。

## 1. 仓库结构

```text
Kuka_ML-Agents/
├── README.md
├── Assets/
│   ├── kuka-pick-up-and-place-obstacle-avoidance-main/
│   ├── kuka_assemble_main/
│   ├── Multiple_Task/
│   │   └── Multiple_Task/
│   └── Welding_main/
│       └── Welding_main/
├── Packages/
│   └── manifest.json
├── ProjectSettings/
└── config/
```

`Assets` 下的四个项目文件夹都要完整保留，尤其不要漏掉 `.meta` 文件。Unity 依靠 `.meta` 保存资源 GUID，缺失后场景、Prefab、材质、脚本和 ONNX 模型引用很容易断开。

## 2. 项目总览

| 项目 | 资源位置 | 复现重点 |
|---|---|---|
| KUKA 抓取、放置与避障 | `Assets/kuka-pick-up-and-place-obstacle-avoidance-main` | 随机零件抓取、投放区放置、障碍规避、PPO 课程学习 |
| KUKA peg-socket 装配 | `Assets/kuka_assemble_main` | 圆柱 peg 拾取、移动到 socket、对齐、插入、释放、视觉快照辅助观测 |
| Multiple Task 整合演示 | `Assets/Multiple_Task/Multiple_Task` | 抓取、夹爪、避障、焊接等多个场景和 ONNX 模型 |
| Welding Main 焊缝追踪 | `Assets/Welding_main/Welding_main` | 末端相机视觉输入、焊缝 Spline、关节本体感知、焊缝跟踪 |

## 3. 关键路径

### 3.1 抓取、放置与避障

```text
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scenes/train.unity
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scenes/SampleScene.unity
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Prefabs/training_area.prefab
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scripts/RobotArmAgent.cs
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scripts/RandomPickupPartSpawner.cs
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scripts/WorkspaceCameraRig.cs
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/KukaPickPlace.onnx
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/KukaReach.onnx
```

训练配置：

```text
config/kuka_ppo.yaml
config/robot_arm_ppo.yaml
```

### 3.2 Peg-Socket 装配

```text
Assets/kuka_assemble_main/Scenes/train.unity
Assets/kuka_assemble_main/Scenes/SampleScene.unity
Assets/kuka_assemble_main/Prehabs/assemble.prefab
Assets/kuka_assemble_main/Scripts/KukaAssembleAgent.cs
Assets/kuka_assemble_main/Scripts/Editor/PegSocketAssemblyBuilder.cs
Assets/kuka_assemble_main/Scripts/Editor/SnapshotCameraSetup.cs
```

训练配置：

```text
config/kuka_assemble_ppo.yaml
```

### 3.3 Multiple Task

```text
Assets/Multiple_Task/Multiple_Task/Scenes/DemoScene.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Training.unity
Assets/Multiple_Task/Multiple_Task/Scenes/NoVision.unity
Assets/Multiple_Task/Multiple_Task/Scenes/OA.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Claw_Train.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Claw_display.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Weld.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Weld_Vision.unity
Assets/Multiple_Task/Multiple_Task/Scripts/RobotArmAgent.cs
Assets/Multiple_Task/Multiple_Task/Scripts/RobotArmAgent_Vector.cs
Assets/Multiple_Task/Multiple_Task/Scripts/Obstacle_Avoidance.cs
Assets/Multiple_Task/Multiple_Task/Scripts/WeldingAgent.cs
Assets/Multiple_Task/Multiple_Task/Scripts/CamWelding.cs
Assets/Multiple_Task/Multiple_Task/TFModels/
```

训练配置：

```text
config/KukaReach.yaml
config/Obstacle_Avoidance.yaml
config/WeldingVisual.yaml
```

### 3.4 Welding Main

```text
Assets/Welding_main/Welding_main/Scenes/Weld.unity
Assets/Welding_main/Welding_main/Scenes/Weld_Display.unity
Assets/Welding_main/Welding_main/Scenes/Training.unity
Assets/Welding_main/Welding_main/Scenes/NoVision.unity
Assets/Welding_main/Welding_main/Scripts/WeldingAgent.cs
Assets/Welding_main/Welding_main/Scripts/RobotArmAgent.cs
Assets/Welding_main/Welding_main/Scripts/RobotArmAgent_Vector.cs
Assets/Welding_main/Welding_main/TFModels/
```

焊缝相关配置：

```text
config/welding.yaml
config/ppo_welding_version.yaml
```

## 4. 环境要求

| 依赖 | 推荐版本/说明 |
|---|---|
| Unity Editor | `6000.0.40f1` |
| Unity Hub | 用于打开 `Kuka_ML-Agents` 工程 |
| Python | `3.10.x`，用于运行 `mlagents-learn` |
| ML-Agents | Unity 包在 `Packages/manifest.json` 中通过 URL 安装 |
| URDF Importer | Unity 包在 `Packages/manifest.json` 中通过 URL 安装 |
| Unity Splines | 焊缝任务需要 |
| URP | 抓取/装配项目设置需要 |

建议先按 Unity 官方 ML-Agents 教程完成一次最基础的学习环境配置和训练流程：

```text
https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html
```

Unity 版本可在这里确认：

```text
ProjectSettings/ProjectVersion.txt
```

包依赖统一在这里：

```text
Packages/manifest.json
```

其中 `com.unity.ml-agents` 和 `com.unity.robotics.urdf-importer` 已经使用 URL，不再依赖本机绝对路径。

## 5. 打开工程

1. 克隆或下载仓库。
2. 用 Unity Hub 打开 `Kuka_ML-Agents`。
3. 等待 Unity 自动导入 `Assets`、`Packages`、`ProjectSettings`。
4. 如果 Console 没有编译错误，再打开对应场景。

建议先打开展示或基础场景检查资源完整性：

```text
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scenes/SampleScene.unity
Assets/kuka_assemble_main/Scenes/SampleScene.unity
Assets/Multiple_Task/Multiple_Task/Scenes/DemoScene.unity
Assets/Welding_main/Welding_main/Scenes/Weld_Display.unity
```

## 6. Python 训练环境

建议使用 Python 3.10 虚拟环境。只要终端能正常执行下面命令即可：

```bash
mlagents-learn --help
```

如果尚未安装 Python 端 ML-Agents，可在虚拟环境中安装与 Unity ML-Agents 版本兼容的训练工具。安装完成后，在仓库根目录 `Kuka_ML-Agents` 下运行训练命令。

## 7. 运行已有模型

通用步骤：

1. 打开 `Kuka_ML-Agents` Unity 工程。
2. 打开对应场景。
3. 在 Hierarchy 中选中带 Agent 脚本的机械臂对象。
4. 在 Inspector 中找到 `Behavior Parameters`。
5. 确认 `Behavior Name` 与任务一致。
6. 将对应 `.onnx` 模型拖入 `Model`。
7. 将 `Behavior Type` 设为 `Inference Only`。
8. 点击 Play。

常用模型位置：

| 项目 | 模型位置 |
|---|---|
| 抓取、放置与避障 | `Assets/kuka-pick-up-and-place-obstacle-avoidance-main/KukaPickPlace.onnx`、`Assets/kuka-pick-up-and-place-obstacle-avoidance-main/KukaReach.onnx` |
| Multiple Task | `Assets/Multiple_Task/Multiple_Task/TFModels/` |
| Welding Main | `Assets/Welding_main/Welding_main/TFModels/` |

## 8. 重新训练

训练前检查：

- Agent 的 `Behavior Parameters -> Model` 必须清空为 `None`。
- Unity 场景里的 `Behavior Name` 必须和 YAML 顶部 `behaviors:` 下的名字完全一致。
- 训练命令在仓库根目录 `Kuka_ML-Agents` 下执行。
- 命令行出现等待提示后，回到 Unity 点击 Play。

### 8.1 抓取、放置与避障

```bash
mlagents-learn config/kuka_ppo.yaml --run-id=KukaPickPlace_reproduce
```

早期/简化 reach 变体：

```bash
mlagents-learn config/robot_arm_ppo.yaml --run-id=KukaReach_reproduce
```

常用场景：

```text
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scenes/train.unity
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Scenes/SampleScene.unity
```

### 8.2 Peg-Socket 装配

```bash
mlagents-learn config/kuka_assemble_ppo.yaml --run-id=KukaAssemble_reproduce
```

常用场景：

```text
Assets/kuka_assemble_main/Scenes/train.unity
Assets/kuka_assemble_main/Scenes/SampleScene.unity
```

### 8.3 Multiple Task 抓取、夹爪、避障、视觉焊接

抓取/夹爪：

```bash
mlagents-learn config/KukaReach.yaml --run-id=KukaReach_reproduce
```

避障：

```bash
mlagents-learn config/Obstacle_Avoidance.yaml --run-id=OA_reproduce
```

视觉焊接：

```bash
mlagents-learn config/WeldingVisual.yaml --run-id=VisualWelding_reproduce
```

常用场景：

```text
Assets/Multiple_Task/Multiple_Task/Scenes/Claw_Train.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Claw_display.unity
Assets/Multiple_Task/Multiple_Task/Scenes/OA.unity
Assets/Multiple_Task/Multiple_Task/Scenes/Weld_Vision.unity
```

### 8.4 Welding Main 焊缝追踪

向量版焊接：

```bash
mlagents-learn config/welding.yaml --run-id=Welding_reproduce
```

备用/旧版视觉配置：

```bash
mlagents-learn config/ppo_welding_version.yaml --run-id=WeldingVision_reproduce
```

常用场景：

```text
Assets/Welding_main/Welding_main/Scenes/Weld.unity
Assets/Welding_main/Welding_main/Scenes/Weld_Display.unity
Assets/Welding_main/Welding_main/Scenes/Training.unity
```

## 9. 训练监控和续训

查看训练曲线：

```bash
tensorboard --logdir results
```

浏览器打开：

```text
http://localhost:6006
```

续训：

```bash
mlagents-learn <yaml路径> --run-id=<已有run-id> --resume
```

覆盖旧训练：

```bash
mlagents-learn <yaml路径> --run-id=<已有run-id> --force
```

## 10. 复现要点

### 10.1 抓取、放置与避障

`RobotArmAgent.cs` 负责状态机、观测、奖励和障碍碰撞处理。训练采用 PPO 和课程学习，`phase` 从 Reach、Pick、Carry 逐步过渡到完整 Place。`RandomPickupPartSpawner.cs` 负责每回合随机生成零件，`WorkspaceCameraRig.cs` 负责工作区相机快照。

重点检查：

- `Behavior Name` 应与 YAML 中 `KukaPickPlace` 或 `KukaReach` 一致。
- 新训练前清空 `Model` 字段。
- `Assets/kuka-pick-up-and-place-obstacle-avoidance-main/Prefabs/training_area.prefab` 中机器人、桌面、投放区、相机、spawner 引用不能丢。

### 10.2 Peg-Socket 装配

`KukaAssembleAgent.cs` 使用关节状态、末端姿态、peg/socket 相对几何、任务阶段和一次性视觉快照来训练装配策略。`PegSocketAssemblyBuilder.cs` 可在 Editor 中生成 peg/socket 结构，`SnapshotCameraSetup.cs` 可创建或重置用于识别 peg 和 socket 的俯视相机。

重点检查：

- `Behavior Name` 应为 `KukaAssemble`。
- 场景中 `Peg`、`Socket`、`PreInsertPoint`、`InsertTarget`、`SocketAxis`、`WorkSurface` 引用要完整。
- `SnapshotCamera` 看不到 peg/socket 时，脚本会回退到 ground truth，但训练和复现时仍建议修好相机。
- `curriculum_lesson` 是核心课程参数，复现训练时不要随意删除。

### 10.3 Multiple Task

`Multiple_Task` 包含多个场景、Prefab、脚本、材质、RenderTexture、URDF/mesh/glb 和 ONNX 模型。它包含抓取、夹爪、避障和部分焊接场景。

重点检查：

- 必须完整保留 `Assets/Multiple_Task/Multiple_Task` 和所有 `.meta` 文件。
- `Assets/Multiple_Task/Multiple_Task/TFModels/` 下模型应与场景中的 Behavior Name 对应。
- `Assets/Multiple_Task/Multiple_Task/kuka_kr6_support` 下 URDF、mesh、glb 资源不能漏。

### 10.4 Welding Main

`Welding_main` 重点是焊缝追踪训练流程。视觉版使用末端相机的 `Camera Sensor Component`，同时输入关节本体感知。

当前视觉焊接脚本实际添加的是：

```text
6 个关节位置 + 6 个关节速度 + 1 个状态机阶段 = 13 维向量观测
```

因此按当前代码复现时，Unity 面板里的 Vector Observation Space Size 应设为 `13`。

推荐 Camera Sensor 设置：

```text
Sensor Name: weld_eye
Width: 84
Height: 84
Grayscale: false
Observation Stacks: 3
```

训练前一定要检查 `WeldEyeCamera` 的画面，让焊缝清楚可见。焊缝建议用亮黄色或橙黄色，地面用浅灰色，背景保持简单。

## 11. 常见问题

### Unity 打开后脚本报错

优先检查 `Packages/manifest.json` 是否导入成功，尤其是 `com.unity.ml-agents`、`com.unity.robotics.urdf-importer`、`com.unity.splines`、URP。如果 `Agent`、`BehaviorParameters`、`SplineContainer` 等类型找不到，通常是 Unity 包没有正确导入。

### 机械臂模型或 mesh 丢失

检查对应资源目录下的 KUKA 支持文件是否完整存在：

```text
Assets/Multiple_Task/Multiple_Task/kuka_kr6_support
Assets/Welding_main/Welding_main/kuka_kr6_support
Assets/kuka-pick-up-and-place-obstacle-avoidance-main/kuka
Assets/kuka_assemble_main/kuka
```

不要只复制 `.unity` 场景文件，必须保留完整资源文件夹和 `.meta`。

### Prefab、材质或模型引用丢失

通常是 `.meta` 文件缺失或复制位置改变导致。建议从完整仓库重新拉取，不要手动拼文件。

### `mlagents-learn` 找不到

重新激活 Python 虚拟环境，并确认 Python 端 ML-Agents 已安装。

### 点击 Play 后机械臂不动

检查：

- `Behavior Type` 是否设错。
- 推理时是否挂了 `.onnx`。
- 训练时 `Model` 是否清空。
- Continuous Actions 是否为 6。
- `link_1` 到 `link_6` 是否按顺序拖入。
- `Decision Requester` 是否存在。
- `ArticulationBody` 的关节驱动是否被锁死。

### Observation Size 报错

这是 Unity 面板里的 Vector Observation Space Size 和脚本实际 `AddObservation` 数量不一致。尤其注意焊接视觉版当前是 13 维向量观测加 Camera Sensor。

### ONNX 模型没有效果

检查：

- `Behavior Name` 是否和模型训练时一致。
- `Behavior Type` 是否为 `Inference Only`。
- `Model` 是否挂到正确 Agent 对象上。
- 场景里是否有多个 Agent，模型是不是挂到了错误对象。

### 焊缝视觉版学不会

先降低难度：

- 使用固定焊缝。
- 增大容差。
- 确认 `WeldEyeCamera` 画面能看到高对比度焊缝。
- 先跑向量版确认机械臂、Spline 和奖励逻辑正常，再跑视觉版。

## 12. 提交前检查清单

正式提交 `Kuka_ML-Agents` 前，请检查：

- `README.md` 位于仓库根目录。
- `Assets/kuka-pick-up-and-place-obstacle-avoidance-main` 完整存在。
- `Assets/kuka_assemble_main` 完整存在。
- `Assets/Multiple_Task/Multiple_Task` 完整存在。
- `Assets/Welding_main/Welding_main` 完整存在。
- `Packages/manifest.json` 存在，且只有这一份包清单。
- `ProjectSettings/` 存在。
- `config/` 下所有训练 YAML 已提交。
- `.meta`、`.unity`、`.prefab`、`.cs`、`.onnx`、`.mat`、`.renderTexture`、URDF/mesh/glb 文件都已提交。
- 如果单个文件超过 GitHub 限制，使用 Git LFS 管理模型或大型资源。
