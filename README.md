# 视训工作台 · Vision Workbench

> **项目声明：本项目仅用于学习和二次开发，基于 Ultralytics YOLO26 模型进行训练。**

完整声明见 [NOTICE.md](NOTICE.md)。

面向工业视觉缺陷检测的可视化训练与模型部署工作台，支持图片标注、类别分层划分、数据版本、模型持续迭代、SSH/WSL 训练环境校验及 TensorRT 产线导出。

当前版本 **v0.36.1 / schema16**。Windows 桌面安装包内置控制端 Python，无需另装 Python、Git 或 Node.js。

## 普通用户下载安装

1. 打开 [最新版本下载页](https://github.com/studyhard-zhang/vision-workbench/releases/latest)。
2. 在 Assets 下载 `VisionWorkbench-Setup-0.36.1-x64.exe`，双击运行。
3. 首次使用选择“创建新工作区”；已有数据选择“使用已有工作区”，指向包含 `platform.db` 的目录。
4. 安装后从桌面或开始菜单打开“视训工作台”。

适用 Windows 10 2004+ / Windows 11 x64。安装器检查 WebView2，缺少时通过微软引导程序安装，此步骤需要联网。当前安装包未数字签名，请核对下载来源及发布页的 SHA-256。

工作区默认位于 `%LOCALAPPDATA%\VisionWorkbenchData\var`，独立于程序目录。新工作区不含演示项目、图片、模型或设备连接。升级和卸载保留工作区。

完整步骤见 [安装与使用](docs/安装与使用.md)。安装控制端后即可管理图片和标注；训练及 GPU 推理需要另外连接已准备好的计算环境，见 [训练环境准备](docs/训练环境准备.md)。

## 功能

| 功能 | 说明 |
| --- | --- |
| 数据管理与标注 | 项目、图片数据集、检测框、多边形、统一 ROI 与高倍缩放 |
| 类别分层划分 | 按缺陷类别分配样本，尽可能覆盖训练/验证集合，展示各类目标数量分段图 |
| 不可变数据版本 | 冻结图片、标注、类别映射与划分，保留训练来源 |
| GPU 训练 | 连接本机 WSL2 或远程 Linux，校验环境、管理任务、日志和结果 |
| 持续迭代 | 使用已有模型 best.pt 和新的补标版本训练下一版，保留父模型来源 |
| 图片推理 | 检测/分割结果叠图、阈值筛选、图片和 JSON 下载 |
| TensorRT 产线交付 | FP16 推理、引擎缓存、计时，以及目标端 ONNX 构建/推理包 |
| 数据保留 | 项目备份与恢复、训练回收站、中止核实及设备状态恢复 |

## 使用流程

创建项目 → 导入图片 → 定义类别并标注 → 检查与划分 → 生成数据版本 → 连接训练设备 → 创建并启动训练 → 图片推理或产线导出。

产线漏检图片补标后，生成新的数据版本，在模型详情点击“基于此模型训练下一版”。每版训练保留来源，便于长期迭代。

## 开发者源码运行

安装 Python 3.11+，在 PowerShell 执行：

```powershell
git clone https://github.com/studyhard-zhang/vision-workbench.git
cd vision-workbench\workbench
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --port 8765 --data-dir "$env:LOCALAPPDATA\VisionWorkbenchData\var"
```

浏览器打开 <http://127.0.0.1:8765/>。保持终端运行，后续启动使用同一个数据目录。下载 ZIP 源码的用户进入解压目录中的 `workbench` 执行相同命令。

控制端使用 Python、SQLite 和原生 HTML/CSS/JavaScript；GPU 执行端使用 PyTorch 与 Ultralytics。依赖和权重由用户在计算环境中自行准备，工作台只校验并给出缺失项及手动处理指引。

## 开发验证与构建

```powershell
cd workbench
python -m unittest discover -s tests
node --test tests/box-model.test.mjs tests/editor-flow.test.mjs tests/editor-view.test.mjs tests/polygon-model.test.mjs tests/training-chart.test.mjs
```

测试使用临时目录和合成数据。桌面构建见 [构建说明](docs/构建与发布.md)。

## 目录

```text
vision-workbench/
├─ README.md
├─ AGENTS.md
├─ docs/                 # 公开用户与构建说明
└─ workbench/
   ├─ app/              # API、数据存储、任务管理
   ├─ web/              # 用户界面与标注工作台
   ├─ tools/            # 执行器、校验和部署脚本
   ├─ desktop/          # Windows 客户端与安装器
   ├─ tests/            # 自动化测试
   ├─ desktop.py
   └─ run.py
```

仓库与安装包不包含个人项目、业务数据库、图片、权重、SSH 凭据或个人测试记录。工作区迁移和备份见 [安装与使用](docs/安装与使用.md)。
