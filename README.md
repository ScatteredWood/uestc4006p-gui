# UESTC4006P GUI

UESTC4006P GUI 是一个基于 Python 的桌面图形界面工具，用于对目标检测、图像分割以及级联推理流程进行可视化操作与结果导出。

查看主项目：https://github.com/ScatteredWood/UESTC4006P-Individual-Project

---

## 主要功能

- 图形化选择输入图片、视频或文件夹
- 支持检测、分割及级联推理流程
- 支持加载自定义模型权重与默认模型配置
- 支持结果预览与导出
- 支持将项目进一步打包为 Windows 单文件 `.exe`

---

## 运行依赖（开发态）

- `ultralytics`（内部会使用其 `YOLO` API）
- `torch`（由 ultralytics 推理依赖）
- `PySide6` / `opencv-python` / `numpy` / `PyYAML` / `Pillow`

> 当前 GUI 已去除对外部 `E:\repositories\ultralytics` 源码仓库路径的运行时依赖。
> 运行时只依赖已安装 Python 包（开发态）或打包产物（分发态）。

---

## Windows 打包

主目标为 onefile：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Mode onefile
```

备选 onedir：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Mode onedir
```

默认使用 `uestc4006p_gui.spec`，并自动带上 Qt 插件、torch、cv2、ultralytics 相关依赖。

---

## 项目结构

```text
uestc4006p-gui/
├─ configs/
│  └─ default_models.yaml
├─ src/
│  └─ uestc4006p_gui/
│     ├─ app.py
│     ├─ core/
│     ├─ inference/
│     └─ ui/
├─ tests/
├─ pyproject.toml
└─ README.md
