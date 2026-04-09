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