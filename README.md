# UESTC4006P GUI - Road Crack Detection System

## 1. Overview
This repository provides a PySide6-based desktop GUI for the final-year project **"Design and Implementation of Road Crack Detection System Based on YOLO Network Model"**.

The GUI supports object detection, crack segmentation, and ROI-guided cascade inference.
It is designed as the visual application layer of the project, while model training and evaluation are maintained in the main project repository.

## 2. Relationship with Other Repositories
- Main project repository:  
  https://github.com/ScatteredWood/UESTC4006P-Individual-Project/tree/feature/segmentation-improvement  
  This repository contains model training, validation, prediction scripts, and core YOLO experiments.
- GUI repository:  
  This repository (`uestc4006p-gui`) focuses on desktop visualization, model loading, parameter configuration, inference preview, and output export.
- YOLO26 repository:  
  https://github.com/ScatteredWood/uestc4006p-yolo26  
  YOLO26-related experiments are maintained separately and are **not a required dependency** of this GUI repository.

## 3. Features
- Load custom detection and segmentation model weights.
- Run detection-only inference.
- Run segmentation-only inference.
- Run ROI-guided cascade inference.
- Adjust confidence threshold, IoU threshold, and maximum detections.
- Preview inference results in the GUI.
- Export visualized results.
- Support Windows packaging with PyInstaller.

## 4. Project Structure
```text
uestc4006p-gui/
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ configs/
│  └─ default_models.yaml
├─ scripts/
│  ├─ build_exe.ps1
│  ├─ build_exe.bat
│  └─ pyi_rth_runtime.py
├─ src/
│  └─ uestc4006p_gui/
│     ├─ __init__.py
│     ├─ app.py
│     ├─ core/
│     ├─ inference/
│     └─ ui/
├─ tests/
│  └─ test_bridge_import.py
├─ docs/
│  └─ assets/
│     └─ .gitkeep
└─ uestc4006p_gui.spec
```

## 5. Installation
Recommended environment setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

Core dependencies are declared in `pyproject.toml`, including `PySide6`, `ultralytics`, `torch`, `opencv-python`, `numpy`, `PyYAML`, and `Pillow`.

GPU inference requires a correctly installed CUDA + PyTorch environment.
CPU inference is supported but usually slower.

## 6. Usage
Run the GUI:

```powershell
python -m uestc4006p_gui.app
```

Typical workflow:
1. Choose input image/folder or video.
2. Choose detection and/or segmentation weights.
3. Adjust inference parameters.
4. Run inference.
5. Preview and save outputs.

## 7. Model Weights
- Model weights are **not included** in this repository.
- Users should select their own `.pt` weights exported from the training repository.
- Detection and segmentation weights can be selected independently in the GUI.
- `configs/default_models.yaml` stores optional defaults only and should avoid machine-specific absolute paths.

Chinese note: this repository is submission-oriented for GUI delivery, so training artifacts and large model files are intentionally excluded.

## 8. Packaging
For Windows packaging, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Mode onedir
```

Optional onefile mode:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Mode onefile
```

## 9. Notes
- This repository focuses on GUI implementation and deployment demonstration.
- Training, validation, and quantitative evaluation are maintained in the main repository.
- YOLO26 experiments are maintained separately.

## 10. License / Acknowledgement
Please refer to the repository license if provided.
