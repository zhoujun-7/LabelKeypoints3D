# 3D 和 2D 关键点标注应用程序


## 简介

本应用程序用于在静态场景中标注 2D 和 3D 关键点。它使用 ArUco 标记来计算相机内参和外参，从而能够高效地在多个视角下进行关键点标注。

**主要功能：**
- 使用 ArUco 标记自动进行相机标定
- 交互式 2D 关键点标注界面
- 从多个视角自动重建 3D 关键点
- 导出为 JSON 格式和 YOLO 姿态标签格式

## 要求

- Python 3.x
- 图像中包含 ArUco 标记
- 相机内参参数

**注意：** 本项目使用 torch-cpu 版本即可，不需要 GPU。

## 安装

```bash
pip install -r requirements.txt
```

**测试环境：**
- macOS 26
- Ubuntu 22.04

## 使用方法

### 1. 启动程序
```
python main.py
```

### 2. 加载元数据

1. 点击 **Browse Video/Directory** 按钮加载图像
2. 确保图像中包含 ArUco 标记
3. 提供相机内参参数

![Interface](fig/start.png)

### 3. 计算相机外参

1. 点击 **Start Processing** 按钮
2. 应用程序将计算所有图像的相机外参
3. 相机轨迹将被可视化，以便手动验证

![Extrinsic](fig/camera_extrinsic.png)

**注意：** 包含少于 3 个 ArUco 标记的图像将在处理过程中被忽略。

### 4. 标注关键点

1. 切换到 **Labeling** 标签页
2. 选择**至少 3 张**不同视角的图像
3. 在每张选定的图像中标注 2D 关键点

![labeling](fig/labelling.jpg)

4. 应用程序将自动为其余图像生成标签

![other view](fig/other_view.png)

### 5. 保存结果

1. 点击 **Save Label** 按钮
2. 有效图像和标注的 JSON 格式注释将保存到 `{import_path}_valid`

### 6. 转换为 YOLO 姿态标签格式（可选）

要将 JSON 注释转换为 YOLO 姿态标签格式：

```bash
python json_to_YoloLabel.py {annotation_path.json} {valid_image_path}
```

转换后的注释将保存在 `{valid_image_path}` 目录中。

## 联系方式

如有问题、建议或贡献，请联系：

- jun7zhou@gmail.com
- june.zhou@x-humanoid.com

