# qp-perception

[![PyPI](https://img.shields.io/pypi/v/qp-perception)](https://pypi.org/project/qp-perception/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

穹沛科技模块化视觉感知库 — 检测、跟踪、Re-ID、目标选择。

被 [RWS](https://github.com/Kitjesen/RWS) 作为核心感知依赖使用。

## 安装

```bash
pip install qp-perception

# 带 Re-ID 支持（需要 PyTorch）:
pip install "qp-perception[reid]"
```

本地开发：

```bash
git clone https://github.com/Kitjesen/qp-perception.git
cd qp-perception
pip install -e ".[dev]"
```

## 快速上手

```python
from qp_perception.types import BoundingBox, Detection
from qp_perception.tracking import SimpleIoUTracker
from qp_perception.selection import WeightedTargetSelector

tracker = SimpleIoUTracker()
selector = WeightedTargetSelector()

detections = [
    Detection(
        bbox=BoundingBox(x=100, y=120, w=60, h=90),
        confidence=0.95,
        class_id="person",
    )
]

tracks = tracker.update(detections, timestamp=0.0)
target = selector.select(tracks, timestamp=0.0)
```

## 包结构

```
src/qp_perception/
├── __init__.py          # 顶层导出
├── types.py             # BoundingBox, Detection, Track 等数据类
├── interfaces.py        # Detector, Tracker, TargetSelector 协议
├── config.py            # DetectorConfig, SelectorConfig
├── kalman.py            # CentroidKalman2D, CentroidKalmanCA
├── detection/           # YoloDetector, PassthroughDetector
├── tracking/            # YoloSegTracker, FusionMOT, SimpleIoUTracker
├── selection/           # WeightedTargetSelector, WeightedMultiTargetSelector
└── reid/                # OSNet 特征提取, AppearanceGallery
```

## 发布新版本

### 方式一：Git tag 自动发布（推荐）

项目已配置 GitHub Actions，push tag 自动构建并上传到 PyPI。

```bash
# 1. 改代码，测试通过
pytest tests/ -v

# 2. 更新 pyproject.toml 中的版本号
#    version = "0.1.0"  →  "0.2.0"

# 3. 提交
git add -A
git commit -m "release: v0.2.0"

# 4. 打 tag 并推送
git tag v0.2.0
git push origin master --tags
```

GitHub Actions 会自动执行 `python -m build` + 上传到 PyPI。

> **首次使用需配置 Trusted Publishing**：
> 1. 登录 https://pypi.org → 进入 qp-perception 项目 → Settings → Publishing
> 2. 添加 "New pending publisher"：
>    - Owner: `Kitjesen`
>    - Repository: `qp-perception`
>    - Workflow: `publish.yml`
>    - Environment: 留空
> 3. 配置完成后，push tag 即可自动发布，无需 API token

### 方式二：手动发布

```bash
# 1. 更新 pyproject.toml 版本号

# 2. 构建
pip install build twine
python -m build

# 3. 上传（需要 PyPI API token）
twine upload dist/qp_perception-0.2.0*
```

### 版本号规范

遵循 [语义化版本](https://semver.org/lang/zh-CN/)：

| 改动类型 | 版本变化 | 示例 |
|----------|----------|------|
| Bug 修复 | patch +1 | 0.1.0 → 0.1.1 |
| 新功能（向后兼容） | minor +1 | 0.1.0 → 0.2.0 |
| 破坏性变更 | major +1 | 0.x → 1.0.0 |

### 下游项目更新

发布新版后，依赖 qp-perception 的项目（如 RWS）执行：

```bash
pip install --upgrade qp-perception
```

## 开发

```bash
# 测试
pytest tests/ -v

# 代码检查
ruff check src/ tests/
ruff format src/ tests/

# 类型检查
mypy src/qp_perception --ignore-missing-imports
```

## 许可证

[MIT License](LICENSE)
