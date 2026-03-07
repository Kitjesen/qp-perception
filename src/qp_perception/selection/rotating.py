"""Rotating multi-target selector for single gimbal."""

from __future__ import annotations

import logging

from qp_perception.config import SelectorConfig
from qp_perception.types import TargetObservation, Track

logger = logging.getLogger(__name__)


class RotatingTargetSelector:
    """单云台轮询多目标选择器.

    在多个高分目标间定期切换，让单个云台能够"照顾"多个目标。

    工作模式：
    1. 选择 Top-N 个目标（按评分排序）
    2. 每隔 rotation_interval_s 切换到下一个目标
    3. 如果当前目标丢失，立即切换到下一个

    适用场景：
    - 单云台需要监控多个目标
    - 目标移动较慢，允许轮询
    - 不需要持续精确跟踪单个目标
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: SelectorConfig | None = None,
        rotation_interval_s: float = 3.0,
        max_targets: int = 3,
        *,
        dwell_time_s: float | None = None,
    ):
        """初始化轮询选择器.

        Parameters
        ----------
        frame_width : int
            画面宽度
        frame_height : int
            画面高度
        config : SelectorConfig
            选择器配置
        rotation_interval_s : float
            轮询间隔（秒），默认3秒
        max_targets : int
            最多跟踪几个目标，默认3个
        """
        self._w = frame_width
        self._h = frame_height
        self._cfg = config if config is not None else SelectorConfig()
        # dwell_time_s is a backwards-compatible alias for rotation_interval_s
        self._rotation_interval = dwell_time_s if dwell_time_s is not None else rotation_interval_s
        self._max_targets = max_targets

        self._target_pool: list[int] = []  # 目标池（track_id列表）
        self._current_index = 0  # 当前跟踪的目标索引
        self._last_switch_time = 0.0  # 上次切换时间
        self._current_target_id: int | None = None

    def select(self, tracks: list[Track], timestamp: float) -> TargetObservation | None:
        """选择当前应该跟踪的目标.

        Parameters
        ----------
        tracks : List[Track]
            所有跟踪到的目标
        timestamp : float
            当前时间戳

        Returns
        -------
        Optional[TargetObservation]
            当前应该跟踪的目标（如果有）
        """
        if not tracks:
            self._target_pool = []
            self._current_target_id = None
            return None

        # 1. 更新目标池：选择 Top-N 个目标
        self._update_target_pool(tracks)

        if not self._target_pool:
            self._current_target_id = None
            return None

        # 2. 检查是否需要切换目标
        should_switch = self._should_switch(tracks, timestamp)

        if should_switch:
            self._switch_to_next_target(timestamp)

        # 3. 获取当前目标
        if self._current_target_id is None:
            # 首次选择
            self._current_target_id = self._target_pool[0]
            self._current_index = 0
            self._last_switch_time = timestamp
            logger.info("Initial target selected: ID %d", self._current_target_id)

        # 4. 查找当前目标的 Track
        current_track = next((t for t in tracks if t.track_id == self._current_target_id), None)

        if current_track is None:
            # 当前目标丢失，立即切换
            logger.warning("Current target %d lost, switching immediately", self._current_target_id)
            self._switch_to_next_target(timestamp)

            # 重新查找
            if self._current_target_id is not None:
                current_track = next(
                    (t for t in tracks if t.track_id == self._current_target_id), None
                )

        if current_track is None:
            return None

        # 5. 返回观测
        return TargetObservation(
            timestamp=timestamp,
            track_id=current_track.track_id,
            bbox=current_track.bbox,
            confidence=current_track.confidence,
            class_id=current_track.class_id,
            velocity_px_per_s=current_track.velocity_px_per_s,
            acceleration_px_per_s2=current_track.acceleration_px_per_s2,
            mask_center=current_track.mask_center,
        )

    def _update_target_pool(self, tracks: list[Track]) -> None:
        """更新目标池（Top-N 个目标）."""
        # 对所有目标评分
        scored = [(self._compute_score(t), t) for t in tracks]
        scored.sort(key=lambda x: x[0], reverse=True)

        # 取 Top-N
        top_tracks = [t for _, t in scored[: self._max_targets]]
        new_pool = [t.track_id for t in top_tracks]

        # 如果目标池变化，记录日志
        if set(new_pool) != set(self._target_pool):
            logger.info("Target pool updated: %s -> %s", self._target_pool, new_pool)

        self._target_pool = new_pool

        # 如果当前目标不在新池中，重置索引
        if self._current_target_id not in self._target_pool:
            self._current_index = 0
            self._current_target_id = None

    def _should_switch(self, tracks: list[Track], timestamp: float) -> bool:
        """判断是否应该切换目标."""
        if len(self._target_pool) <= 1:
            # 只有一个目标，不需要切换
            return False

        if self._current_target_id is None:
            # 还没有选择目标
            return False

        # 检查当前目标是否还存在
        current_exists = any(t.track_id == self._current_target_id for t in tracks)
        if not current_exists:
            # 当前目标丢失，立即切换
            return True

        # 检查是否到了轮询时间
        elapsed = timestamp - self._last_switch_time
        if elapsed >= self._rotation_interval:
            return True

        return False

    def _switch_to_next_target(self, timestamp: float) -> None:
        """切换到下一个目标."""
        if not self._target_pool:
            self._current_target_id = None
            return

        # 轮询到下一个
        self._current_index = (self._current_index + 1) % len(self._target_pool)
        old_id = self._current_target_id
        self._current_target_id = self._target_pool[self._current_index]
        self._last_switch_time = timestamp

        logger.info(
            "Target switched: ID %s -> %s (index %d/%d)",
            old_id,
            self._current_target_id,
            self._current_index + 1,
            len(self._target_pool),
        )

    def _compute_score(self, track: Track) -> float:
        """计算目标评分（与 WeightedTargetSelector 相同）."""
        cx, cy = track.bbox.center
        area = track.bbox.area

        # 1. 置信度
        conf_score = track.confidence

        # 2. 尺寸
        max_area = self._w * self._h
        size_score = min(area / max_area, 1.0)

        # 3. 中心距离
        dx = (cx - self._w / 2) / (self._w / 2)
        dy = (cy - self._h / 2) / (self._h / 2)
        dist = (dx**2 + dy**2) ** 0.5
        center_score = max(0.0, 1.0 - dist)

        # 4. 轨迹年龄
        age_norm = min(track.age_frames / self._cfg.age_norm_frames, 1.0)
        age_score = age_norm

        # 5. 类别偏好
        preferred = self._cfg.preferred_classes or {}
        class_bonus = preferred.get(track.class_id, 0.5)

        # 加权组合
        weights = self._cfg.weights
        score = (
            conf_score * weights.confidence
            + size_score * weights.size
            + center_score * weights.center_proximity
            + age_score * weights.track_age
            + class_bonus * weights.class_weight
        )

        return score

    def get_target_pool(self) -> list[int]:
        """获取当前目标池（用于调试/可视化）."""
        return self._target_pool.copy()

    def get_current_target_info(self) -> dict:
        """获取当前目标信息（用于调试/可视化）."""
        return {
            "current_id": self._current_target_id,
            "current_index": self._current_index,
            "pool_size": len(self._target_pool),
            "target_pool": self._target_pool,
        }
