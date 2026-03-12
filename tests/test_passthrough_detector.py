"""PassthroughDetector 单元测试。"""


from qp_perception.detection.passthrough import PassthroughDetector
from qp_perception.types import BoundingBox, Detection


class TestPassthroughDetector:
    def test_no_detections(self):
        d = PassthroughDetector()
        assert d.detect(None) == []

    def test_inject_and_detect(self):
        d = PassthroughDetector()
        dets = [
            Detection(
                bbox=BoundingBox(x=10, y=20, w=30, h=40),
                confidence=0.9,
                class_id="person",
            )
        ]
        d.inject(dets)
        result = d.detect(None)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_detect_clears_buffer(self):
        d = PassthroughDetector()
        d.inject(
            [
                Detection(
                    bbox=BoundingBox(x=0, y=0, w=10, h=10),
                    confidence=0.8,
                    class_id="car",
                )
            ]
        )
        d.detect(None)
        assert d.detect(None) == []

    def test_multiple_inject(self):
        d = PassthroughDetector()
        d.inject(
            [
                Detection(
                    bbox=BoundingBox(x=0, y=0, w=10, h=10),
                    confidence=0.8,
                    class_id="a",
                )
            ]
        )
        d.inject(
            [
                Detection(
                    bbox=BoundingBox(x=50, y=50, w=10, h=10),
                    confidence=0.7,
                    class_id="b",
                )
            ]
        )
        result = d.detect(None)
        assert len(result) == 1  # second inject replaces first
