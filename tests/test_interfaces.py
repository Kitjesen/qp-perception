"""接口协议单元测试 -- 验证感知层接口定义。"""


from qp_perception.interfaces import Detector, TargetSelector, Tracker


class TestProtocolsExist:
    """验证所有协议/接口类可以被导入。"""

    def test_detector(self):
        assert Detector is not None

    def test_tracker(self):
        assert Tracker is not None

    def test_target_selector(self):
        assert TargetSelector is not None


class TestDetectorInterface:
    def test_has_detect_method(self):
        assert hasattr(Detector, "detect")


class TestTrackerInterface:
    def test_has_update_method(self):
        assert hasattr(Tracker, "update")


class TestTargetSelectorInterface:
    def test_has_select_method(self):
        assert hasattr(TargetSelector, "select")
