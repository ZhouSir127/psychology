from app.contracts import Prediction


class MockPredictor:
    """返回场景指定分数，验证业务流程；不把模态特征冒充风险评估。"""
    def __init__(self, settings):
        config = settings.options["prediction"]
        if len(config["scenarios"]) != len(config["scores"]):
            raise ValueError("演示场景和分数的数量不一致")
        self.scores = dict(zip(config["scenarios"], config["scores"]))
        self.default = config["default_scenario"]

    def predict(self, features, sample, history, scenario=None):
        selected = scenario or self.default
        if selected not in self.scores:
            raise ValueError("未知演示场景，请从配置的场景中选择")
        return Prediction(
            score=self.scores[selected], model_version="mock-v1", is_simulated=True,
            explanation=f"{selected}：分数由测试场景指定，未进行心理风险推断。"
                        f"已处理{len(features.modalities)}种模态；接口收到{len(history)}条更早记录。",
        )
