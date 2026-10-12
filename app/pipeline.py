from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import logging
from threading import BoundedSemaphore, Lock
from app.contracts import FeatureBundle, Prediction, Sample

logger = logging.getLogger(__name__)


def load_plugin(reference, settings):
    module_name, class_name = reference.split(":", 1)
    return getattr(import_module(module_name), class_name)(settings)


class Pipeline:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db
        self.extractor = load_plugin(settings.options["plugins"]["extractor"], settings)
        self.predictor = load_plugin(settings.options["plugins"]["predictor"], settings)
        # 一条线程串行使用模型，避免重复加载、共享模型并发访问。
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.capacity = BoundedSemaphore(settings.options["app"]["max_pending_tasks"])
        self.submit_lock = Lock()

    def schedule(self, task_id):
        self.executor.submit(self.run, task_id)

    def run(self, task_id):
        try:
            task = self.db.get(task_id)
            self.db.update(task_id, "running")
            payload = task["request"]
            sample = Sample(**{k: payload[k] for k in
                              ("subject_id", "collection_id", "collected_at", "text", "source")},
                            files=task["files"])
            features = FeatureBundle.model_validate(self.extractor.extract(sample))
            # 只传入同一人的更早结果，避免未来记录泄漏给历史预测。
            history = self.db.history(sample.subject_id, task["collected_at"])
            prediction = Prediction.model_validate(self.predictor.predict(
                features, sample, history, payload.get("scenario")))
            risk = self.settings.options["risk"]
            level = next(i for i, upper in enumerate(risk["upper_bounds"]) if prediction.score <= upper)
            result = {**prediction.model_dump(), "risk_label": risk["labels"][level],
                      "risk_level": level, "features": features.model_dump()}
            self.db.update(task_id, "succeeded", result=result)
        except Exception as exc:
            logger.exception("任务 %s 失败", task_id)
            self.db.update(task_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:600]}")
        finally:
            self.capacity.release()

    def close(self):
        self.executor.shutdown(wait=True)
        if hasattr(self.extractor, "close"):
            self.extractor.close()
        if hasattr(self.predictor, "close"):
            self.predictor.close()
