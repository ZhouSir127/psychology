"""真实读取公开样例并提取简单特征；预处理仅用于工程联调。"""
import numpy as np
from app.contracts import FeatureBundle, ModalityFeature


class DemoFeatureExtractor:
    def __init__(self, settings):
        self.settings = settings
        self.face = None
        self.text_encoder = None

    def extract(self, sample):
        modalities, notes = {}, []
        for kind, path in sample.files.items():
            modalities[kind] = getattr(self, "extract_" + kind)(path)
        if sample.text.strip():
            modalities["text"] = self.extract_text(sample.text)
        if not modalities:
            raise ValueError("至少提供一种模态")
        if sample.source == "demo":
            notes.append("公开样例来自不同来源，属于模拟组合，不具备跨模态同步关系。")
        notes.append("特征处理用于工程演示，未执行完整伪迹清理或医学有效性验证。")
        return FeatureBundle(extractor_version="demo-features-v1", modalities=modalities,
                             missing_modalities=[m for m in ("eeg", "face", "audio", "text")
                                                 if m not in modalities], notes=notes)

    def extract_eeg(self, path):
        import mne
        # 从EDF读取通道与采样率，不假定64或128通道。MNE以伏特返回EEG。
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        try:
            if raw.times[-1] > 600:
                raise ValueError("学习版暂只支持不超过10分钟的脑电文件")
            raw.pick("eeg")
            if len(raw.ch_names) == 0:
                raise ValueError("EDF中没有EEG通道")
            spectrum = raw.compute_psd(method="welch", fmin=1,
                                       fmax=min(45, raw.info["sfreq"] / 2), verbose="ERROR")
            psd, freqs = spectrum.get_data(return_freqs=True)
            bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
                     "beta": (13, 30), "gamma": (30, 45)}
            values = []
            for low, high in bands.values():
                mask = (freqs >= low) & (freqs < high)
                if not mask.any():
                    raise ValueError("采样率过低，不能提取示例配置的全部频带")
                values.append(float(np.log10(max(float(psd[:, mask].mean()), 1e-30))))
            return ModalityFeature(values=values, names=list(bands), metadata={
                "method": "Welch PSD在频带及通道上取均值，再取log10",
                "psd_unit": "V^2/Hz", "sample_rate_hz": raw.info["sfreq"],
                "channel_count": len(raw.ch_names), "channel_names": raw.ch_names,
                "duration_seconds": raw.n_times / raw.info["sfreq"],
            })
        finally:
            raw.close()

    def extract_audio(self, path):
        import librosa
        import soundfile as sf
        info = sf.info(str(path))
        if info.duration > 600:
            raise ValueError("学习版暂只支持不超过10分钟的音频")
        y, sr = librosa.load(str(path), sr=16000, mono=True)
        if len(y) < 2048 or not np.isfinite(y).all():
            raise ValueError("音频过短或含无效样本")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = float(np.sqrt(np.mean(y ** 2)))
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        return ModalityFeature(values=[*mfcc.tolist(), rms, zcr],
                               names=[f"mfcc_{i}" for i in range(13)] + ["rms", "zcr"],
                               metadata={"sample_rate_hz": sr, "duration_seconds": len(y) / sr,
                                         "method": "13维MFCC时间均值、RMS和过零率"})

    def extract_face(self, path):
        import mediapipe as mp
        from PIL import Image
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = self.settings.path(self.settings.options["features"]["face_model"])
        if not model_path.is_file():
            raise ValueError("缺少人脸模型，请先执行 python scripts/download_resources.py")
        # 模型仅在后台线程首次使用时加载一次。未来换成队友AU模型即可。
        if self.face is None:
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                output_face_blendshapes=True, num_faces=1,
            )
            self.face = vision.FaceLandmarker.create_from_options(options)
        with Image.open(path) as picture:
            picture.thumbnail((1280, 1280))
            pixels = np.asarray(picture.convert("RGB"))
        result = self.face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels))
        if not result.face_blendshapes:
            raise ValueError("没有检测到人脸，请使用清晰的正面人脸图片")
        scores = sorted(result.face_blendshapes[0], key=lambda item: item.category_name)
        return ModalityFeature(values=[x.score for x in scores], names=[x.category_name for x in scores],
                               metadata={"method": "MediaPipe Face Landmarker blendshapes",
                                         "note": "blendshape系数并非FACS AU或情绪概率"})

    def extract_text(self, text):
        mode = self.settings.options["features"]["text_mode"]
        if mode == "simple":
            return ModalityFeature(
                values=[float(len(text)), float(sum("\u4e00" <= c <= "\u9fff" for c in text)),
                        float(text.count("！") + text.count("!")),
                        float(text.count("？") + text.count("?"))],
                names=["characters", "chinese_characters", "exclamations", "questions"],
                metadata={"method": "文本统计占位", "is_placeholder": True},
            )
        if mode != "minilm":
            raise ValueError("features.text_mode 只能为 simple 或 minilm")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("请先安装 requirements-text.txt 并下载文本模型") from exc
        path = self.settings.path(self.settings.options["features"]["text_model"])
        if not path.is_dir():
            raise ValueError("请执行 python scripts/download_resources.py --text 下载文本模型")
        if self.text_encoder is None:
            self.text_encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True)
        vector = self.text_encoder.encode(text)
        return ModalityFeature(values=vector.tolist(), metadata={
            "method": "paraphrase-multilingual-MiniLM-L12-v2", "is_placeholder": False,
            "note": "语义向量，不是情绪概率；超出模型最大长度的文本将被截断",
        })

    def close(self):
        if self.face is not None:
            self.face.close()
