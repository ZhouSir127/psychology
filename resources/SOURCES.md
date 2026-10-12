# 公开资源与用途

本目录资源用于工程联调，不能合并为真实配对训练数据。下载脚本在manifest.json中记录原始链接、文件大小和SHA256；音频另与librosa官方登记摘要比对。其余摘要仅记录本次下载，不代表独立签名验证。

| 资源 | 官方来源 | 用途与限制 |
| --- | --- | --- |
| S001R01.edf、S001R02.edf | https://physionet.org/content/eegmmidb/1.0.0/ | 64通道、160Hz、睁眼/闭眼基线，测试读取及PSD；不含本项目心理风险标签 |
| 5703-47212-0000.ogg | https://librosa.org/data/audio/5703-47212-0000.ogg | librosa的libri1朗读样例，测试MFCC/RMS；不是情绪标签语音 |
| 人脸示例business-person.png | https://storage.googleapis.com/mediapipe-assets/business-person.png | MediaPipe官方教学例图，用于验证人脸处理 |
| face_landmarker.task | https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker | 官方模型版本1，输出landmarks及blendshapes；不等于FACS AU或心理诊断 |
| 多语言MiniLM（可选） | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 中文语义向量，不输出情绪或风险，约500MB相关文件 |
| 模拟评分 | 项目自带app/adapters/predictor.py | 场景分数10/30/50/70/90，用于功能测试，无训练或预测有效性声明 |

## 授权与引用

- EEG数据页标注Open Data Commons Attribution License v1.0。引用Schalk (2009), EEG Motor Movement/Imagery Dataset, DOI: 10.13026/C28G6P，并保留数据页列出的原始论文归属。
- librosa样例来源描述为The Ashiel Mystery, chapter XVI，朗读者Garth Comira。查看具体音频归属可使用 `librosa.example_info('libri1')`；代码库的许可证不自动覆盖全部样例音频。
- MediaPipe代码示例采用Apache-2.0；教程标明人脸照片来自 https://unsplash.com/photos/mt2fyrdXxzk 。模型、照片的使用应分别遵守来源条款，不能统一宣称均由本项目原创。
- MiniLM模型卡标注Apache-2.0。
- 下载地址来自官方资源页/样例代码。后续公开软件时保留本来源文件，并按各资源许可决定是否再分发二进制资源。

## 参考代码

- MNE EDF读取：https://mne.tools/stable/generated/mne.io.read_raw_edf.html
- librosa：https://github.com/librosa/librosa
- MediaPipe Python示例：https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/python
- 本地模型下载：https://huggingface.co/docs/huggingface_hub/guides/download
- PyTorch保存加载：https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
