"""
 * @author: zkyuan
 * @date: 2025/3/8 23:04
 * @description: speech to text
"""
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.utils import download_asset

# 测试环境，查看版本
print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# 加载音频
SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
# C:\Users\HP\.cache\torch\hub\torchaudio\tutorial-assets\Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav
# print(SPEECH_FILE)
# 播放音频
from playsound import playsound

print("-----------------正在播放音频-----------------")
playsound(SPEECH_FILE)

# 创建管道
# 创建一个 Wav2Vec2 模型，该模型执行特征提取和分类
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# 模型加载的默认位置为--> C:\Users\HP\.cache\torch\hub\checkpoints
print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())
"""词库
采样率Sample Rate: 16000
类别标签Labels: ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
"""

# 此过程将自动获取预训练权重并将其加载到模型中
model = bundle.get_model().to(device)

print(model.__class__)  # <class 'torchaudio.models.wav2vec2.model.Wav2Vec2Model'>


# 这里报错 RuntimeError: Couldn't find appropriate backend to handle uri data\Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav and format None.
# 缺少音频库，pip install pysoundfile，运行这个解决
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)
# 如果采样率与管道期望的不同，那么我们可以使用 torchaudio.functional.resample() 进行重采样
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# 提取声学特征
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

# 返回的特征是张量列表。每个张量都是 transformer 层的输出
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from transformer layer {i + 1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()

# 特征分类
with torch.inference_mode():
    emission, _ = model(waveform)

# 将其可视化
plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())


# 生成文本记录
class GreedyCTCDecoder(torch.nn.Module):
    """定义贪婪解码算法"""

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


# 创建解码器对象并解码文本记录
decoder = GreedyCTCDecoder(labels=bundle.get_labels())

transcript = decoder(emission[0])
# 检查结果
print(transcript)
print("-----------------再次播放音频-----------------")
playsound(SPEECH_FILE)

"""all

2.6.0+cu124
2.6.0+cu124
cuda
-----------------正在播放音频-----------------
Sample Rate: 16000
Labels: ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
<class 'torchaudio.models.wav2vec2.model.Wav2Vec2Model'>
Class labels: ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|
-----------------再次播放音频-----------------

"""