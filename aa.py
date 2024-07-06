import librosa
import numpy as np


def load_audio_file(file_path):
    # 加载音频文件
    y, sr = librosa.load(file_path, sr=None)  # sr=None表示使用音频文件本身的采样率

    # 你可以选择对音频数据进行一些预处理，比如归一化
    y_normalized = y / np.max(np.abs(y))  # 归一化到[-1, 1]

    # 对于机器学习，你可能需要将音频数据转换成某种形式的特征。
    # 这里我们简单地使用音频的原始数据作为特征，但在实践中可能需要更复杂的特征提取。
    # 例如，你可以使用MFCCs（Mel频率倒谱系数）或其他音频特征。
    # mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=40)

    # 如果使用原始音频数据，可以将其视为时间序列数据
    # 这里假设每个样本点作为一个特征，每个时间点的样本作为一个样本
    # 注意：这种表示方式可能不适合所有类型的机器学习模型
    # 对于某些模型（如RNNs），你可能需要保持时间序列的完整性

    # 假设我们直接使用归一化后的音频数据
    return y_normalized, sr


from pathlib import Path

# 指定要读取的目录路径
directory_path = Path('./humbugdb_neurips_2021_1')
audio_path = []
# 使用Path对象的iterdir()方法遍历目录下的所有项
for entry in directory_path.iterdir():
    # 使用is_file()方法检查当前项是否为文件
    if entry.is_file():
        audio_path.append(entry)
        print(entry.name)  # entry.name获取文件名（不包含路径）
# print(len(audio_path))

# 使用函数
file_path = 'humbugdb_neurips_2021_1/53.wav'  # 替换为你的音频文件路径
audio_data, sample_rate = load_audio_file(file_path)

# 现在audio_data包含了音频数据，sample_rate是采样率
# 你可以根据需要进一步处理这些数据
# print(audio_data)
#
# print(sample_rate)