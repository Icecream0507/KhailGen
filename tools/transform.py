import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# 加载双声道音频
y, sr = librosa.load(r"data\wav\01. 爱爱爱.wav", sr=None, mono=False)  # y.shape = (2, samples)

# === 修改点：分别处理左右声道 ===
# 左声道
y_left = y[0]
# 右声道
y_right = y[1]

n_fft = 2048
hop_length = 512

# 傅里叶变换：左声道
stft_left = librosa.stft(y_left, n_fft=n_fft, hop_length=hop_length)
magnitude_left = np.abs(stft_left)
phase_left = np.angle(stft_left)

# 傅里叶变换：右声道
stft_right = librosa.stft(y_right, n_fft=n_fft, hop_length=hop_length)
magnitude_right = np.abs(stft_right)
phase_right = np.angle(stft_right)

# # 从左声道的幅度谱计算梅尔频谱图（用于展示）
# n_mels = 128
# mel_spec = librosa.feature.melspectrogram(
#     S=librosa.amplitude_to_db(magnitude_left),
#     sr=sr,
#     n_mels=n_mels,
#     fmax=sr//2
# )

with open('magnitude_left.npy', 'wb') as f:
    np.save(f, magnitude_left)
with open('phase_left.npy', 'wb') as f:
    np.save(f, phase_left)

# # 绘图部分保持不变，只展示左声道的频谱
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# librosa.display.specshow(magnitude_left, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram (Left Channel)')

# plt.subplot(1, 2, 2)
# librosa.display.specshow(phase_left, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(label='Phase (radians)')
# plt.title('Phase Spectrum (Left Channel)')
# plt.tight_layout()

# plt.savefig('mel_spectrogram_left.png')



def reconstruct_from_stft(magnitude, phase, sr, hop_length):
    """
    使用傅里叶变换的幅度谱和相位谱进行精确重建。
    """
    stft = magnitude * np.exp(1j * phase)
    return librosa.istft(stft, hop_length=hop_length)


# === 修复的关键点在这里！ ===
# 分别重建左右声道
y_reconstructed_left = reconstruct_from_stft(magnitude_left, phase_left, sr, hop_length)
y_reconstructed_right = reconstruct_from_stft(magnitude_right, phase_right, sr, hop_length)

# 将重建后的左右声道合并成一个双声道数组
y_reconstructed = np.vstack([y_reconstructed_left, y_reconstructed_right])

# 保存重建的音频
sf.write('reconstructed.wav', y_reconstructed.T, sr)
