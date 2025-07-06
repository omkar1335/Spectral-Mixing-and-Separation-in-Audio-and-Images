import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import welch

# Load audio
audio_path = r'C:\Users\bsomk\Downloads\song_with_2piccolo.wav'  # Replace with your actual file path
y, sr = librosa.load(audio_path, sr=None)
print(f"Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f} s")

# Normalize
y = y / np.max(np.abs(y))

# Power Spectral Density
f_psd, Pxx = welch(y, sr, nperseg=2048)
plt.figure(figsize=(10, 4))
plt.semilogy(f_psd, Pxx)
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.tight_layout()
plt.show()

# Frequency bands to remove (manually identified from spectrogram)
bands_to_remove = [
    (1000, 1800), (2300, 5300), (5800, 6250), (6400, 6500), (6475, 6525),
    (6900, 7100), (7300, 7925), (8500, 9000), (9305, 9805), (10400, 10580),
    (10800, 11000), (11500, 12000), (12180, 12240)
]

# FFT
N = len(y)
yf = np.fft.fftshift(np.fft.fft(y))
freq = np.fft.fftshift(np.fft.fftfreq(N, 1/sr))

# Band-stop filter
H = np.ones(N)
for band in bands_to_remove:
    H[np.logical_and(np.abs(freq) >= band[0], np.abs(freq) <= band[1])] = 0

# Apply filter
yf_filtered = yf * H
y_filtered = np.fft.ifft(np.fft.ifftshift(yf_filtered)).real
y_filtered = y_filtered / np.max(np.abs(y_filtered))  # Normalize

# Save cleaned audio
sf.write("restored_audio.wav", y_filtered, sr)
print("Filtered audio saved as 'restored_audio.wav'")

# Bode-style magnitude response (only positive frequencies)
plt.figure(figsize=(10, 4))
plt.plot(freq[freq >= 0], 20 * np.log10(np.abs(H[freq >= 0]) + 1e-6))
plt.title("Filter Gain (Bode-style plot)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plot original spectrogram ----------
D_orig = librosa.stft(y)
S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db_orig, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Original Audio Spectrogram")
plt.tight_layout()
plt.show()

# ---------- Plot filtered spectrogram ----------
D_filt = librosa.stft(y_filtered)
S_db_filt = librosa.amplitude_to_db(np.abs(D_filt), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db_filt, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Filtered Audio Spectrogram")
plt.tight_layout()
plt.show()
