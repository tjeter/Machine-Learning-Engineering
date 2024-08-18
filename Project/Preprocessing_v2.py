import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from librosa import core, onset, feature, display
import soundfile as sf
from IPython.display import Audio
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.utils import shuffle

df = pd.read_csv("data/birdsong_metadata.csv")
df.head()

def load_audio(file_id):
    data, samplerate = sf.read("data/songs/xc"+str(file_id)+".flac")
    # data, samplerate = sf.read("data/songs/songs/xc"+str(file_id)+".flac")
    s = len(data)/samplerate
    sg = feature.melspectrogram(y = data, sr=samplerate, hop_length=512)

    # Take mean amplitude M from frame with highest energy
    centerpoint = np.argmax(sg.mean(axis=0))
    M = sg[:,centerpoint].mean()

    # Filter out all frames with energy less than 5% of M
    mask = sg.mean(axis=0)>=M/20

    audio_mask = np.zeros(len(data), dtype=bool)
    for i in range(0,len(mask)):
        audio_mask[i*512:] = mask[i]
    return sg, mask, data, audio_mask, samplerate

df['length'] = np.zeros(len(df))

waves = {}

for file_id in df['file_id']:
    sg, mask, data, audio_mask, sample_rate = load_audio(file_id)
    waves[file_id] = data[audio_mask]
    df.loc[df['file_id'] == file_id,'length'] = len(data[audio_mask])
    # print(len(data[audio_mask])/sample_rate)


# We set window to 6.144000e+03 frames as it's the minimum length among our audio files
df['windows'] = df['length'].apply(lambda x: int(x/6.144000e+03))
df['length'].hist()
plt.show()
df['length'].describe()


counts = [list(df.genus).count(code) for code in set(df.genus)]
idx = np.argsort(counts)
y = np.array(counts)[idx]
code_dict = np.array(list(set(df.genus)))[idx]
fig, ax = plt.subplots(figsize=(12,15))
ax.barh(code_dict, y, height=0.75, color="slateblue")


# ## Understanding melspectrogram
# 
# To understand the melspectrogram, one needs to first understand the fourier transform. In simple terms, it is a transformation that allows us to analyze the frequency content of a signal effectively. But unfortunately, the fourier transform requires the signal to be periodic to get optimal results.
# 
# This problem can be solved by applying the fourier transform on multiple windowed segments of the signals to capture the frequency content as it changes over time. This is called the short-time fourier transform or STFT for short. When the fourier transform is applied on overlapping windowed segments of the signal, what we get what is called a spectrogram.
# 
# A spectrogram can be thought of as several fourier transforms stacked on top of each other. It is a way to visually represent a signal’s loudness, or amplitude, as it varies over time at different frequencies. The y-axis is converted to a log scale, and the color dimension is converted to decibels. This is because humans can only perceive a very small and concentrated range of frequencies and amplitudes.
# 
# Moreover, studies have shown that humans do not perceive frequencies on a linear scale. We are better at detecting differences in lower frequencies than higher frequencies. For example, we can easily tell the difference between 500 and 1000 Hz, but we will hardly be able to tell a difference between 10,000 and 10,500 Hz, even though the distance between the two pairs are the same. In 1937, Stevens, Volkmann, and Newmann proposed a unit of pitch such that equal distances in pitch sounded equally distant to the listener. This is called the mel scale. We perform a mathematical operation on frequencies to convert them to the mel scale.
# 
# A spectrogram converted to the mel scale is called a melspectrogram and is a great way to convert audio signal data to a visual feature map to train an image model (like ResNet-34) on. Refer to this article for an excellent explanation of melspectrogram. Now, since we understand how the melspectrogram works, let us move on to the modelling part
# 
# ## Visualize melspectrogram
# 
# Next we visualize the melspectrogram feature maps for sample signals to get an better understanding. We can see that the melspectrogram contains visual information about the trends (frequency and amplitude) in the audio signal over time.
# 

# Assuming you have some audio data in 'y' and 'sr' (sampling rate)
y, sr = librosa.load("data/songs/xc"+str(file_id)+".flac")
# y, sr = librosa.load("data/songs/songs/xc"+str(file_id)+".flac")


def show_melspectrogram(data, samplerate):
    # Compute the Mel spectrogram
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=samplerate)

    # Convert power spectrogram to dB
    melspectrogram_db = librosa.power_to_db(melspectrogram)

    # Normalize the Mel spectrogram to [0, 255]
    normalized_melspectrogram = librosa.util.normalize(melspectrogram_db)

    # Convert to image representation
    image = librosa.display.specshow(normalized_melspectrogram, cmap='viridis')

    # Plot or use the image for further processing
    plt.show()



def get_melspectrogram(data, samplerate):
    # Compute the Mel spectrogram
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=samplerate)

    # Convert power spectrogram to dB
    melspectrogram_db = librosa.power_to_db(melspectrogram)

    # Normalize the Mel spectrogram to [0, 255]
    normalized_melspectrogram = librosa.util.normalize(melspectrogram_db)

    return normalized_melspectrogram


# visualize one example of spectrogram
file_id = 132608
data, samplerate = sf.read("data/songs/xc"+str(file_id)+".flac")
# data, samplerate = sf.read("data/songs/songs/xc"+str(file_id)+".flac")
show_melspectrogram(data, samplerate)
get_melspectrogram(data, samplerate)