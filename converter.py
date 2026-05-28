import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import soundfile as sf

def convert_to_8d(input_file, output_file):
    print("Loading song...")
    data, samplerate = sf.read(input_file)
    
    # Make sure audio is stereo
    if len(data.shape) == 1:
        data = np.column_stack([data, data])
    
    print("Applying 8D effect...")
    total_samples = len(data)
    output = np.zeros_like(data)
    
    # Rotation speed - slow dreamy rotation
    rotation_speed = 0.08  # lower = slower rotation
    
    for i in range(total_samples):
        # Calculate angle for this sample
        angle = (i / samplerate) * rotation_speed * 2 * np.pi
        
        # Smooth panning using sine wave
        left_gain  = (1 + np.sin(angle)) / 2
        right_gain = (1 - np.sin(angle)) / 2
        
        # Mix both channels with gains
        mono = (data[i][0] + data[i][1]) / 2
        output[i][0] = mono * left_gain
        output[i][1] = mono * right_gain
    
    # Add reverb effect for depth
    print("Adding reverb and depth...")
    reverb_delay = int(samplerate * 0.03)
    reverb_decay = 0.4
    for i in range(reverb_delay, total_samples):
        output[i] += output[i - reverb_delay] * reverb_decay
    
    # Normalize to avoid distortion
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.95
    
    print("Saving 8D song...")
    sf.write(output_file, output, samplerate)
    print("Done! Your 8D song is ready!")

# Run it
convert_to_8d("song.wav", "song_8d.wav")