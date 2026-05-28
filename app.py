from flask import Flask, request, send_file, render_template, jsonify
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import os
import uuid
import threading
import subprocess
import imageio_ffmpeg

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def convert_to_8d(input_file, output_file_mp3, job_id):
    try:
        jobs[job_id] = 'processing'

        # Load audio at original sample rate
        data, sr = librosa.load(input_file, sr=None, mono=False)

        # Ensure shape is (2, samples)
        if data.ndim == 1:
            data = np.vstack([data, data])
        elif data.shape[0] != 2:
            data = data.T

        n = data.shape[1]

        # Mix to mono — clean source
        mono = (data[0] + data[1]) / 2.0
        t = np.arange(n) / sr

        # ── ROTATION ──────────────────────────────────────────
        rot_freq = 0.07  # Hz — one full rotation every ~14 seconds (slow dreamy)

        # Left-right panning (infinity shape)
        pan_lr = np.sin(2 * np.pi * rot_freq * t)

        # Up-down elevation (double frequency = figure 8 / ∞ shape)
        elevation = np.sin(2 * np.pi * rot_freq * 2 * t)

        # ── VOLUME PANNING ────────────────────────────────────
        # Minimum 10% — sound NEVER fully cuts off
        min_v   = 0.10
        left_g  = min_v + (1.0 - min_v) * (1.0 - pan_lr) / 2.0
        right_g = min_v + (1.0 - min_v) * (1.0 + pan_lr) / 2.0

        left_ch  = mono * left_g
        right_ch = mono * right_g

        # ── REVERB ────────────────────────────────────────────
        # Multi-tap delay reverb — gives depth and space
        left_rev  = left_ch.copy()
        right_rev = right_ch.copy()

        delay_times = [0.022, 0.035, 0.055, 0.089]
        decay_vals  = [0.40,  0.28,  0.18,  0.10 ]

        for delay_s, decay in zip(delay_times, decay_vals):
            d = int(delay_s * sr)
            if d < n:
                left_rev[d:]  += left_ch[:n - d]  * decay
                right_rev[d:] += right_ch[:n - d] * decay

        # ── ELEVATION EFFECT ──────────────────────────────────
        # Above/below head feeling using high frequency modulation
        # Human ears detect elevation through pinna filtering of high freqs
        sos_high = butter(2, min(8000, sr//2 - 1) / (sr / 2),
                          btype='high', output='sos')
        high_l = sosfilt(sos_high, left_rev)
        high_r = sosfilt(sos_high, right_rev)

        low_l = left_rev  - high_l
        low_r = right_rev - high_r

        # Boost highs when above head, cut when below
        elev_gain = 1.0 + 0.18 * elevation

        final_l = low_l + high_l * elev_gain
        final_r = low_r + high_r * elev_gain

        # ── OUTPUT ────────────────────────────────────────────
        output = np.vstack([final_l, final_r]).T  # (n, 2)

        # Clean normalize — no distortion
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.88

        output = output.astype(np.float32)

        # Save temp WAV
        temp_wav = output_file_mp3.replace('.mp3', '_temp.wav')
        sf.write(temp_wav, output, sr)

        # Convert WAV → MP3 (high quality)
        subprocess.run([
            FFMPEG, '-y',
            '-i', temp_wav,
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',
            output_file_mp3
        ], check=True, capture_output=True)

        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        if os.path.exists(input_file):
            os.remove(input_file)

        jobs[job_id] = 'done'

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id] = 'error'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    if 'song' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['song']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    job_id   = str(uuid.uuid4())
    filename = file.filename.lower()
    ext      = '.mp3' if filename.endswith('.mp3') else '.wav'
    input_path  = os.path.join(UPLOAD_FOLDER, f'{job_id}{ext}')
    output_path = os.path.join(OUTPUT_FOLDER,  f'{job_id}_8d.mp3')

    file.save(input_path)
    jobs[job_id] = 'queued'

    threading.Thread(
        target=convert_to_8d,
        args=(input_path, output_path, job_id),
        daemon=True
    ).start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    return jsonify({'status': jobs.get(job_id, 'not_found')})


@app.route('/download/<job_id>')
def download(job_id):
    path = os.path.join(OUTPUT_FOLDER, f'{job_id}_8d.mp3')
    if os.path.exists(path):
        return send_file(path, as_attachment=True,
                         download_name='song_8d.mp3')
    return jsonify({'error': 'File not ready'}), 404


if __name__ == '__main__':
    app.run(debug=True, threaded=True)