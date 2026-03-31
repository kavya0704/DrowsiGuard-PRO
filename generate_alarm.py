"""
Generate alarm sound WAV file programmatically.
Run once: python generate_alarm.py
"""
import wave
import struct
import math
import os

def generate_alarm(filename="static/sounds/alarm.wav", duration=3.0, sample_rate=44100):
    """Generate an attention-grabbing alternating frequency alarm tone."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    n_samples = int(duration * sample_rate)
    samples = []
    
    for i in range(n_samples):
        t = i / sample_rate
        
        # Alternating between two frequencies every 0.15 seconds (urgent feel)
        cycle = t % 0.3
        if cycle < 0.15:
            freq = 880   # A5 - high tone
        else:
            freq = 660   # E5 - lower tone
        
        # Add slight frequency modulation for urgency
        freq_mod = freq + 30 * math.sin(2 * math.pi * 8 * t)
        
        # Generate sine wave with envelope
        envelope = min(1.0, t * 10) * min(1.0, (duration - t) * 10)  # Fade in/out
        amplitude = 0.7 * envelope
        
        sample = amplitude * math.sin(2 * math.pi * freq_mod * t)
        
        # Add harmonic for richer sound
        sample += 0.2 * amplitude * math.sin(2 * math.pi * freq_mod * 2 * t)
        
        # Clip and convert to 16-bit integer
        sample = max(-1.0, min(1.0, sample))
        samples.append(int(sample * 32767))
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)         # 16-bit
        wav_file.setframerate(sample_rate)
        for s in samples:
            wav_file.writeframes(struct.pack('<h', s))
    
    print(f"[OK] Alarm sound generated: {filename} ({duration}s, {sample_rate}Hz)")

if __name__ == "__main__":
    generate_alarm()
