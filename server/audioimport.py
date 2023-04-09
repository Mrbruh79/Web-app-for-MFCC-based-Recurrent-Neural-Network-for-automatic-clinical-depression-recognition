from pydub import AudioSegment


audio_file = "300_AUDIO.wav"
# import wave
# samplerate, data = wavfile.read('./300_AUDIO.wav')
audio_segment = AudioSegment.from_wav(audio_file)