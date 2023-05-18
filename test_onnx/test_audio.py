
import pyaudio
import wave
 
CHUNK = 1024
audio_file = '../vits_infer_out/bert_vits_stream1.wav'
wf = wave.open(audio_file, 'rb')
p = pyaudio.PyAudio()
print("sampwidth() ",wf.getsampwidth())
print("channels() ",wf.getnchannels())
print("framerate() ",wf.getframerate())
print("format_from_width() ",p.get_format_from_width(wf.getsampwidth()))

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
 
data = wf.readframes(CHUNK)
while data != b"":
    stream.write(data)
    data = wf.readframes(CHUNK)
 
stream.stop_stream()
stream.close()
p.terminate()