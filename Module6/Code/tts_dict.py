import os
import sys
import time
import readline
import requests

from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile

parser = ConfigArgParser()

parser.add_argument('--model', default='fastpitch_hifigan', type=str)
parser.add_argument('--warmup', default=5, type=int, help='number of warmup runs')
parser.add_argument('--output-device', default=None, type=str, help='audio out')
parser.add_argument('--output-wav', default=None, type=str, help='output directory')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)

if args.list_devices:
    list_audio_devices()
    sys.exit()

tts = TTS(args.model)

if args.output_device:
    audio_device = AudioOutput(args.output_device, tts.sample_rate)

if args.output_wav:
    wav_is_dir = len(os.path.splitext(args.ouptut_wav)[1]) == 0
    wav_count = 0
    if wav_is_dir and not os.path.exists(args.output_wav):
        os.makedirs(args.output_wav)

while True:
    print(f'\nEnter text, or Q to quit:')
    input_text = input('> ')

    if input_text.upper() == 'Q':
        sys.exit()

    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{text}"
    response = requests.get(url)

    all_meanings = ""

    if response.status_code == 200:
        data = response.json()[0]
            
        for meaning in data.get("meanings", []):
            part_of_speech = meaning.get("partOfSpeech", "unknown")
                
                
            definitions = [d["definition"] for d in meaning.get("definitions", [])]
            joined_definitions = "\n".join(definitions)
                
            all_meanings += f" --- {part_of_speech} ---\n{joined_definitions}\n\n"
    else:
        all_meanings = f"Sorry, I could not find any meanings for the word {text}."

    print(all_meanings)

    print('')

    for run in range(args.warmup+1):
        start = time.perf_counter()
        audio = tts(all_meanings)
        stop = time.perf_counter()
        latency = stop-start
        duration = audio.shape[0]/tts.sample_rate
        print(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:2f}s of audio. RTFx={duration/latency:.2f}.")

    if args.output_device:
        audio_device.write(audio)

    if args.output_wav:
        wav_path = os.path.join(args.output_wav, f'{wav_count}.wav') if wav_is_dir else args.output_wav
        wav = SoundFile(wav_path, mode='w', samplerate=tts.sample_rate, channels=1)
        wav.write(audio)
        wav.close()
        wav_count += 1
        print(f"\nWrote audio to {wav_path}")
