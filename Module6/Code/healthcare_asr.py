#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
from soundfile import SoundFile

from jetson_voice import ASR, TTS, QuestionAnswer, AudioInput, AudioOutput, ConfigArgParser

# 1. Setup arguments
parser = ConfigArgParser()
parser.add_argument('--wav', required=True, type=str, help='path to input wav/ogg/flac file with the user\'s spoken question')
parser.add_argument('--asr-model', default='quartznet', type=str, help='path to ASR model')
parser.add_argument('--model', default='distilbert_qa_384', type=str, help='path to QA model')
parser.add_argument('--top_k', default=1, type=int, help='show the top N answers (default 1)')
parser.add_argument('--output-device', default=None, type=str, help='audio output device name or number')
parser.add_argument('--output-wav', default=None, type=str, help='directory or file path to save output wavs')
parser.add_argument('--warmup', default=5, type=int, help="Number of warmups for TTS")

args = parser.parse_args()

if not os.path.exists(args.wav):
    print(f"Error: Input wav file '{args.wav}' not found.")
    sys.exit(1)

print("Initializing Jetson Voice models (ASR, QA, and TTS)...")
asr = ASR(args.asr_model)
qa_model = QuestionAnswer(args.model)
tts = TTS("fastpitch_hifigan")

medical_context = (
    "Type 2 diabetes is a chronic condition affecting how the body metabolizes glucose. "
    "The main symptoms of type two diabetes include increased thirst, frequent urination, fatigue, and blurred vision. "
    "Treatment focuses on diet, exercise, and blood sugar monitoring. "
    "During a sudden asthma attack, standard clinical protocol states that patients should sit upright, "
    "use a rescue inhaler immediately, and seek emergency care if symptoms do not improve. "
    "Avoid lying down as it restricts airway capacity. "
    "The primary goal of a medical triage chatbot is to assess the severity of symptoms to route patients "
    "to the correct care level. It provides automated guidance based on established clinical assessment protocols."
)

audio_device = None
if args.output_device:
    audio_device = AudioOutput(args.output_device, tts.sample_rate)

if args.output_wav:
    wav_is_dir = len(os.path.splitext(args.output_wav)[1]) == 0
    if wav_is_dir and not os.path.exists(args.output_wav):
        os.makedirs(args.output_wav)

try:
    print(f"\nReading and transcribing audio question file: {args.wav}")
    stream = AudioInput(wav=args.wav, sample_rate=asr.sample_rate, chunk_size=asr.chunk_size)
    user_question = ""

    for samples in stream:
        results = asr(samples)
        if not asr.classification:
            for transcript in results:
                user_question += transcript["text"]

    print(f"Transcribed Question: {user_question}")

    if not user_question.strip():
        print("ASR did not detect any text or question in the audio file. Exiting.")
        sys.exit(0)

    print("\nProcessing question with QA Model...")
    query = {
        'context': medical_context,
        'question': user_question
    }
    
    qa_results = qa_model(query, top_k=args.top_k)
    
    if args.top_k == 1:
        qa_results = [qa_results]
        
    response_text = qa_results[0]['answer']
    print(f"Extracted Answer: {response_text}")
    print(f"Confidence Score: {qa_results[0]['score']:.4f}")

    print("\nGenerating TTS voice output...")
    audio = None
    for run in range(args.warmup + 1):
        start = time.perf_counter()
        audio = tts(response_text)
        stop = time.perf_counter()
        
        latency = stop - start
        duration = audio.shape[0] / tts.sample_rate
        
        if run == args.warmup:
            print(f"TTS Latency: {latency:.3f}s | Audio Duration: {duration:.2f}s | RTFx: {duration/latency:.2f}x")

    if audio_device and audio is not None:
        print("Playing response audio...")
        audio_device.write(audio)

    if args.output_wav and audio is not None:
        wav_path = os.path.join(args.output_wav, 'output.wav') if wav_is_dir else args.output_wav
        wav = SoundFile(wav_path, mode='w', samplerate=tts.sample_rate, channels=1)
        wav.write(audio)
        wav.close()
        print(f"Wrote TTS audio to {wav_path}")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Exiting.")
    sys.exit(0)