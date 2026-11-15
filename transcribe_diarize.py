"""
Interactive Meeting Transcription & Diarization CLI Tool
========================================================
Production command-line interface for multi-speaker speech-to-text
transcription with speaker attribution ("who said what").

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import soundfile as sf
import torch
import torchaudio.transforms as T

from main import (
    Wav2Vec2ASREngine,
    NeuralDiarizerEngine,
    SileroVADSegmenter,
    ASRDiarizationPipeline,
    SAMPLE_RATE,
    VAD_MODEL_PATH,
    OUTPUTS_DIR
)

def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def format_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds % 60)
    m = int((seconds // 60) % 60)
    h = int(seconds // 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    parser = argparse.ArgumentParser(description="Multi-Speaker Transcription & Diarization (ASR + Diarization)")
    parser.add_argument("--audio", type=str, help="Path to input meeting WAV audio file")
    parser.add_argument("--num-speakers", type=int, default=3, help="Expected number of speakers (default 3)")
    parser.add_argument("--threshold", type=float, default=0.50, help="VAD speech trigger threshold (default 0.50)")
    parser.add_argument("--export-txt", type=str, help="Path to export formatted text transcript")
    parser.add_argument("--export-json", type=str, help="Path to export structured JSON turns")
    parser.add_argument("--export-rttm", type=str, help="Path to export NIST RTTM diarization")
    parser.add_argument("--export-srt", type=str, help="Path to export SubRip (SRT) subtitle file")

    args = parser.parse_args()

    audio_path = args.audio
    if not audio_path:
        default_path = os.path.join(OUTPUTS_DIR, "meeting_session.wav")
        if os.path.exists(default_path):
            audio_path = default_path
        else:
            parser.print_help()
            sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("      MULTI-SPEAKER TRANSCRIPTION & DIARIZATION (WHO SAID WHAT)")
    print("=" * 72)
    print(f"Input Audio File : {audio_path}")
    print(f"Target Speakers  : {args.num_speakers}")
    print(f"VAD Threshold    : {args.threshold:.2f}")

    # Load audio
    wav, sr = sf.read(audio_path)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)
    if wav_tensor.ndim > 1:
        wav_tensor = wav_tensor.mean(dim=-1)
    if sr != SAMPLE_RATE:
        wav_tensor = T.Resample(sr, SAMPLE_RATE)(wav_tensor.unsqueeze(0)).squeeze(0)

    duration_s = len(wav_tensor) / SAMPLE_RATE
    print(f"Audio Duration   : {duration_s:.2f} seconds ({len(wav_tensor):,} samples)")
    print("-" * 72)

    # Initialize engines
    print("Loading models (Silero VAD + Wav2Vec2 CTC + SpeakerEmbeddingNet)...")
    vad_segmenter = SileroVADSegmenter(VAD_MODEL_PATH)
    asr_engine = Wav2Vec2ASREngine()
    diarizer_engine = NeuralDiarizerEngine(num_speakers=args.num_speakers)

    pipeline = ASRDiarizationPipeline(asr_engine, diarizer_engine, vad_segmenter)

    # Process pipeline
    t_start = time.perf_counter()
    aligned_turns, embeddings, timing = pipeline.process(wav_tensor, num_speakers=args.num_speakers)
    total_time = time.perf_counter() - t_start
    rtf = total_time / duration_s if duration_s > 0 else 0.0

    print("\n" + "=" * 72)
    print("       SPEAKER-ATTRIBUTED TRANSCRIPT ('WHO SAID WHAT')")
    print("=" * 72)

    transcript_lines = []
    for turn in aligned_turns:
        st_str = format_timestamp(turn["start"])
        et_str = format_timestamp(turn["end"])
        line = f"[{st_str} - {et_str}] {turn['speaker_id']}: {turn['text']}"
        transcript_lines.append(line)
        print(line)

    print("-" * 72)
    print("PIPELINE SUMMARY & BENCHMARK:")
    print(f"  • Dialogue Turns Diarized : {len(aligned_turns)}")
    print(f"  • Distinct Speakers Found : {len(set(t['speaker_id'] for t in aligned_turns))}")
    print(f"  • Total Audio Duration    : {duration_s:.2f} s")
    print(f"  • Total Execution Time    : {total_time:.2f} s (RTF: {rtf:.4f}x, {1.0/rtf:.1f}x speedup)")
    print("=" * 72)

    # Export formats
    if args.export_txt:
        with open(args.export_txt, "w") as f:
            f.write("\n".join(transcript_lines) + "\n")
        print(f"Plaintext transcript exported to: {args.export_txt}")

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(aligned_turns, f, indent=2)
        print(f"JSON turns exported to: {args.export_json}")

    if args.export_rttm:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        with open(args.export_rttm, "w") as f:
            for turn in aligned_turns:
                f.write(f"SPEAKER {base_name} 1 {turn['start']:.3f} {turn['duration']:.3f} <NA> <NA> {turn['speaker_id']} <NA> <NA>\n")
        print(f"NIST RTTM exported to: {args.export_rttm}")

    if args.export_srt:
        with open(args.export_srt, "w") as f:
            for idx, turn in enumerate(aligned_turns):
                f.write(f"{idx + 1}\n")
                f.write(f"{format_srt_time(turn['start'])} --> {format_srt_time(turn['end'])}\n")
                f.write(f"[{turn['speaker_id']}] {turn['text']}\n\n")
        print(f"SRT subtitles exported to: {args.export_srt}")

if __name__ == "__main__":
    main()
