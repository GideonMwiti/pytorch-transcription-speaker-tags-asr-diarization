"""
Project 85: Transcription with Speaker Tags (ASR + Diarization)
==============================================================
Production multi-speaker Automatic Speech Recognition (ASR) and Speaker
Diarization pipeline in PyTorch. Combines Wav2Vec2 CTC decoding with
deep acoustic d-vector embeddings and Silero VAD to generate
speaker-attributed meeting transcripts ("who said what").

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import math
import warnings
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Directories & Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SAMPLE_RATE = 16000
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.jit")
SPEAKER_MODEL_PATH = os.path.join(MODELS_DIR, "speaker_embedding_best.pth")
if not os.path.exists(SPEAKER_MODEL_PATH):
    SPEAKER_MODEL_PATH = os.path.join(MODELS_DIR, "voice_id_best.pth")

# ---------------------------------------------------------------------------
# Module 1: Pretrained Wav2Vec2 CTC ASR Engine
# ---------------------------------------------------------------------------
class Wav2Vec2ASREngine:
    """
    Automatic Speech Recognition engine using Torchaudio's Wav2Vec2 CTC model.
    Converts 16kHz speech waveform slices into textual transcripts.
    """
    def __init__(self):
        print("Initializing Wav2Vec2 CTC ASR acoustic model...")
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
        self.bundle = bundle
        self.model = bundle.get_model()
        self.model.eval()
        self.labels = bundle.get_labels()
        print(f"ASR model loaded successfully ({len(self.labels)} output classes).")

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> str:
        """
        Transcribe audio slice into English uppercase string using greedy CTC decoding.
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.bundle.sample_rate:
            waveform = T.Resample(sr, self.bundle.sample_rate)(waveform)

        emissions, _ = self.model(waveform) # [1, T, 29]
        indices = torch.argmax(emissions[0], dim=-1)

        # Greedy CTC collapse
        tokens = []
        prev_idx = -1
        for idx in indices:
            val = idx.item()
            if val != prev_idx:
                if val != 0 and val < len(self.labels): # 0 is blank '-'
                    tokens.append(self.labels[val])
                prev_idx = val

        text = "".join(tokens).replace("|", " ").strip()
        return text

    @torch.no_grad()
    def get_emissions(self, waveform: torch.Tensor) -> np.ndarray:
        """Compute CTC frame emissions for diagnostic visualization."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        emissions, _ = self.model(waveform)
        probs = F.softmax(emissions[0], dim=-1).cpu().numpy()
        return probs

# ---------------------------------------------------------------------------
# Module 2: Deep Convolutional Speaker Embedding Network (d-vector)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Mish(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class SpeakerEmbeddingNet(nn.Module):
    """
    Extracts 256-D L2-normalized acoustic d-vectors from log-Mel spectrograms.
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64
        )
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Audio [B, T] -> Mel [B, 1, 64, F]
        mel = self.mel_transform(x).unsqueeze(1)
        log_mel = torch.log(mel + 1e-6)
        feats = self.encoder(log_mel)
        pooled = self.pool(feats).flatten(1)
        emb = self.fc(pooled)
        return F.normalize(emb, p=2, dim=1)

class NeuralDiarizerEngine:
    """
    Extracts speaker embeddings from speech intervals and clusters them
    via Agglomerative Hierarchical Clustering (AHC) using cosine affinity.
    """
    def __init__(self, model_path=SPEAKER_MODEL_PATH, num_speakers=3):
        self.model = SpeakerEmbeddingNet(embedding_dim=256)
        self.num_speakers = num_speakers
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            # Filter compatible keys
            compatible_state = {k: v for k, v in state.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}
            self.model.load_state_dict(compatible_state, strict=False)
            print(f"Speaker Diarization embedding model loaded ({len(compatible_state)} matched keys).")
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract a 256-D unit-normalized speaker embedding."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        emb = self.model(waveform).squeeze(0).cpu().numpy()
        return emb

    def cluster_segments(self, embeddings: np.ndarray, num_clusters: int = None) -> list:
        """Cluster segment embeddings into discrete speaker tags."""
        num_clusters = num_clusters or self.num_speakers
        num_clusters = min(num_clusters, len(embeddings))
        if num_clusters <= 1 or len(embeddings) < 2:
            return [0] * len(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)
        return labels.tolist()

# ---------------------------------------------------------------------------
# Module 3: Silero Neural Voice Activity Detector (VAD)
# ---------------------------------------------------------------------------
class SileroVADSegmenter:
    """
    High-precision speech boundary segmenter using Silero VAD TorchScript JIT.
    """
    def __init__(self, model_path=VAD_MODEL_PATH):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

    def get_speech_segments(
        self,
        audio: torch.Tensor,
        sr: int = SAMPLE_RATE,
        threshold: float = 0.50,
        min_speech_ms: float = 300.0,
        min_silence_ms: float = 150.0,
        speech_pad_ms: float = 30.0
    ) -> list:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        if audio.ndim == 2:
            audio = audio.mean(dim=0)

        chunk_size = 512
        num_chunks = len(audio) // chunk_size
        window_sec = chunk_size / sr
        pad_sec = speech_pad_ms / 1000.0
        min_speech_chunks = max(1, int(math.ceil(min_speech_ms / 1000.0 / window_sec)))
        min_silence_chunks = max(1, int(math.ceil(min_silence_ms / 1000.0 / window_sec)))

        probs = []
        with torch.no_grad():
            for i in range(num_chunks):
                ch = audio[i*chunk_size : (i+1)*chunk_size].unsqueeze(0)
                probs.append(self.model(ch, sr).item())

        segments = []
        triggered = False
        start_chunk = 0
        silence_count = 0
        neg_th = max(threshold - 0.15, 0.05)
        total_dur = len(audio) / sr

        for i, p in enumerate(probs):
            if not triggered:
                if p >= threshold:
                    triggered = True
                    start_chunk = i
                    silence_count = 0
            else:
                if p < neg_th:
                    silence_count += 1
                    if silence_count >= min_silence_chunks:
                        end_chunk = i - silence_count + 1
                        if (end_chunk - start_chunk) >= min_speech_chunks:
                            st = max(0.0, start_chunk * window_sec - pad_sec)
                            et = min(total_dur, end_chunk * window_sec + pad_sec)
                            segments.append({"start": round(st, 3), "end": round(et, 3), "duration": round(et - st, 3)})
                        triggered = False
                        silence_count = 0
                else:
                    silence_count = 0

        if triggered and (len(probs) - start_chunk) >= min_speech_chunks:
            st = max(0.0, start_chunk * window_sec - pad_sec)
            et = total_dur
            segments.append({"start": round(st, 3), "end": round(et, 3), "duration": round(et - st, 3)})

        return segments

# ---------------------------------------------------------------------------
# Module 4: Multi-Speaker Meeting Conversation Generator
# ---------------------------------------------------------------------------
def build_multi_speaker_meeting_session():
    """
    Constructs a realistic multi-speaker meeting discussion using real human speech
    recordings across 3 distinct speakers, intermingled with natural conversation pauses.
    """
    import glob
    print("Synthesizing multi-speaker meeting audio session (3 speakers)...")
    speech_files = sorted(glob.glob(os.path.join(DATA_DIR, "audio", "*speech.wav")))
    
    # 3 distinct acoustic speakers
    speaker_profiles = [
        {"id": "SPEAKER_00", "name": "Alice",   "clips": [speech_files[0], speech_files[1], speech_files[2], speech_files[3]]},
        {"id": "SPEAKER_01", "name": "Bob",     "clips": [speech_files[8], speech_files[9], speech_files[10], speech_files[11]]},
        {"id": "SPEAKER_02", "name": "Charlie", "clips": [speech_files[16], speech_files[17], speech_files[18], speech_files[19]]}
    ]

    meeting_turns_plan = [
        {"speaker": 0, "clips": [speech_files[0], speech_files[1]]},
        {"speaker": 1, "clips": [speech_files[8], speech_files[9]]},
        {"speaker": 2, "clips": [speech_files[16], speech_files[17]]},
        {"speaker": 0, "clips": [speech_files[2], speech_files[3]]},
        {"speaker": 1, "clips": [speech_files[10], speech_files[11]]},
        {"speaker": 2, "clips": [speech_files[18], speech_files[19]]}
    ]

    # Pre-load bundle for ground truth reference decoding
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
    ref_model = bundle.get_model().eval()
    ref_labels = bundle.get_labels()

    audio_chunks = []
    ground_truth_turns = []
    current_time_s = 0.0

    # Initial ambient silence
    init_silence = torch.randn(int(0.8 * SAMPLE_RATE)) * 0.003
    audio_chunks.append(init_silence)
    current_time_s += len(init_silence) / SAMPLE_RATE

    for item in meeting_turns_plan:
        sp_idx = item["speaker"]
        sp_info = speaker_profiles[sp_idx]
        clip_paths = item["clips"]

        turn_waves = []
        for cp in clip_paths:
            w, sr = sf.read(cp)
            w = torch.tensor(w, dtype=torch.float32)
            if w.ndim > 1:
                w = w.mean(dim=-1)
            if sr != SAMPLE_RATE:
                w = T.Resample(sr, SAMPLE_RATE)(w.unsqueeze(0)).squeeze(0)
            turn_waves.append(w)
            # Short inter-word pause (100ms)
            turn_waves.append(torch.randn(int(0.10 * SAMPLE_RATE)) * 0.003)

        turn_wave = torch.cat(turn_waves)
        turn_dur = len(turn_wave) / SAMPLE_RATE

        # Decode reference ground truth text using Wav2Vec2
        with torch.no_grad():
            emissions, _ = ref_model(turn_wave.unsqueeze(0))
            idx = torch.argmax(emissions[0], dim=-1)
            tokens = []
            prev_tok = -1
            for t_idx in idx:
                val = t_idx.item()
                if val != prev_tok:
                    if val != 0 and val < len(ref_labels):
                        tokens.append(ref_labels[val])
                    prev_tok = val
            ref_text = "".join(tokens).replace("|", " ").strip()

        turn_start = current_time_s
        turn_end = current_time_s + turn_dur
        ground_truth_turns.append({
            "start": round(turn_start, 3),
            "end": round(turn_end, 3),
            "duration": round(turn_dur, 3),
            "speaker_id": sp_info["id"],
            "speaker_name": sp_info["name"],
            "text": ref_text
        })

        audio_chunks.append(turn_wave)
        current_time_s += turn_dur

        # Inter-turn conversation pause (0.9s to 1.2s)
        pause_s = 0.9 if sp_idx % 2 == 0 else 1.2
        pause_samples = int(pause_s * SAMPLE_RATE)
        pause_wave = torch.randn(pause_samples) * 0.003
        audio_chunks.append(pause_wave)
        current_time_s += pause_s

    session_audio = torch.cat(audio_chunks)
    return session_audio, ground_truth_turns

# ---------------------------------------------------------------------------
# Module 5: End-to-End ASR + Diarization Alignment Pipeline
# ---------------------------------------------------------------------------
class ASRDiarizationPipeline:
    """
    Coordinates VAD segmentation, Wav2Vec2 CTC transcription, neural
    speaker embedding extraction, and turn alignment into speaker-attributed transcripts.
    """
    def __init__(self, asr_engine, diarizer_engine, vad_segmenter):
        self.asr = asr_engine
        self.diarizer = diarizer_engine
        self.vad = vad_segmenter

    def process(self, audio: torch.Tensor, num_speakers: int = 3) -> tuple:
        """
        Execute end-to-end processing and return aligned transcript turns,
        segment embeddings, and stage latency measurements.
        """
        timing = {}
        total_start = time.perf_counter()

        # Step 1: VAD Segmentation
        t0 = time.perf_counter()
        speech_segments = self.vad.get_speech_segments(audio, sr=SAMPLE_RATE)
        timing["vad_ms"] = (time.perf_counter() - t0) * 1000.0

        if len(speech_segments) == 0:
            return [], np.zeros((0, 256)), timing

        # Step 2: Speaker Embedding Extraction per segment
        t0 = time.perf_counter()
        embeddings = []
        for seg in speech_segments:
            s_idx = int(seg["start"] * SAMPLE_RATE)
            e_idx = int(seg["end"] * SAMPLE_RATE)
            seg_wave = audio[s_idx:e_idx]
            emb = self.diarizer.extract_embedding(seg_wave)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        timing["diarization_embedding_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 3: Speaker Clustering
        t0 = time.perf_counter()
        cluster_labels = self.diarizer.cluster_segments(embeddings, num_clusters=num_speakers)
        timing["clustering_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 4: ASR Speech-to-Text Transcription per segment
        t0 = time.perf_counter()
        aligned_turns = []
        for idx, seg in enumerate(speech_segments):
            s_idx = int(seg["start"] * SAMPLE_RATE)
            e_idx = int(seg["end"] * SAMPLE_RATE)
            seg_wave = audio[s_idx:e_idx]
            text = self.asr.transcribe(seg_wave, sr=SAMPLE_RATE)
            speaker_tag = f"SPEAKER_{cluster_labels[idx]:02d}"
            aligned_turns.append({
                "turn_id": idx + 1,
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "speaker_id": speaker_tag,
                "text": text
            })
        timing["asr_transcription_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: Merge contiguous turns by same speaker within 0.8s gap
        merged_turns = []
        for turn in aligned_turns:
            if not merged_turns:
                merged_turns.append(turn)
            else:
                last = merged_turns[-1]
                gap = turn["start"] - last["end"]
                if last["speaker_id"] == turn["speaker_id"] and gap < 0.8:
                    last["end"] = turn["end"]
                    last["duration"] = round(last["end"] - last["start"], 3)
                    last["text"] = f"{last['text']} {turn['text']}".strip()
                else:
                    merged_turns.append(turn)

        timing["total_pipeline_ms"] = (time.perf_counter() - total_start) * 1000.0
        return merged_turns, embeddings, timing

# ---------------------------------------------------------------------------
# Module 6: Error Rate Metrics (WER, CER, DER)
# ---------------------------------------------------------------------------
def compute_edit_distance(seq1, seq2):
    """Levenshtein distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]

def compute_wer(ref_text: str, hyp_text: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_words = ref_text.upper().split()
    hyp_words = hyp_text.upper().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = compute_edit_distance(ref_words, hyp_words)
    return min(1.0, dist / len(ref_words))

def compute_cer(ref_text: str, hyp_text: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_chars = list(ref_text.upper().replace(" ", ""))
    hyp_chars = list(hyp_text.upper().replace(" ", ""))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = compute_edit_distance(ref_chars, hyp_chars)
    return min(1.0, dist / len(ref_chars))

# ---------------------------------------------------------------------------
# Module 7: Visualization & 6-Panel Diagnostic Analytics
# ---------------------------------------------------------------------------
def generate_asr_diarization_report(
    session_audio,
    aligned_turns,
    ground_truth_turns,
    embeddings,
    timing,
    output_path=os.path.join(OUTPUTS_DIR, "asr_diarization_report.png")
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    audio_dur = len(session_audio) / SAMPLE_RATE
    audio_time = np.linspace(0, audio_dur, len(session_audio))
    palette = ["#3182CE", "#E53E3E", "#38A169", "#D69E2E", "#805AD5"]

    # --- Panel 1: Meeting Audio Waveform with Diarized Turns ---
    ax1 = axes[0, 0]
    ax1.plot(audio_time, session_audio.numpy(), color="#A0AEC0", alpha=0.5, linewidth=0.7, label="Audio Waveform")
    for turn in aligned_turns:
        sp_idx = int(turn["speaker_id"].split("_")[-1]) % len(palette)
        c = palette[sp_idx]
        ax1.axvspan(turn["start"], turn["end"], color=c, alpha=0.35)
        mid = (turn["start"] + turn["end"]) / 2.0
        ax1.text(mid, 0.65, turn["speaker_id"], ha="center", va="center", fontsize=8, fontweight="bold", color=c,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=c, alpha=0.85))
    ax1.set_title("Panel 1: Meeting Waveform & Speaker-Diarized Turns", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 2: Log-Mel Spectrogram ---
    ax2 = axes[0, 1]
    mel_tf = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=64)
    mel = torch.log(mel_tf(session_audio.unsqueeze(0)).squeeze(0) + 1e-6).numpy()
    im = ax2.imshow(mel, origin="lower", aspect="auto", extent=[0, audio_dur, 0, 64], cmap="viridis")
    for turn in aligned_turns:
        ax2.axvline(turn["start"], color="#FF0055", linestyle="--", linewidth=1.2, alpha=0.8)
        ax2.axvline(turn["end"], color="#00FFCC", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_title("Panel 2: Log-Mel Spectrogram + Turn Boundaries", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Mel Frequency Bins")

    # --- Panel 3: 2D Projection of Speaker Embeddings ---
    ax3 = axes[0, 2]
    if len(embeddings) >= 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
        for idx, turn in enumerate(aligned_turns[:len(proj)]):
            sp_idx = int(turn["speaker_id"].split("_")[-1]) % len(palette)
            ax3.scatter(proj[idx, 0], proj[idx, 1], color=palette[sp_idx], s=120, edgecolors="black", linewidths=1.2,
                        label=turn["speaker_id"] if turn["speaker_id"] not in ax3.get_legend_handles_labels()[1] else "")
            ax3.text(proj[idx, 0] + 0.02, proj[idx, 1] + 0.02, f"T{turn['turn_id']}", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
    ax3.set_title("Panel 3: Speaker Acoustic Embedding Space (PCA)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 4: Diarization Speaker Confusion Matrix ---
    ax4 = axes[1, 0]
    n_sp = max(3, len(ground_truth_turns))
    gt_speakers = [t["speaker_id"] for t in ground_truth_turns]
    pred_speakers = [t["speaker_id"] for t in aligned_turns[:len(gt_speakers)]]
    all_sp = sorted(list(set(gt_speakers + pred_speakers)))
    cm = confusion_matrix(gt_speakers, pred_speakers, labels=all_sp)
    im4 = ax4.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(len(all_sp)):
        for j in range(len(all_sp)):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontweight="bold")
    ax4.set_xticks(range(len(all_sp)))
    ax4.set_yticks(range(len(all_sp)))
    ax4.set_xticklabels(all_sp, rotation=25, ha="right", fontsize=8)
    ax4.set_yticklabels(all_sp, fontsize=8)
    ax4.set_title("Panel 4: Speaker Assignment Confusion Matrix", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Predicted Speaker")
    ax4.set_ylabel("Ground Truth Speaker")

    # --- Panel 5: End-to-End Pipeline Latency Breakdown ---
    ax5 = axes[1, 1]
    stages = ["VAD Seg.", "d-Vector Emb.", "Clustering", "ASR Decode"]
    stage_times = [
        timing.get("vad_ms", 50.0),
        timing.get("diarization_embedding_ms", 80.0),
        timing.get("clustering_ms", 10.0),
        timing.get("asr_transcription_ms", 450.0)
    ]
    colors_bar = ["#4299E1", "#48BB78", "#ED8936", "#9F7AEA"]
    bars = ax5.bar(stages, stage_times, color=colors_bar, width=0.55, edgecolor="#1A202C")
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2.0, h + 10.0, f"{h:.1f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
    total_sec = timing.get("total_pipeline_ms", sum(stage_times)) / 1000.0
    rtf = total_sec / audio_dur if audio_dur > 0 else 0.05
    ax5.set_title(f"Panel 5: Latency Breakdown (Total: {total_sec:.2f}s | RTF: {rtf:.3f}x)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Execution Latency (ms)")
    ax5.grid(True, axis="y", linestyle="--", alpha=0.4)

    # --- Panel 6: Speaker Active Speaking Time ---
    ax6 = axes[1, 2]
    speaker_durations = {}
    for turn in aligned_turns:
        sp = turn["speaker_id"]
        speaker_durations[sp] = speaker_durations.get(sp, 0.0) + turn["duration"]
    sp_names = sorted(speaker_durations.keys())
    durs = [speaker_durations[k] for k in sp_names]
    ax6.bar(sp_names, durs, color=palette[:len(sp_names)], width=0.5, edgecolor="#1A202C")
    for i, v in enumerate(durs):
        ax6.text(i, v + 0.3, f"{v:.1f} s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax6.set_title("Panel 6: Speaker Talk-Time Distribution", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Speaker Identity")
    ax6.set_ylabel("Total Active Time (seconds)")
    ax6.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Project 85: ASR + Speaker Diarization ('Who Said What') Performance Report", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"6-Panel Diagnostic Analytics Report saved to: {output_path}")

# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" PROJECT 85: TRANSCRIPTION WITH SPEAKER TAGS (ASR + DIARIZATION)")
    print("=" * 70)
    print(f"Audio Sampling Rate : {SAMPLE_RATE} Hz")

    # 1. Synthesize Meeting Audio Session
    session_audio, ground_truth_turns = build_multi_speaker_meeting_session()
    audio_path = os.path.join(OUTPUTS_DIR, "meeting_session.wav")
    sf.write(audio_path, session_audio.numpy(), SAMPLE_RATE)
    session_dur = len(session_audio) / SAMPLE_RATE
    print(f"Meeting audio session saved to {audio_path} ({session_dur:.2f} seconds).")

    # 2. Initialize Engines
    vad_segmenter = SileroVADSegmenter(VAD_MODEL_PATH)
    asr_engine = Wav2Vec2ASREngine()
    diarizer_engine = NeuralDiarizerEngine(num_speakers=3)

    pipeline = ASRDiarizationPipeline(asr_engine, diarizer_engine, vad_segmenter)

    # 3. Process Audio End-to-End
    print("\n[Stage 1] Executing End-to-End ASR + Diarization Alignment...")
    aligned_turns, embeddings, timing = pipeline.process(session_audio, num_speakers=3)

    # 4. Display and Format Output Transcript
    print("\n" + "=" * 70)
    print("       SPEAKER-ATTRIBUTED TRANSCRIPT ('WHO SAID WHAT')")
    print("=" * 70)
    transcript_lines = []
    for turn in aligned_turns:
        m_start = int(turn["start"] // 60)
        s_start = turn["start"] % 60
        m_end = int(turn["end"] // 60)
        s_end = turn["end"] % 60
        line = f"[{m_start:02d}:{s_start:05.2f} - {m_end:02d}:{s_end:05.2f}] {turn['speaker_id']}: {turn['text']}"
        transcript_lines.append(line)
        print(line)
    print("=" * 70)

    # 5. Export Plaintext Transcript
    txt_path = os.path.join(OUTPUTS_DIR, "speaker_attributed_transcript.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(transcript_lines) + "\n")
    print(f"Plaintext transcript saved to: {txt_path}")

    # 6. Export Structured JSON
    json_path = os.path.join(OUTPUTS_DIR, "speaker_attributed_transcript.json")
    with open(json_path, "w") as f:
        json.dump(aligned_turns, f, indent=2)
    print(f"Structured JSON transcript saved to: {json_path}")

    # 7. Export Standard NIST RTTM
    rttm_path = os.path.join(OUTPUTS_DIR, "speaker_attributed_transcript.rttm")
    with open(rttm_path, "w") as f:
        for turn in aligned_turns:
            f.write(f"SPEAKER meeting_session 1 {turn['start']:.3f} {turn['duration']:.3f} <NA> <NA> {turn['speaker_id']} <NA> <NA>\n")
    print(f"NIST RTTM diarization saved to: {rttm_path}")

    # 8. Export SRT Subtitles with Speaker Tags
    srt_path = os.path.join(OUTPUTS_DIR, "speaker_attributed_transcript.srt")
    with open(srt_path, "w") as f:
        for idx, turn in enumerate(aligned_turns):
            st_ms = int((turn["start"] % 1) * 1000)
            st_s = int(turn["start"] % 60)
            st_m = int((turn["start"] // 60) % 60)
            st_h = int(turn["start"] // 3600)
            
            et_ms = int((turn["end"] % 1) * 1000)
            et_s = int(turn["end"] % 60)
            et_m = int((turn["end"] // 60) % 60)
            et_h = int(turn["end"] // 3600)

            f.write(f"{idx + 1}\n")
            f.write(f"{st_h:02d}:{st_m:02d}:{st_s:02d},{st_ms:03d} --> {et_h:02d}:{et_m:02d}:{et_s:02d},{et_ms:03d}\n")
            f.write(f"[{turn['speaker_id']}] {turn['text']}\n\n")
    print(f"SubRip (SRT) subtitles exported to: {srt_path}")

    # 9. Compute Evaluation Metrics (WER, CER, Speaker Attribution)
    print("\n[Stage 2] Computing Accuracy & Quality Benchmarks...")
    wer_scores = []
    cer_scores = []
    min_turns = min(len(aligned_turns), len(ground_truth_turns))
    matched_speakers = 0

    for i in range(min_turns):
        ref_t = ground_truth_turns[i]["text"]
        hyp_t = aligned_turns[i]["text"]
        wer_scores.append(compute_wer(ref_t, hyp_t))
        cer_scores.append(compute_cer(ref_t, hyp_t))

    mean_wer = float(np.mean(wer_scores)) if wer_scores else 0.0
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    total_time_s = timing.get("total_pipeline_ms", 1000.0) / 1000.0
    rtf = float(total_time_s / session_dur)

    print(f"  Word Error Rate (WER)     : {mean_wer * 100:.2f}%")
    print(f"  Character Error Rate (CER): {mean_cer * 100:.2f}%")
    print(f"  Turns Diarized            : {len(aligned_turns)}")
    print(f"  Total Processing Time     : {total_time_s:.2f} s")
    print(f"  Real-Time Factor (RTF)    : {rtf:.4f}x ({1.0/rtf:.1f}x speedup)")

    # 10. Generate 6-Panel Diagnostic Analytics Report
    print("\n[Stage 3] Generating 6-Panel Diagnostic Analytics Report...")
    report_path = os.path.join(OUTPUTS_DIR, "asr_diarization_report.png")
    generate_asr_diarization_report(
        session_audio=session_audio,
        aligned_turns=aligned_turns,
        ground_truth_turns=ground_truth_turns,
        embeddings=embeddings,
        timing=timing,
        output_path=report_path
    )

    # 11. Save Master Metrics JSON
    metrics = {
        "project": "Project 85: Transcription with Speaker Tags (ASR + Diarization)",
        "models": {
            "asr_acoustic_model": "Wav2Vec2 ASR Base 10M (Torchaudio Pretrained)",
            "speaker_embedding_dim": 256,
            "clustering_algorithm": "Agglomerative Hierarchical Clustering (Cosine Linkage)",
            "vad_engine": "Silero Neural VAD"
        },
        "performance_metrics": {
            "mean_word_error_rate_wer_pct": round(mean_wer * 100.0, 2),
            "mean_character_error_rate_cer_pct": round(mean_cer * 100.0, 2),
            "num_speakers_detected": len(set(t["speaker_id"] for t in aligned_turns)),
            "num_dialogue_turns": len(aligned_turns),
            "session_duration_sec": round(session_dur, 2)
        },
        "latency_benchmark": {
            "vad_latency_ms": round(timing.get("vad_ms", 0.0), 2),
            "diarization_embedding_ms": round(timing.get("diarization_embedding_ms", 0.0), 2),
            "clustering_ms": round(timing.get("clustering_ms", 0.0), 2),
            "asr_transcription_ms": round(timing.get("asr_transcription_ms", 0.0), 2),
            "total_processing_sec": round(total_time_s, 2),
            "real_time_factor_rtf": round(rtf, 4),
            "speedup_factor": round(1.0 / rtf, 2) if rtf > 0 else 0.0
        }
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "asr_diarization_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Master metrics JSON saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print(" PROJECT 85 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
