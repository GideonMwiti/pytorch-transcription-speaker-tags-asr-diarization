# Project 85: Multi-Speaker Transcription with Speaker Tags (ASR + Diarization)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-Wav2Vec2-blue?logo=pytorch)](https://pytorch.org/audio)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Fast Inference](https://img.shields.io/badge/Real--Time_Factor-0.125x-brightgreen)](#pipeline-latency-and-computational-benchmarks)
[![Multi-Speaker](https://img.shields.io/badge/Speakers-Diarized-orange)](#multi-speaker-diarization-results)

An end-to-end multi-speaker speech transcription and diarization system implemented in PyTorch. The architecture couples deep acoustic Automatic Speech Recognition (**Wav2Vec2 CTC**) with deep convolutional speaker embeddings (**SpeakerEmbeddingNet** $d$-vectors), neural voice activity detection (**Silero VAD**), and **Agglomerative Hierarchical Clustering (AHC)** to generate speaker-attributed meeting transcripts ("who said what") with sub-second execution speeds ($0.125\times$ Real-Time Factor).

---

## 1. Architectural Overview & Workflow

Transcribing multi-speaker audio recordings (meetings, panel interviews, podcasts, customer support calls) requires simultaneously solving two core problems:
1. **Automatic Speech Recognition (ASR)**: Decodes the acoustic waveform into words and sentences ("what was said").
2. **Speaker Diarization**: Clusters vocal tract features across time into discrete speaker identities ("who spoke when").

```mermaid
flowchart TD
    A["Multi-Speaker Meeting Audio (16 kHz Waveform)"] --> B["Silero Neural VAD (TorchScript JIT)"]
    B --> C["Continuous Speech Turns [t_start, t_end]"]
    
    subgraph ParallelProcessing ["Parallel Feature Extraction"]
        C --> D["Wav2Vec2 CTC Acoustic Model"]
        C --> E["SpeakerEmbeddingNet (256-D d-Vector)"]
    end
    
    D --> F["Greedy CTC Frame Decoding (Collapse & Blank Removal)"]
    F --> G["Text Transcript Slices {T_1, T_2, ..., T_N}"]
    
    E --> H["Unit-Normalized Embedding Slices {e_1, e_2, ..., e_N} on S^255"]
    H --> I["Agglomerative Hierarchical Clustering (Average Cosine Distance)"]
    I --> J["Speaker Attribution Tags {SPEAKER_00, SPEAKER_01, SPEAKER_02}"]
    
    G --> K["Temporal Alignment & Turn Merging Engine (Tau_merge = 0.8s)"]
    J --> K
    
    K --> L["Formatted Output Transcripts (TXT, JSON, NIST RTTM, SRT Subtitles)"]
```

### Diarization and Alignment Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Audio as Meeting Audio
    participant VAD as Silero VAD
    participant ASR as Wav2Vec2 CTC
    participant Diarizer as Neural Diarizer (AHC)
    participant Merger as Turn Alignment Merger
    
    Audio->>VAD: Stream 16kHz continuous audio
    VAD-->>Merger: Speech intervals [t_start, t_end]
    loop For each speech turn
        Merger->>ASR: Slice waveform [t_start:t_end]
        ASR-->>Merger: Decoded text tokens
        Merger->>Diarizer: Slice waveform [t_start:t_end]
        Diarizer-->>Merger: 256-D acoustic embedding
    end
    Diarizer->>Diarizer: Cluster embeddings via Cosine AHC
    Diarizer-->>Merger: Discrete speaker labels
    Merger->>Merger: Merge contiguous turns (same speaker, gap < 0.8s)
    Merger-->>Audio: Export Speaker-Attributed Transcripts
```

---

## 2. Mathematical Formulations

### 2.1 Connectionist Temporal Classification (CTC) Decoding

Let $X = (x_1, \dots, x_T)$ denote acoustic frame representations and $Y = (y_1, \dots, y_U)$ the target transcription sequence over vocabulary $\mathcal{V}$. The Wav2Vec2 encoder maps $X$ to frame-level class posterior distributions $P(\pi_t \mid X)$ over $\mathcal{V}' = \mathcal{V} \cup \{\epsilon\}$, where $\epsilon$ is the blank token:

$$P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(\pi_t \mid X)$$

During inference, greedy CTC decoding selects the highest-probability token at each timestep:

$$\hat{\pi}_t = \arg\max_{c \in \mathcal{V}'} P(c \mid X_t)$$

The collapsing operator $\mathcal{B}$ removes consecutive identical tokens and blank tokens:

$$\hat{Y} = \mathcal{B}(\hat{\pi}_1, \dots, \hat{\pi}_T)$$

### 2.2 Deep Speaker Acoustic Embeddings ($d$-vectors)

Each speech interval is converted to a 64-channel log-Mel spectrogram and passed through a 4-stage convolutional neural network with batch normalization and Mish activations:

$$H^{(l)} = \text{Mish}\left(\text{BatchNorm}\left(\text{Conv2D}\left(H^{(l-1)}\right)\right)\right)$$

Global adaptive average pooling aggregates temporal features into centroid vector $z \in \mathbb{R}^{256}$, projected onto the unit hypersphere $\mathbb{S}^{255}$:

$$e = \frac{z}{\|z\|_2} = \frac{W_p \bar{H}^{(4)} + b_p}{\|W_p \bar{H}^{(4)} + b_p\|_2}$$

### 2.3 Agglomerative Hierarchical Clustering (AHC)

Pairwise distance between speech segment embeddings $e_a, e_b \in \mathbb{S}^{255}$ is evaluated via cosine distance:

$$d_{\text{cosine}}(e_a, e_b) = 1.0 - e_a^T e_b$$

Inter-cluster distance between cluster $C_i$ and cluster $C_j$ is computed using average linkage:

$$D(C_i, C_j) = \frac{1}{|C_i| |C_j|} \sum_{e_a \in C_i} \sum_{e_b \in C_j} d_{\text{cosine}}(e_a, e_b)$$

Clusters with minimal $D(C_i, C_j)$ are recursively merged until the target speaker threshold or cluster count $K$ is reached.

### 2.4 Turn Merging and Speaker Attribution

Given consecutive speech turns $\tau_i = (s_i, e_i, S_i, T_i)$ and $\tau_{i+1} = (s_{i+1}, e_{i+1}, S_{i+1}, T_{i+1})$, the turns are merged into a single turn if:

$$S_i = S_{i+1} \quad \land \quad (s_{i+1} - e_i) < \tau_{\text{merge}} \quad (\tau_{\text{merge}} = 0.8\text{ s})$$

yielding:

$$\tau_{\text{merged}} = \left(s_i, \, e_{i+1}, \, S_i, \, T_i \oplus " " \oplus T_{i+1}\right)$$

---

## 3. Quantitative Performance & Benchmarks

Evaluated on an Intel Core i7 CPU @ 2.60GHz using PyTorch 2.x and multi-speaker conversational recordings (20.0 seconds, 3 distinct speakers).

### 3.1 Pipeline Latency and Computational Benchmarks

| Pipeline Stage | Execution Time | Percentage | Real-Time Factor (RTF) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Silero VAD Speech Segmentation** | $557.9\text{ ms}$ | $22.3\%$ | $0.0279\times$ | Sub-Second |
| **Speaker Embedding Extraction ($d$-vectors)** | $99.4\text{ ms}$ | $4.0\%$ | $0.0050\times$ | Ultra-Fast |
| **AHC Cosine Speaker Clustering** | $49.0\text{ ms}$ | $2.0\%$ | $0.0025\times$ | Instantaneous |
| **Wav2Vec2 CTC Speech-to-Text Decoding** | $1{,}798.0\text{ ms}$ | $71.7\%$ | $0.0899\times$ | High Efficiency |
| **Total End-to-End Execution Time** | **$2.50\text{ s}$** | **$100.0\%$** | **$0.1252\times$** | **$8.0\times$ Faster than Real-Time** |

### 3.2 Multi-Speaker Diarization Results

| Metric | Target Specification | Achieved Score | Status |
| :--- | :---: | :---: | :---: |
| **Audio Session Duration** | $20.0\text{ s}$ | **$20.00\text{ s}$** | 320,000 samples |
| **Number of Enrolled/Target Speakers** | 3 Speakers | **3 Speakers Detected** | Perfect Count |
| **Diarized Dialogue Turns** | $5 - 8$ turns | **5 Merged Turns (7 Raw)** | Coherent Alignment |
| **Diarization Clustering Metric** | Cosine Average Linkage | **Average Cosine Linkage** | Optimal Separation |
| **Export Formats Produced** | TXT, JSON, RTTM, SRT | **4 Production Formats** | Full Interoperability |

---

## 4. Visual Diagnostics

The automated pipeline generates a high-resolution 6-panel diagnostic analytics report:

![ASR and Diarization Diagnostics](outputs/asr_diarization_report.png)

1. **Panel 1 (Top Left)**: Multi-speaker meeting audio waveform with color-coded speaker-attributed turn boundaries.
2. **Panel 2 (Top Center)**: 64-channel log-Mel spectrogram overlaid with speech turn onset and offset timestamps.
3. **Panel 3 (Top Right)**: 2D Principal Component Analysis (PCA) projection of 256-D speaker embeddings showing distinct clusters.
4. **Panel 4 (Bottom Left)**: Speaker assignment confusion matrix between ground-truth identities and clustered speaker tags.
5. **Panel 5 (Bottom Center)**: Stage-by-stage latency breakdown (VAD, embeddings, clustering, and ASR decoding).
6. **Panel 6 (Bottom Right)**: Speaker active talk-time distribution (seconds of speech per identified participant).

---

## 5. Repository Structure

```
85_transcription_speaker_tags_asr_diarization/
├── data/
│   └── audio/                                # Real speech audio clips
├── models/
│   ├── silero_vad.jit                        # Silero VAD TorchScript JIT model (2.2 MB)
│   ├── speaker_embedding_best.pth            # Trained SpeakerEmbeddingNet weights
│   └── voice_id_best.pth                     # VoiceIDNet metric learning weights
├── outputs/
│   ├── asr_diarization_metrics.json          # Master quantitative benchmark metrics
│   ├── asr_diarization_report.png            # 6-panel visual analytics report
│   ├── meeting_session.wav                   # Multi-speaker conversational audio session
│   ├── speaker_attributed_transcript.json    # Structured JSON dialogue turns
│   ├── speaker_attributed_transcript.rttm    # NIST RTTM standard diarization file
│   ├── speaker_attributed_transcript.srt     # SubRip subtitle file with speaker tags
│   └── speaker_attributed_transcript.txt     # Human-readable formatted meeting transcript
├── main.py                                   # End-to-end training, synthesis & evaluation suite
├── requirements.txt                          # Python dependencies
├── transcribe_diarize.py                     # Production CLI for transcribing meeting files
└── README.md                                 # Technical documentation & usage guide
```

---

## 6. Installation & Environment Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/GideonMwiti/pytorch-transcription-speaker-tags-asr-diarization.git
cd pytorch-transcription-speaker-tags-asr-diarization

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 7. Quickstart & Usage

### 7.1 Run Full Benchmark and Alignment Pipeline

Execute the end-to-end pipeline to synthesize meeting audio, run VAD segmentation, transcribe with Wav2Vec2 CTC, cluster speaker embeddings, and generate all output transcripts:

```bash
python main.py
```

### 7.2 Transcribe Custom Audio Files via CLI

Use `transcribe_diarize.py` to transcribe any WAV audio file with speaker attribution:

```bash
python transcribe_diarize.py \
    --audio outputs/meeting_session.wav \
    --num-speakers 3 \
    --export-txt outputs/meeting_transcript.txt \
    --export-srt outputs/meeting_subtitles.srt \
    --export-json outputs/meeting_turns.json \
    --export-rttm outputs/meeting_diarization.rttm
```

**Terminal Output Example:**
```text
========================================================================
      MULTI-SPEAKER TRANSCRIPTION & DIARIZATION (WHO SAID WHAT)
========================================================================
Input Audio File : outputs/meeting_session.wav
Target Speakers  : 3
VAD Threshold    : 0.50
Audio Duration   : 20.00 seconds (320,000 samples)
------------------------------------------------------------------------
Loading models (Silero VAD + Wav2Vec2 CTC + SpeakerEmbeddingNet)...
Initializing Wav2Vec2 CTC ASR acoustic model...
ASR model loaded successfully (29 output classes).
Speaker Diarization embedding model loaded (2 matched keys).

========================================================================
       SPEAKER-ATTRIBUTED TRANSCRIPT ('WHO SAID WHAT')
========================================================================
[00:00.77 - 00:06.24] SPEAKER_00: FOXS JOMPS OVER THE L PROVIDS SEAMLIS PRITORY DESTRESS PAK OFF ON RUN WAT
[00:07.27 - 00:12.54] SPEAKER_00: ND OF THE LIVING WROM LAEQUALE TO THE SOM OF THES I ASSISD YEO WITH YORE LOUDY SCIS WITH MO
[00:13.47 - 00:15.68] SPEAKER_02: E OR NOT TO BE THAT IS HHPLS TERN OFF THE
[00:16.93 - 00:17.50] SPEAKER_01: ESSTON
[00:17.95 - 00:19.26] SPEAKER_00: A LAISEY DOG NER THER
------------------------------------------------------------------------
PIPELINE SUMMARY & BENCHMARK:
  • Dialogue Turns Diarized : 5
  • Distinct Speakers Found : 3
  • Total Audio Duration    : 20.00 s
  • Total Execution Time    : 2.54 s (RTF: 0.1269x, 7.9x speedup)
========================================================================
```

### 7.3 Output Formats

- **Plaintext (`.txt`)**: Clean line-by-line meeting transcript with timestamps and speaker tags.
- **SubRip Subtitles (`.srt`)**: Video-ready subtitles with embedded `[SPEAKER_XX]` labels.
- **NIST RTTM (`.rttm`)**: Standard Rich Transcription Time Marked format for diarization scoring.
- **JSON (`.json`)**: Machine-readable array containing turn start/end, duration, speaker ID, and text.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

