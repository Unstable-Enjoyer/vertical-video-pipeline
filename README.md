# Vertical Video Pipeline

Automatically detects split-screen videos, selects the cleanest panel, crops it out, and encodes it as a standalone vertical video.

Built for batch processing - point it at a folder and it handles the rest. All progress is tracked in a local database so you can stop and resume at any time.

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)

## What It Does

This tool:

1. **Detects** whether a video has a split-screen layout (2, 3, or 4 panels)
2. **Selects** the cleanest panel (least overlays/clutter) using edge density analysis
3. **Crops** that panel out
4. **Upscales** it to your target resolution (optional)
5. **Encodes** it to HEVC with configurable quality presets

### Before → After

```
┌─────────┬─────────┐         ┌─────────┐
│         │         │         │         │
│ Panel 1 │ Panel 2 │   →     │ Panel 1 │
│         │         │         │         │
└─────────┴─────────┘         └─────────┘
  Input (split-screen)       Output (single panel)
```

## Requirements

- **Python 3.8+**
- **FFmpeg** (with `ffprobe`) - must be available in your system PATH
- **NVIDIA GPU** (optional) - if available, uses hardware-accelerated HEVC encoding (`hevc_nvenc`). If no compatible GPU is found at startup, the script uses CPU encoding (`libx265`) instead.

### Python Dependencies

```
opencv-python
numpy
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/unstable-enjoyer/vertical-video-pipeline.git
   cd vertical-video-pipeline
   ```

2. **Install Python dependencies:**

   ```bash
   pip install opencv-python numpy
   ```

3. **Verify FFmpeg is installed:**

   ```bash
   ffmpeg -version
   ```

   If you don't have FFmpeg, install it:
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or use `winget install ffmpeg`
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora)

## Quick Start

The simplest usage - process all videos in a folder:

```bash
python vertical_pipeline.py /path/to/your/videos
```

Output files are saved to an `output/` folder next to the script. That's it.

### Scan subfolders too

```bash
python vertical_pipeline.py /path/to/your/videos --recursive
```

### Preview without encoding

```bash
python vertical_pipeline.py /path/to/your/videos --dry-run
```

This runs detection only and shows which videos have splits and which panel would be selected, without encoding anything.

## Configuration

All settings live at the top of `vertical_pipeline.py` in the `CONFIG` block. Open the file and edit the values directly - no config files needed.

### Detection

| Setting | Default | Description |
|---|---|---|
| `THRESHOLD` | `0.75` | Similarity threshold for split detection. Lower = more sensitive (may produce false positives). Higher = stricter. |
| `SAMPLE_INTERVAL` | `2.0` | Seconds between sampled frames. Lower = more accurate but slower detection. |
| `DETECT_2_PANEL` | `True` | Enable/disable 2-panel split detection. |
| `DETECT_3_PANEL` | `True` | Enable/disable 3-panel split detection. |
| `DETECT_4_PANEL` | `False` | Enable/disable 4-panel split detection. |
| `EDGE_DENSITY_MIN_DIFF` | `0.10` | Minimum relative difference in edge density to prefer one panel over another. If panels are too similar, defaults to the first panel. |

### Encoding

| Setting | Default | Description |
|---|---|---|
| `ENCODE_PRESET` | `"balanced"` | Encoding preset (see [Presets](#presets) below). |
| `MIN_BITRATE_KBPS` | `500` | Minimum output bitrate floor in kbps. |
| `UPSCALE_ENABLED` | `True` | Upscale the cropped panel to `TARGET_HEIGHT`. |
| `TARGET_HEIGHT` | `1440` | Target height in pixels when upscaling is enabled. |

### General

| Setting | Default | Description |
|---|---|---|
| `MAX_DURATION` | `360` | Skip videos longer than this (in seconds). |
| `VIDEO_EXTENSIONS` | `.mp4 .mkv .mov .avi .webm` | File extensions to look for. |

### Presets

Presets control the trade-off between encoding speed, file size, and quality:

| Preset | Bitrate Ratio | GPU Preset | CPU Preset | Use When |
|---|---|---|---|---|
| `fast` | 0.9x original | p3 | faster | You want quick results and don't mind larger files |
| `balanced` | 0.7x original | p5 | medium | General use - good balance of speed, size, and quality |
| `quality` | 0.7x original | p7 | slow | You want the best quality per bit and can wait longer |

The bitrate ratio is applied to the source video's bitrate. For example, a source at 10,000 kbps with the `balanced` preset produces output at 7,000 kbps.

## Command-Line Options

Most settings can also be overridden from the command line:

```
python vertical_pipeline.py <input_dir> [options]
```

| Option | Description |
|---|---|
| `input_root` | Directory containing video files to process (required) |
| `-o`, `--output DIR` | Output directory (default: `output/` next to script) |
| `-r`, `--recursive` | Scan input directory recursively |
| `--threshold N` | Override split detection threshold |
| `--sample-interval N` | Override frame sampling interval (seconds) |
| `--max-duration N` | Override max video duration (seconds) |
| `--preset NAME` | Override encode preset (`fast`, `balanced`, `quality`) |
| `--no-upscale` | Disable upscaling, keep native resolution after crop |
| `--dry-run` | Detect splits only, don't encode |
| `--force-panel N` | Force a specific panel (0-indexed) instead of auto-selection |
| `--workers N` | Number of detection workers (default: 2, use 0 for sequential mode) |
| `--db PATH` | Custom path for the SQLite database |
| `--reset-errors` | Reset all errored videos back to pending for retry |
| `--rerun MODE` | `missing`: re-evaluate no_split videos with current threshold. `all`: re-evaluate + re-encode everything (also retries duration-skipped videos) |

## How It Works

### Split Detection

The tool samples frames at regular intervals and compares panels using normalized cross-correlation. If the average similarity between panels exceeds the threshold, the video is classified as a split-screen.

### Panel Selection

Once a split is detected, each panel's edge density is measured using Canny edge detection. The panel with the **lowest** edge density is selected - this tends to be the panel with the least overlays, watermarks, or UI clutter. If all panels have similar edge density (within `EDGE_DENSITY_MIN_DIFF`), the first panel is used.

### Encoding

- **GPU (NVIDIA):** Uses `hevc_nvenc` with hardware acceleration. Automatically detected at startup.
- **CPU fallback:** Uses `libx265` with equivalent settings. Same bitrate targets, pixel format (10-bit), and color metadata.
- Output is encoded in HEVC (H.265) with 10-bit color (`yuv420p10le`) for accurate color reproduction.

### Database

All state is tracked in an SQLite database (`pipeline.db` in the output directory). This enables:

- **Resumability** - stop and restart without reprocessing completed videos
- **Re-runs** - lower the threshold later and promote previously skipped videos
- **Status tracking** - see how many videos are done, skipped, errored, etc.

The database is created automatically on first run. You can safely delete it to start fresh.

## Output Structure

Output mirrors the input directory structure:

```
# Input                          # Output
videos/                          output/
├── clip1.mp4                    ├── clip1_vertical.mp4
├── clip2.mp4                    ├── clip2_vertical.mp4
└── subfolder/                   └── subfolder/
    └── clip3.mp4                    └── clip3_vertical.mp4
```

Videos that are not detected as split-screen are skipped (no output file is created).

## Examples

**Process a single folder with quality preset:**

```bash
python vertical_pipeline.py ./my_videos --preset quality
```

**Dry run to preview detections:**

```bash
python vertical_pipeline.py ./my_videos --dry-run
```

**Process recursively, no upscaling, custom output:**

```bash
python vertical_pipeline.py ./my_videos -r --no-upscale -o ./processed
```

**Lower the threshold and re-evaluate previously skipped videos:**

```bash
python vertical_pipeline.py ./my_videos --threshold 0.6 --rerun missing
```

**Re-encode everything from scratch:**

```bash
python vertical_pipeline.py ./my_videos --rerun all
```

**Sequential mode (single-threaded, simpler output):**

```bash
python vertical_pipeline.py ./my_videos --workers 0
```

## Troubleshooting

### "No videos found"
- Make sure your videos have a supported extension (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`).
- Use `--recursive` if your videos are in subdirectories.

### All videos show "NO SPLIT"
- Your videos may not be split-screen, or the threshold is too high. Try lowering it:
  ```bash
  python vertical_pipeline.py ./my_videos --threshold 0.5 --dry-run
  ```

### Encoding errors
- Run `ffmpeg -version` to make sure FFmpeg is installed and in PATH.
- If you see NVENC errors, your GPU may not support HEVC encoding. The encoder is chosen once at startup - if GPU detection passed but encoding still fails, use `--reset-errors` to retry. You can also force CPU encoding by editing the script (set `use_gpu = False` in `main()`).
- Use `--reset-errors` to retry failed videos after fixing the issue.

### Slow performance
- CPU encoding (`libx265`) is significantly slower than GPU encoding. If you have an NVIDIA GPU, make sure your FFmpeg build includes NVENC support.
- Use the `fast` preset for quicker encoding at the cost of slightly larger files.
- Reduce `--workers` if detection is using too much memory.

## License

MIT
