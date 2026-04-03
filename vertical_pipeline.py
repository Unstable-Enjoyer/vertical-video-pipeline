#!/usr/bin/env python3
"""
Vertical Video Pipeline
=======================
Detects split-screen videos (2/3/4-panel), selects the cleanest panel,
crops, optionally upscales, and encodes to HEVC.

Supports parallel mode (default): CPU workers for detection, GPU threads for encoding.
Use --workers 0 for sequential legacy mode.

All state tracked in SQLite for resumability.
"""

import subprocess
import json
import os
import sys
import signal
import sqlite3
import argparse
import time
import threading
import multiprocessing as mp
import queue
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

# ==========================
#        CONFIG
# ==========================
# Detection
THRESHOLD = 0.75              # split detection similarity threshold
SAMPLE_INTERVAL = 2.0         # seconds between sampled frames
DETECT_2_PANEL = True         # detect 2-panel split-screen
DETECT_3_PANEL = True         # detect 3-panel split-screen
DETECT_4_PANEL = False        # detect 4-panel split-screen
EDGE_DENSITY_MIN_DIFF = 0.10  # 10% relative difference to prefer one panel

# Encoding presets: each defines bitrate ratio and encoder speed
ENCODE_PRESETS = {
    "fast":     {"bitrate_multiplier": 0.9, "gpu_preset": "p3", "cpu_preset": "faster"},
    "balanced": {"bitrate_multiplier": 0.7, "gpu_preset": "p5", "cpu_preset": "medium"},
    "quality":  {"bitrate_multiplier": 0.7, "gpu_preset": "p7", "cpu_preset": "slow"},
}
ENCODE_PRESET = "balanced"    # fast | balanced | quality

MIN_BITRATE_KBPS = 500        # floor for output bitrate
UPSCALE_ENABLED = True        # upscale cropped panel to TARGET_HEIGHT
TARGET_HEIGHT = 1440           # upscale target height (when enabled)

# General
MAX_DURATION = 360             # skip videos longer than this (seconds)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
# ==========================

stop_requested = False


def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==============================================================
#                    DATA CLASSES
# ==============================================================
@dataclass
class DetectionResult:
    """Result from a detection worker — no DB access needed."""
    input_path: str
    subfolder: str
    filename: str
    status: str  # 'split_detected', 'no_split', 'skipped_duration', 'error'
    duration: float = 0.0
    width: int = 0
    height: int = 0
    bitrate: int = 0
    split_type: Optional[int] = None
    conf_2: float = 0.0
    conf_3: float = 0.0
    conf_4: float = 0.0
    panel_idx: int = 0
    edge_densities: List[float] = field(default_factory=list)
    error_message: str = ""


@dataclass
class EncodeJob:
    """Item passed from detection to encode queue."""
    input_path: str
    subfolder: str
    filename: str
    duration: float
    width: int
    height: int
    bitrate: int
    split_type: int
    panel_idx: int
    encode_idx: int = 0
    total_encodes: int = 0


# ==============================================================
#                   TERMINAL UI
# ==============================================================
class TerminalUI:
    """
    Thread-safe terminal output with fixed encoder status lines at bottom.

    Layout (TTY):
        ... scrolling detection/log lines ...
        [ENC 1] filename  ████░░░░  45.2% — ETA 12s
        [ENC 2] filename  ██░░░░░░  18.0% — ETA 28s

    Layout (non-TTY / piped):
        Simple line-per-event output, no cursor tricks.
    """

    def __init__(self, num_encoders: int, is_tty: bool):
        self._lock = threading.Lock()
        self._num_encoders = num_encoders
        self._is_tty = is_tty
        self._encoder_lines: Dict[int, str] = {}
        self._started = False
        self._term_width = 80
        self._bar_width = 20
        if is_tty:
            try:
                self._term_width = os.get_terminal_size().columns
            except OSError:
                pass

    def _refresh_term_width(self):
        if self._is_tty:
            try:
                self._term_width = os.get_terminal_size().columns
            except OSError:
                pass

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _build_bar(self, progress: float) -> str:
        filled = int(self._bar_width * progress)
        empty = self._bar_width - filled
        return "█" * filled + "░" * empty

    def _draw_encoder_lines(self):
        """Redraw the fixed encoder lines at the bottom. Must hold _lock."""
        if not self._is_tty or self._num_encoders == 0:
            return

        self._refresh_term_width()

        if self._started:
            # Move cursor up to first encoder line
            sys.stdout.write(f"\033[{self._num_encoders}A")

        for eid in range(1, self._num_encoders + 1):
            text = self._encoder_lines.get(eid, "waiting...")
            line = self._truncate(f"  [ENC {eid}] {text}", self._term_width)
            sys.stdout.write(f"\033[K{line}\n")

        sys.stdout.flush()
        self._started = True

    def log(self, message: str):
        """Print a scrolling log line above the encoder status area."""
        with self._lock:
            if self._is_tty and self._started and self._num_encoders > 0:
                # Move above encoder lines, print, then redraw encoders
                sys.stdout.write(f"\033[{self._num_encoders}A")
                sys.stdout.write(f"\033[K{message}\n")
                # Redraw encoder lines
                for eid in range(1, self._num_encoders + 1):
                    text = self._encoder_lines.get(eid, "waiting...")
                    line = self._truncate(f"  [ENC {eid}] {text}", self._term_width)
                    sys.stdout.write(f"\033[K{line}\n")
                sys.stdout.flush()
            else:
                print(message, flush=True)

    def update_encoder(self, encoder_id: int, filename: str, progress: float,
                       elapsed: float, eta: Optional[float] = None):
        """Update an encoder's progress bar."""
        with self._lock:
            bar = self._build_bar(progress)
            if eta is not None and eta >= 0:
                text = f"{filename}  {bar}  {progress:5.1%} — ETA {eta:.0f}s"
            else:
                text = f"{filename}  {bar}  {progress:5.1%} — elapsed {elapsed:.0f}s"
            self._encoder_lines[encoder_id] = text
            if self._is_tty:
                self._draw_encoder_lines()

    def set_encoder_status(self, encoder_id: int, status: str):
        """Set encoder to a text status (done, waiting, error)."""
        with self._lock:
            self._encoder_lines[encoder_id] = status
            if self._is_tty:
                self._draw_encoder_lines()

    def clear_encoders(self):
        """Remove encoder lines from terminal."""
        with self._lock:
            if self._is_tty and self._started and self._num_encoders > 0:
                sys.stdout.write(f"\033[{self._num_encoders}A")
                for _ in range(self._num_encoders):
                    sys.stdout.write("\033[K\n")
                sys.stdout.write(f"\033[{self._num_encoders}A")
                sys.stdout.flush()
            self._started = False

    def init_encoder_lines(self):
        """Reserve space for encoder lines."""
        with self._lock:
            if self._is_tty and self._num_encoders > 0:
                for eid in range(1, self._num_encoders + 1):
                    self._encoder_lines[eid] = "waiting..."
                for eid in range(1, self._num_encoders + 1):
                    text = self._encoder_lines[eid]
                    line = self._truncate(f"  [ENC {eid}] {text}", self._term_width)
                    sys.stdout.write(f"\033[K{line}\n")
                sys.stdout.flush()
                self._started = True


# ==============================================================
#                        DATABASE
# ==============================================================
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_path TEXT UNIQUE NOT NULL,
    subfolder TEXT NOT NULL,
    filename TEXT NOT NULL,
    output_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    duration_seconds REAL,
    original_width INTEGER,
    original_height INTEGER,
    original_bitrate_kbps INTEGER,
    split_type INTEGER,
    confidence_2panel REAL,
    confidence_3panel REAL,
    confidence_4panel REAL,
    selected_panel INTEGER,
    panel_edge_densities TEXT,
    output_width INTEGER,
    output_height INTEGER,
    output_bitrate_kbps INTEGER,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS subfolders (
    name TEXT PRIMARY KEY,
    video_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _run_migrations(conn: sqlite3.Connection):
    """Apply schema migrations for backwards compatibility with older DBs."""
    migrations = [
        "ALTER TABLE videos ADD COLUMN confidence_4panel REAL",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


class ThreadSafeDB:
    """SQLite wrapper with thread-safe writes for parallel mode."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(DB_SCHEMA)
        self._conn.commit()
        _run_migrations(self._conn)

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def fetchone(self, sql: str, params=()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params=()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def update_video(self, input_path: str, **kwargs):
        kwargs["updated_at"] = now_iso()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [input_path]
        self.execute(f"UPDATE videos SET {sets} WHERE input_path = ?", vals)

    def close(self):
        self._conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db_sequential(db_path: str) -> sqlite3.Connection:
    """Init DB for sequential mode (plain connection)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(DB_SCHEMA)
    conn.commit()
    _run_migrations(conn)
    return conn


def upsert_video(conn, input_path: str, subfolder: str, filename: str):
    """Insert new video. Works with Connection or ThreadSafeDB."""
    ts = now_iso()
    sql = """INSERT OR IGNORE INTO videos (input_path, subfolder, filename, status, created_at, updated_at)
             VALUES (?, ?, ?, 'pending', ?, ?)"""
    if isinstance(conn, ThreadSafeDB):
        conn.execute(sql, (input_path, subfolder, filename, ts, ts))
    else:
        conn.execute(sql, (input_path, subfolder, filename, ts, ts))
        conn.commit()


def update_video(conn, input_path: str, **kwargs):
    """Update video record. Works with Connection or ThreadSafeDB."""
    if isinstance(conn, ThreadSafeDB):
        conn.update_video(input_path, **kwargs)
    else:
        kwargs["updated_at"] = now_iso()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [input_path]
        conn.execute(f"UPDATE videos SET {sets} WHERE input_path = ?", vals)
        conn.commit()


def get_status_counts(conn) -> dict:
    if isinstance(conn, ThreadSafeDB):
        rows = conn.fetchall("SELECT status, COUNT(*) as cnt FROM videos GROUP BY status")
    else:
        rows = conn.execute("SELECT status, COUNT(*) as cnt FROM videos GROUP BY status").fetchall()
    return {r["status"]: r["cnt"] for r in rows}


def promote_no_split_videos(conn, threshold: float, sample_interval: float,
                            force_panel: Optional[int]) -> int:
    """
    Promote no_split videos whose stored confidence meets the new threshold.
    Runs panel selection on promoted videos (detection is skipped — we use stored scores).
    Returns the number of promoted videos.
    """
    sql = (
        "SELECT input_path, subfolder, filename, duration_seconds, "
        "original_width, original_height, original_bitrate_kbps, "
        "confidence_2panel, confidence_3panel, confidence_4panel "
        "FROM videos WHERE status = 'no_split' "
        "AND (confidence_2panel >= ? OR confidence_3panel >= ? OR confidence_4panel >= ?)"
    )
    if isinstance(conn, ThreadSafeDB):
        candidates = conn.fetchall(sql, (threshold, threshold, threshold))
    else:
        candidates = conn.execute(sql, (threshold, threshold, threshold)).fetchall()

    if not candidates:
        return 0

    promoted = 0
    for row in candidates:
        input_path = row["input_path"]
        conf_2 = row["confidence_2panel"] or 0.0
        conf_3 = row["confidence_3panel"] or 0.0
        conf_4 = row["confidence_4panel"] or 0.0

        # Determine split type — only consider enabled panel counts
        scores = {}
        if DETECT_2_PANEL and conf_2 >= threshold:
            scores[2] = conf_2
        if DETECT_3_PANEL and conf_3 >= threshold:
            scores[3] = conf_3
        if DETECT_4_PANEL and conf_4 >= threshold:
            scores[4] = conf_4

        if not scores:
            continue

        split_type = max(scores, key=scores.get)

        # Panel selection
        if force_panel is not None:
            panel_idx = force_panel
            densities = []
        else:
            frames = extract_sample_frames(input_path, sample_interval)
            if not frames:
                panel_idx = 0
                densities = []
            else:
                panel_idx, densities = select_cleanest_panel(frames, split_type)

        update_video(
            conn, input_path,
            status="split_detected",
            split_type=split_type,
            selected_panel=panel_idx,
            panel_edge_densities=json.dumps([round(d, 6) for d in densities]) if densities else None,
        )

        conf = scores[split_type]
        density_str = ", ".join(f"{d:.4f}" for d in densities) if densities else "forced"
        label = f"{row['subfolder']}/{row['filename']}"
        print(f"  ↑ {label} — {split_type}-PANEL ({conf:.1%}) — panel {panel_idx + 1} (edge: [{density_str}])")
        promoted += 1

    return promoted


# ==============================================================
#                        FFPROBE
# ==============================================================
def get_video_info(input_path: str) -> Tuple[float, int, int, int]:
    """Returns (duration, width, height, bitrate_kbps)"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,bit_rate:format=duration",
        "-of", "json", input_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        bitrate = int(stream.get("bit_rate", 6000000)) // 1000
        duration = float(fmt.get("duration", 0))
        return duration, width, height, bitrate
    except Exception:
        return 0, 0, 0, 0


# ==============================================================
#                  SPLIT DETECTION
# ==============================================================
def extract_sample_frames(video_path: str, interval_seconds: float) -> List[np.ndarray]:
    """Extract frames at regular intervals using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = max(1, int(fps * interval_seconds))
    frames = []

    for idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized cross-correlation on grayscale images."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    if g1.shape != g2.shape:
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = cv2.resize(g1, (w, h))
        g2 = cv2.resize(g2, (w, h))

    f1 = g1.astype(np.float32)
    f2 = g2.astype(np.float32)

    n1 = (f1 - f1.mean()) / (f1.std() + 1e-10)
    n2 = (f2 - f2.mean()) / (f2.std() + 1e-10)

    corr = np.mean(n1 * n2)
    return float(max(0.0, min(1.0, corr)))


def check_panels(frame: np.ndarray, num_panels: int) -> float:
    """Check if a frame has N identical panels. Returns average similarity."""
    h, w = frame.shape[:2]
    pw = w // num_panels
    panels = [frame[:, i * pw:(i + 1) * pw] for i in range(num_panels)]

    sims = [compute_similarity(panels[0], panels[i]) for i in range(1, num_panels)]
    return float(np.mean(sims)) if sims else 0.0


def detect_split(frames: List[np.ndarray], threshold: float) -> Tuple[Optional[int], float, float, float]:
    """
    Test frames for enabled panel splits.
    Returns: (split_type, confidence_2, confidence_3, confidence_4)
    """
    if not frames:
        return None, 0.0, 0.0, 0.0

    conf_2 = float(np.mean([check_panels(f, 2) for f in frames])) if DETECT_2_PANEL else 0.0
    conf_3 = float(np.mean([check_panels(f, 3) for f in frames])) if DETECT_3_PANEL else 0.0
    conf_4 = float(np.mean([check_panels(f, 4) for f in frames])) if DETECT_4_PANEL else 0.0

    detected = {}
    if DETECT_2_PANEL and conf_2 >= threshold:
        detected[2] = conf_2
    if DETECT_3_PANEL and conf_3 >= threshold:
        detected[3] = conf_3
    if DETECT_4_PANEL and conf_4 >= threshold:
        detected[4] = conf_4

    if not detected:
        return None, conf_2, conf_3, conf_4

    best = max(detected, key=detected.get)
    return best, conf_2, conf_3, conf_4


def format_conf_scores(conf_2: float, conf_3: float, conf_4: float) -> str:
    """Format confidence scores for display, showing only enabled detection types."""
    parts = []
    if DETECT_2_PANEL:
        parts.append(f"2p: {conf_2:.1%}")
    if DETECT_3_PANEL:
        parts.append(f"3p: {conf_3:.1%}")
    if DETECT_4_PANEL:
        parts.append(f"4p: {conf_4:.1%}")
    return ", ".join(parts)


# ==============================================================
#              CLEANEST PANEL SELECTION
# ==============================================================
def compute_edge_density(panel: np.ndarray) -> float:
    """Canny edge pixel count normalized by panel area."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if len(panel.shape) == 3 else panel
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges)) / edges.size


def select_cleanest_panel(frames: List[np.ndarray], num_panels: int) -> Tuple[int, List[float]]:
    """
    Select the panel with lowest edge density (least overlays/clutter).
    Returns: (0-indexed panel, list of avg edge densities per panel)
    """
    h, w = frames[0].shape[:2]
    pw = w // num_panels

    totals = [0.0] * num_panels

    for frame in frames:
        for i in range(num_panels):
            panel = frame[:, i * pw:(i + 1) * pw]
            totals[i] += compute_edge_density(panel)

    avg_densities = [t / len(frames) for t in totals]

    min_idx = int(np.argmin(avg_densities))
    min_val = avg_densities[min_idx]

    others = [avg_densities[i] for i in range(num_panels) if i != min_idx]
    if others:
        avg_others = np.mean(others)
        relative_diff = (avg_others - min_val) / (avg_others + 1e-10)
        if relative_diff < EDGE_DENSITY_MIN_DIFF:
            min_idx = 0

    return min_idx, avg_densities


# ==============================================================
#        DETECTION WORKER (runs in separate process)
# ==============================================================
def _detect_worker(args_tuple) -> DetectionResult:
    """
    Top-level picklable function for multiprocessing.
    No DB access — returns a DetectionResult.
    """
    input_path, subfolder, filename, threshold, sample_interval, max_duration, force_panel = args_tuple

    result = DetectionResult(
        input_path=input_path,
        subfolder=subfolder,
        filename=filename,
        status="error",
    )

    # Probe
    duration, width, height, bitrate = get_video_info(input_path)
    if duration == 0 or width == 0:
        result.error_message = "Failed to probe video"
        return result

    result.duration = duration
    result.width = width
    result.height = height
    result.bitrate = bitrate

    # Duration filter
    if duration > max_duration:
        result.status = "skipped_duration"
        return result

    # Extract frames
    frames = extract_sample_frames(input_path, sample_interval)
    if not frames:
        result.error_message = "No frames extracted"
        return result

    # Split detection
    split_type, conf_2, conf_3, conf_4 = detect_split(frames, threshold)
    result.conf_2 = conf_2
    result.conf_3 = conf_3
    result.conf_4 = conf_4

    if split_type is None:
        result.status = "no_split"
        return result

    result.split_type = split_type

    # Panel selection
    if force_panel is not None:
        result.panel_idx = force_panel
    else:
        panel_idx, densities = select_cleanest_panel(frames, split_type)
        result.panel_idx = panel_idx
        result.edge_densities = densities

    result.status = "split_detected"
    return result


# ==============================================================
#                   GPU DETECTION
# ==============================================================
def detect_gpu_encoder() -> bool:
    """Check if NVENC HEVC encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "hevc_nvenc" in result.stdout
    except Exception:
        return False


# ==============================================================
#             CROP + OPTIONAL UPSCALE + ENCODE
# ==============================================================
def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    src_width: int,
    src_height: int,
    num_panels: int,
    panel_idx: int,
    target_bitrate_kbps: int,
    use_gpu: bool,
    encoder_preset: str,
    upscale: bool,
) -> tuple:
    """Build FFmpeg command with crop + optional scale."""
    panel_width = src_width // num_panels
    x_offset = panel_idx * panel_width

    filter_parts = [f"crop={panel_width}:{src_height}:{x_offset}:0"]

    if upscale:
        scale_factor = TARGET_HEIGHT / src_height
        out_width = round(panel_width * scale_factor / 2) * 2
        out_height = TARGET_HEIGHT
        filter_parts.append(
            f"scale={out_width}:{out_height}:flags=lanczos+accurate_rnd+full_chroma_int"
        )
    else:
        # Ensure even dimensions for encoder compatibility
        out_width = panel_width - (panel_width % 2)
        out_height = src_height - (src_height % 2)
        if out_width != panel_width or out_height != src_height:
            filter_parts.append(f"scale={out_width}:{out_height}")

    filter_chain = ",".join(filter_parts)

    if use_gpu:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
            "-i", input_path,
            "-vf", filter_chain,
            "-c:v", "hevc_nvenc", "-preset", encoder_preset, "-rc", "vbr",
            "-b:v", f"{target_bitrate_kbps}k",
            "-maxrate", f"{target_bitrate_kbps}k",
            "-bufsize", f"{target_bitrate_kbps * 2}k",
            "-pix_fmt", "yuv420p10le",
            "-c:a", "copy",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
            "-tag:v", "hvc1",
            "-y", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-vf", filter_chain,
            "-c:v", "libx265",
            "-preset", encoder_preset,
            "-b:v", f"{target_bitrate_kbps}k",
            "-maxrate", f"{target_bitrate_kbps}k",
            "-bufsize", f"{target_bitrate_kbps * 2}k",
            "-pix_fmt", "yuv420p10le",
            "-c:a", "copy",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
            "-tag:v", "hvc1",
            "-y", output_path,
        ]

    return cmd, out_width, out_height


def run_ffmpeg_with_ui(
    cmd: list,
    duration_seconds: float,
    encoder_id: int,
    filename: str,
    ui: TerminalUI,
) -> None:
    """Run FFmpeg with progress reported to TerminalUI (parallel mode)."""
    progress_cmd = cmd.copy()
    y_idx = progress_cmd.index("-y")
    progress_cmd.insert(y_idx, "pipe:1")
    progress_cmd.insert(y_idx, "-progress")

    proc = subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t0 = time.time()

    try:
        for line in proc.stdout:
            if stop_requested:
                proc.kill()
                break
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    current_time_us = int(line.split("=")[1])
                except (ValueError, IndexError):
                    continue
                if duration_seconds > 0 and current_time_us > 0:
                    progress = min(current_time_us / (duration_seconds * 1_000_000), 1.0)
                    elapsed = time.time() - t0
                    eta = ((elapsed / progress) - elapsed) if progress > 0.01 else None
                    ui.update_encoder(encoder_id, filename, progress, elapsed, eta)
            elif line.startswith("progress=end"):
                break

        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


def run_ffmpeg_sequential(cmd: list, duration_seconds: float) -> None:
    """Run FFmpeg with simple sequential progress (legacy mode)."""
    progress_cmd = cmd.copy()
    y_idx = progress_cmd.index("-y")
    progress_cmd.insert(y_idx, "pipe:1")
    progress_cmd.insert(y_idx, "-progress")

    proc = subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t0 = time.time()

    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    current_time_us = int(line.split("=")[1])
                except (ValueError, IndexError):
                    continue
                if duration_seconds > 0 and current_time_us > 0:
                    progress = min(current_time_us / (duration_seconds * 1_000_000), 1.0)
                    elapsed = time.time() - t0
                    if progress > 0.01:
                        eta = (elapsed / progress) - elapsed
                        print(f"\r\033[K    ⏳ {progress:5.1%} — elapsed {elapsed:.0f}s — ETA {eta:.0f}s", end="", flush=True)
                    else:
                        print(f"\r\033[K    ⏳ {progress:5.1%} — elapsed {elapsed:.0f}s", end="", flush=True)
            elif line.startswith("progress=end"):
                break

        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    print("\r\033[K", end="", flush=True)

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


# ==============================================================
#                    SCANNING
# ==============================================================
def scan_input_root(input_root: str, conn, recursive: bool = False) -> List[Tuple[str, str, str]]:
    """
    Scan for video files and return new entries not yet in the DB.

    Default: scan only input_root for video files.
    Recursive: scan input_root and all subdirectories.

    Returns list of (input_path, relative_dir, filename) for NEW videos only.
    """
    root = Path(input_root)

    if recursive:
        scan_dirs = sorted({root} | {d for d in root.rglob("*") if d.is_dir()})
    else:
        scan_dirs = [root]

    total_dirs = len(scan_dirs)

    # Load cached directory counts
    if isinstance(conn, ThreadSafeDB):
        cached_rows = conn.fetchall("SELECT name, video_count FROM subfolders")
    else:
        cached_rows = conn.execute("SELECT name, video_count FROM subfolders").fetchall()
    cached = {r["name"]: r["video_count"] for r in cached_rows}

    new_entries = []
    skipped_dirs = 0
    scanned_dirs = 0
    ts = now_iso()

    for i, scan_dir in enumerate(scan_dirs, 1):
        rel_path = str(scan_dir.relative_to(root))

        current_files = [f.name for f in scan_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]

        if not current_files:
            continue

        current_count = len(current_files)
        is_cached = rel_path in cached and cached[rel_path] == current_count

        if is_cached:
            skipped_dirs += 1
            status = f"cached, {current_count} videos"
        else:
            scanned_dirs += 1
            for fname in sorted(current_files):
                fpath = str(scan_dir / fname)
                if isinstance(conn, ThreadSafeDB):
                    existing = conn.fetchone("SELECT status FROM videos WHERE input_path = ?", (fpath,))
                else:
                    existing = conn.execute("SELECT status FROM videos WHERE input_path = ?", (fpath,)).fetchone()
                if existing is None:
                    new_entries.append((fpath, rel_path, fname))

            sql = "INSERT OR REPLACE INTO subfolders (name, video_count, updated_at) VALUES (?, ?, ?)"
            if isinstance(conn, ThreadSafeDB):
                conn.execute(sql, (rel_path, current_count, ts))
            else:
                conn.execute(sql, (rel_path, current_count, ts))
                conn.commit()
            status = f"{current_count} videos, changed"

        display_name = rel_path if rel_path != "." else "(root)"
        print(f"\r\033[K  Scanning: {i}/{total_dirs} — {display_name} ({status})", end="", flush=True)

    print()

    # Detect deleted source files
    if isinstance(conn, ThreadSafeDB):
        all_known = conn.fetchall("SELECT input_path FROM videos WHERE status != 'deleted'")
    else:
        all_known = conn.execute("SELECT input_path FROM videos WHERE status != 'deleted'").fetchall()

    deleted_count = 0
    for row in all_known:
        if not os.path.exists(row["input_path"]):
            update_video(conn, row["input_path"], status="deleted")
            deleted_count += 1

    if skipped_dirs > 0 or scanned_dirs > 0 or deleted_count > 0:
        parts = []
        if scanned_dirs > 0:
            parts.append(f"{scanned_dirs} scanned")
        if skipped_dirs > 0:
            parts.append(f"{skipped_dirs} cached")
        if deleted_count > 0:
            parts.append(f"{deleted_count} source files gone")
        print(f"  Directories: {' | '.join(parts)}")

    return new_entries


# ==============================================================
#              ROUND-ROBIN SUBFOLDER INTERLEAVE
# ==============================================================
def round_robin_by_subfolder(rows: list) -> list:
    """
    Reorder rows so that processing rotates between subfolders,
    taking one video per subfolder per round.

    Preserves the original relative order within each subfolder.
    Works with both sqlite3.Row and dict-like objects (needs 'subfolder' key).
    """
    if not rows:
        return rows

    # Group rows by subfolder, preserving insertion order
    buckets: OrderedDict[str, list] = OrderedDict()
    for row in rows:
        sub = row["subfolder"]
        if sub not in buckets:
            buckets[sub] = []
        buckets[sub].append(row)

    # Interleave: take one from each bucket per round
    result = []
    while buckets:
        exhausted = []
        for sub in list(buckets.keys()):
            result.append(buckets[sub].pop(0))
            if not buckets[sub]:
                exhausted.append(sub)
        for sub in exhausted:
            del buckets[sub]

    return result


# ==============================================================
#              PARALLEL PIPELINE
# ==============================================================
def _encoder_thread(
    encoder_id: int,
    encode_queue: queue.Queue,
    db: ThreadSafeDB,
    output_root: str,
    ui: TerminalUI,
    stats: dict,
    stats_lock: threading.Lock,
    use_gpu: bool,
    encoder_preset: str,
    upscale: bool,
    bitrate_multiplier: float,
):
    """Encoder thread: pulls jobs from queue, encodes, updates DB."""
    while True:
        if stop_requested:
            break

        try:
            job: Optional[EncodeJob] = encode_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if job is None:  # sentinel — shutdown
            encode_queue.task_done()
            break

        label = f"{job.subfolder}/{job.filename}"
        base, ext = os.path.splitext(job.filename)
        output_filename = f"{base}_vertical{ext}"
        output_dir = os.path.normpath(os.path.join(output_root, job.subfolder))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        target_bitrate = max(MIN_BITRATE_KBPS, int(job.bitrate * bitrate_multiplier))

        db.update_video(job.input_path, status="processing", output_path=output_path)

        cmd, out_w, out_h = build_ffmpeg_cmd(
            job.input_path, output_path, job.width, job.height,
            job.split_type, job.panel_idx, target_bitrate,
            use_gpu, encoder_preset, upscale,
        )

        panel_w = job.width // job.split_type
        ui.log(f"  [ENC {encoder_id} {job.encode_idx}/{job.total_encodes}] {label} — {panel_w}x{job.height} → {out_w}x{out_h} @ {target_bitrate}k")

        ui.set_encoder_status(encoder_id, f"{job.filename} — starting...")

        t0 = time.time()
        try:
            run_ffmpeg_with_ui(cmd, job.duration, encoder_id, job.filename, ui)
            elapsed = time.time() - t0
            db.update_video(
                job.input_path,
                status="done",
                output_width=out_w,
                output_height=out_h,
                output_bitrate_kbps=target_bitrate,
            )
            ui.set_encoder_status(encoder_id, f"{job.filename} — done ({elapsed:.1f}s)")
            ui.log(f"  [ENC {encoder_id} {job.encode_idx}/{job.total_encodes}] {label} — done ({elapsed:.1f}s)")
            with stats_lock:
                stats["done"] = stats.get("done", 0) + 1
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            err_msg = e.stderr.strip()[-500:] if e.stderr else str(e)
            db.update_video(job.input_path, status="error", error_message=err_msg)
            ui.set_encoder_status(encoder_id, f"{job.filename} — error ({elapsed:.1f}s)")
            ui.log(f"  [ENC {encoder_id}] {label} — error: {err_msg[:150]}")
            if os.path.exists(output_path):
                os.remove(output_path)
            with stats_lock:
                stats["error"] = stats.get("error", 0) + 1
        finally:
            ui.set_encoder_status(encoder_id, "waiting...")
            encode_queue.task_done()

    ui.set_encoder_status(encoder_id, "stopped")


def run_parallel(
    db: ThreadSafeDB,
    actionable: list,
    pre_detected: list,
    output_root: str,
    threshold: float,
    sample_interval: float,
    max_duration: int,
    dry_run: bool,
    force_panel: Optional[int],
    num_workers: int,
    num_encoders: int,
    use_gpu: bool,
    encoder_preset: str,
    upscale: bool,
    bitrate_multiplier: float,
) -> dict:
    """Run the parallel detection + encode pipeline."""
    is_tty = sys.stdout.isatty()
    ui = TerminalUI(num_encoders=0 if dry_run else num_encoders, is_tty=is_tty)

    total = len(actionable)
    encode_queue: queue.Queue = queue.Queue()
    stats = {"done": 0, "no_split": 0, "error": 0, "split_detected": 0, "skipped_duration": 0}
    stats_lock = threading.Lock()

    # Start encoder threads (unless dry run)
    encoder_threads = []
    if not dry_run:
        ui.init_encoder_lines()
        for eid in range(1, num_encoders + 1):
            t = threading.Thread(
                target=_encoder_thread,
                args=(eid, encode_queue, db, output_root, ui, stats, stats_lock,
                      use_gpu, encoder_preset, upscale, bitrate_multiplier),
                daemon=True,
            )
            t.start()
            encoder_threads.append(t)

    # Enqueue pre-detected videos (split_detected from previous runs) directly
    encode_count = 0
    if pre_detected and not dry_run:
        for row in pre_detected:
            encode_count += 1
            job = EncodeJob(
                input_path=row["input_path"],
                subfolder=row["subfolder"],
                filename=row["filename"],
                duration=row["duration_seconds"],
                width=row["original_width"],
                height=row["original_height"],
                bitrate=row["original_bitrate_kbps"],
                split_type=row["split_type"],
                panel_idx=row["selected_panel"],
                encode_idx=encode_count,
                total_encodes=len(pre_detected),
            )
            encode_queue.put(job)
            label = f"{row['subfolder']}/{row['filename']}"
            ui.log(f"  [RESUME] {label} — {row['split_type']}-PANEL panel {row['selected_panel'] + 1} — queued for encoding")
        ui.log(f"  Queued {len(pre_detected)} pre-detected videos for encoding.\n")
    elif pre_detected and dry_run:
        ui.log(f"  {len(pre_detected)} pre-detected videos (dry run, skipping encode).\n")

    # Build detection work items
    work_items = [
        (row["input_path"], row["subfolder"], row["filename"],
         threshold, sample_interval, max_duration, force_panel)
        for row in actionable
    ]

    # Run detection in process pool
    width_total = len(str(total))

    with mp.Pool(processes=num_workers) as pool:
        results_iter = pool.imap_unordered(_detect_worker, work_items)

        completed = 0
        for result in results_iter:
            if stop_requested:
                pool.terminate()
                break

            completed += 1
            label = f"{result.subfolder}/{result.filename}"

            # Update DB with probe info
            if result.duration > 0:
                db.update_video(
                    result.input_path,
                    duration_seconds=result.duration,
                    original_width=result.width,
                    original_height=result.height,
                    original_bitrate_kbps=result.bitrate,
                    confidence_2panel=round(result.conf_2, 4),
                    confidence_3panel=round(result.conf_3, 4),
                    confidence_4panel=round(result.conf_4, 4),
                )

            if result.status == "error":
                db.update_video(result.input_path, status="error", error_message=result.error_message)
                ui.log(f"  [DET {completed:>{width_total}}/{total}] {label} — {result.error_message}")
                with stats_lock:
                    stats["error"] += 1

            elif result.status == "skipped_duration":
                db.update_video(result.input_path, status="skipped_duration")
                ui.log(f"  [DET {completed:>{width_total}}/{total}] {label} — {result.duration/60:.1f}m > limit — skipped")
                with stats_lock:
                    stats["skipped_duration"] += 1

            elif result.status == "no_split":
                db.update_video(result.input_path, status="no_split", split_type=None)
                conf_str = format_conf_scores(result.conf_2, result.conf_3, result.conf_4)
                ui.log(f"  [DET {completed:>{width_total}}/{total}] {label} — NO SPLIT ({conf_str}) — skipped")
                with stats_lock:
                    stats["no_split"] += 1

            elif result.status == "split_detected":
                density_str = ", ".join(f"{d:.4f}" for d in result.edge_densities) if result.edge_densities else "forced"
                conf = {2: result.conf_2, 3: result.conf_3, 4: result.conf_4}[result.split_type]

                db.update_video(
                    result.input_path,
                    status="split_detected",
                    split_type=result.split_type,
                    selected_panel=result.panel_idx,
                    panel_edge_densities=json.dumps([round(d, 6) for d in result.edge_densities]) if result.edge_densities else None,
                )

                ui.log(
                    f"  [DET {completed:>{width_total}}/{total}] {label} — "
                    f"{result.split_type}-PANEL ({conf:.1%}) — "
                    f"panel {result.panel_idx + 1} (edge: [{density_str}])"
                )

                with stats_lock:
                    stats["split_detected"] += 1

                if not dry_run:
                    encode_count += 1
                    job = EncodeJob(
                        input_path=result.input_path,
                        subfolder=result.subfolder,
                        filename=result.filename,
                        duration=result.duration,
                        width=result.width,
                        height=result.height,
                        bitrate=result.bitrate,
                        split_type=result.split_type,
                        panel_idx=result.panel_idx,
                        encode_idx=encode_count,
                        total_encodes=0,  # not known yet
                    )
                    encode_queue.put(job)

    # Detection complete — wait for encoders
    if not dry_run:
        if not stop_requested:
            ui.log(f"\n  Detection complete ({completed} videos). Waiting for encoder(s)...")

            # Send sentinels
            for _ in encoder_threads:
                encode_queue.put(None)
            for t in encoder_threads:
                t.join()
        else:
            # Drain and stop
            while not encode_queue.empty():
                try:
                    encode_queue.get_nowait()
                    encode_queue.task_done()
                except queue.Empty:
                    break
            for _ in encoder_threads:
                encode_queue.put(None)
            for t in encoder_threads:
                t.join(timeout=2)

        ui.clear_encoders()

    return stats


# ==============================================================
#         SEQUENTIAL PIPELINE (legacy, --workers 0)
# ==============================================================
def process_video(
    conn: sqlite3.Connection,
    input_path: str,
    subfolder: str,
    filename: str,
    output_root: str,
    threshold: float,
    sample_interval: float,
    max_duration: int,
    dry_run: bool,
    force_panel: Optional[int],
    idx: int,
    total: int,
    use_gpu: bool,
    encoder_preset: str,
    upscale: bool,
    bitrate_multiplier: float,
) -> str:
    """
    Full pipeline for a single video (sequential mode).
    Returns status string.
    """
    label = f"{subfolder}/{filename}"

    # --- Probe ---
    duration, width, height, bitrate = get_video_info(input_path)

    if duration == 0 or width == 0:
        update_video(conn, input_path, status="error", error_message="Failed to probe video")
        print(f"  [{idx}/{total}] {label} — probe failed — skipped")
        return "error"

    update_video(
        conn, input_path,
        duration_seconds=duration,
        original_width=width,
        original_height=height,
        original_bitrate_kbps=bitrate,
    )

    # --- Duration filter ---
    if duration > max_duration:
        update_video(conn, input_path, status="skipped_duration")
        print(f"  [{idx}/{total}] {label} — {duration/60:.1f}m > {max_duration/60:.0f}m limit — skipped")
        return "skipped_duration"

    # --- Sample frames ---
    frames = extract_sample_frames(input_path, sample_interval)
    if not frames:
        update_video(conn, input_path, status="error", error_message="No frames extracted")
        print(f"  [{idx}/{total}] {label} — no frames extracted — skipped")
        return "error"

    # --- Split detection ---
    split_type, conf_2, conf_3, conf_4 = detect_split(frames, threshold)

    update_video(
        conn, input_path,
        confidence_2panel=round(conf_2, 4),
        confidence_3panel=round(conf_3, 4),
        confidence_4panel=round(conf_4, 4),
    )

    if split_type is None:
        update_video(conn, input_path, status="no_split", split_type=None)
        conf_str = format_conf_scores(conf_2, conf_3, conf_4)
        print(f"  [{idx}/{total}] {label} — NO SPLIT ({conf_str}) — skipped")
        return "no_split"

    # --- Panel selection ---
    if force_panel is not None:
        panel_idx = force_panel
        densities = []
    else:
        panel_idx, densities = select_cleanest_panel(frames, split_type)

    density_str = ", ".join(f"{d:.4f}" for d in densities) if densities else "forced"

    update_video(
        conn, input_path,
        status="split_detected",
        split_type=split_type,
        selected_panel=panel_idx,
        panel_edge_densities=json.dumps([round(d, 6) for d in densities]) if densities else None,
    )

    conf = {2: conf_2, 3: conf_3, 4: conf_4}[split_type]
    print(f"  [{idx}/{total}] {label} — {split_type}-PANEL ({conf:.1%}) — panel {panel_idx + 1} selected (edge: [{density_str}])")

    if dry_run:
        return "split_detected"

    # --- Build output path ---
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_vertical{ext}"
    output_dir = os.path.normpath(os.path.join(output_root, subfolder))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # --- Encode ---
    target_bitrate = max(MIN_BITRATE_KBPS, int(bitrate * bitrate_multiplier))

    update_video(conn, input_path, status="processing", output_path=output_path)

    cmd, out_w, out_h = build_ffmpeg_cmd(
        input_path, output_path, width, height, split_type, panel_idx, target_bitrate,
        use_gpu, encoder_preset, upscale,
    )

    panel_w = width // split_type
    print(f"  [{idx}/{total}] {label} — {panel_w}x{height} → {out_w}x{out_h} @ {target_bitrate}k — encoding...")

    t0 = time.time()
    try:
        run_ffmpeg_sequential(cmd, duration)
        elapsed = time.time() - t0
        update_video(
            conn, input_path,
            status="done",
            output_width=out_w,
            output_height=out_h,
            output_bitrate_kbps=target_bitrate,
        )
        print(f"  [{idx}/{total}] {label} — done ({elapsed:.1f}s)")
        return "done"
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        err_msg = e.stderr.strip()[-500:] if e.stderr else str(e)
        update_video(conn, input_path, status="error", error_message=err_msg)
        print(f"  [{idx}/{total}] {label} — error ({elapsed:.1f}s)")
        print(f"    {err_msg[:200]}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return "error"


# ==============================================================
#                          MAIN
# ==============================================================
def main():
    default_workers = 2
    parser = argparse.ArgumentParser(
        description="Vertical Video Pipeline — detect split-screen, crop, upscale, encode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_root", type=str, help="Directory containing video files to process")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: output/ next to this script)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Split detection similarity threshold (default: {THRESHOLD})")
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL,
                        help=f"Frame sampling interval in seconds (default: {SAMPLE_INTERVAL})")
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION,
                        help=f"Skip videos longer than N seconds (default: {MAX_DURATION})")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite DB (default: <output_root>/pipeline.db)")
    parser.add_argument("--reset-errors", action="store_true",
                        help="Reset all errored videos back to pending")
    parser.add_argument("--rerun", choices=["missing", "all"], default=None,
                        help="Re-run mode: 'missing' = promote no_split videos that meet new threshold "
                             "(leave done alone); 'all' = promote no_split + reset done for full re-encode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect splits only, don't encode")
    parser.add_argument("--force-panel", type=int, default=None,
                        help="Override panel selection (0-indexed)")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Detection workers (default: {default_workers}, 0 = sequential legacy mode)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Scan input directory recursively (default: top-level only)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(ENCODE_PRESETS.keys()),
                        help=f"Encode preset (default: {ENCODE_PRESET})")
    parser.add_argument("--no-upscale", action="store_true",
                        help="Disable upscaling (keep native resolution after crop)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output) if args.output else os.path.join(script_dir, "output")
    db_path = args.db or os.path.join(output_root, "pipeline.db")
    is_sequential = args.workers == 0

    # Resolve encoding settings
    preset_name = args.preset or ENCODE_PRESET
    preset = ENCODE_PRESETS[preset_name]
    bitrate_multiplier = preset["bitrate_multiplier"]

    use_gpu = detect_gpu_encoder()
    encoder_preset = preset["gpu_preset"] if use_gpu else preset["cpu_preset"]

    upscale = UPSCALE_ENABLED and not args.no_upscale

    # Validate
    if not os.path.isdir(input_root):
        print(f"Error: Input root not found: {input_root}")
        sys.exit(1)

    if not any([DETECT_2_PANEL, DETECT_3_PANEL, DETECT_4_PANEL]):
        print("Error: At least one panel detection type must be enabled in config.")
        sys.exit(1)

    os.makedirs(output_root, exist_ok=True)

    # --- Init DB ---
    if is_sequential:
        conn = init_db_sequential(db_path)
    else:
        conn = ThreadSafeDB(db_path)

    if args.reset_errors:
        if is_sequential:
            count = conn.execute("SELECT COUNT(*) FROM videos WHERE status = 'error'").fetchone()[0]
            conn.execute("UPDATE videos SET status = 'pending', error_message = NULL, updated_at = ? WHERE status = 'error'", (now_iso(),))
            conn.commit()
        else:
            count = conn.fetchone("SELECT COUNT(*) as c FROM videos WHERE status = 'error'")["c"]
            conn.execute("UPDATE videos SET status = 'pending', error_message = NULL, updated_at = ? WHERE status = 'error'", (now_iso(),))
        print(f"Reset {count} errored videos to pending.\n")

    # --- Rerun logic ---
    if args.rerun:
        print(f"Re-run mode: {args.rerun} (threshold {args.threshold})")
        promoted = promote_no_split_videos(conn, args.threshold, args.sample_interval, args.force_panel)
        if promoted:
            print(f"Promoted {promoted} no_split videos to split_detected (threshold {args.threshold})\n")
        else:
            print(f"  No no_split videos qualify at threshold {args.threshold}.\n")

        if args.rerun == "all":
            reset_sql = "UPDATE videos SET status = 'pending', output_path = NULL, updated_at = ? WHERE status = 'done'"
            if is_sequential:
                done_count = conn.execute("SELECT COUNT(*) FROM videos WHERE status = 'done'").fetchone()[0]
                conn.execute(reset_sql, (now_iso(),))
                conn.commit()
            else:
                done_count = conn.fetchone("SELECT COUNT(*) as c FROM videos WHERE status = 'done'")["c"]
                conn.execute(reset_sql, (now_iso(),))
            if done_count:
                print(f"Reset {done_count} done videos to pending for re-encode.\n")

            reset_dur_sql = "UPDATE videos SET status = 'pending', updated_at = ? WHERE status = 'skipped_duration'"
            if is_sequential:
                dur_count = conn.execute("SELECT COUNT(*) FROM videos WHERE status = 'skipped_duration'").fetchone()[0]
                conn.execute(reset_dur_sql, (now_iso(),))
                conn.commit()
            else:
                dur_count = conn.fetchone("SELECT COUNT(*) as c FROM videos WHERE status = 'skipped_duration'")["c"]
                conn.execute(reset_dur_sql, (now_iso(),))
            if dur_count:
                print(f"Reset {dur_count} skipped_duration videos to pending.\n")

    # --- Scan ---
    print(f"Scanning {input_root}{'  (recursive)' if args.recursive else ''}...")
    new_entries = scan_input_root(input_root, conn, recursive=args.recursive)

    for input_path, subfolder, filename in new_entries:
        upsert_video(conn, input_path, subfolder, filename)

    if new_entries:
        print(f"  New videos found: {len(new_entries)}")

    # Check for done videos with missing output files
    if is_sequential:
        done_rows = conn.execute("SELECT input_path, output_path FROM videos WHERE status = 'done'").fetchall()
    else:
        done_rows = conn.fetchall("SELECT input_path, output_path FROM videos WHERE status = 'done'")

    reset_count = 0
    for row in done_rows:
        if row["output_path"] and not os.path.exists(row["output_path"]):
            update_video(conn, row["input_path"], status="pending", output_path=None)
            reset_count += 1
    if reset_count:
        print(f"  Reset {reset_count} done videos with missing output files.")

    # Status summary
    counts = get_status_counts(conn)
    total_count = sum(counts.values())
    active_total = total_count - counts.get("deleted", 0)
    parts = []
    for s in ["pending", "done", "no_split", "skipped_duration", "error", "split_detected", "processing"]:
        if counts.get(s, 0) > 0:
            parts.append(f"{counts[s]} {s}")

    if is_sequential:
        dir_count = conn.execute("SELECT COUNT(*) FROM subfolders").fetchone()[0]
    else:
        dir_count = conn.fetchone("SELECT COUNT(*) as c FROM subfolders")["c"]

    mode_str = "sequential" if is_sequential else f"parallel ({args.workers} detect + 1 encode)"
    rerun_str = f" | rerun={args.rerun}" if args.rerun else ""

    detect_types = []
    if DETECT_2_PANEL:
        detect_types.append("2-panel")
    if DETECT_3_PANEL:
        detect_types.append("3-panel")
    if DETECT_4_PANEL:
        detect_types.append("4-panel")

    encoder_name = "hevc_nvenc (GPU)" if use_gpu else "libx265 (CPU)"
    upscale_str = f"{TARGET_HEIGHT}p" if upscale else "off"

    print(f"  Total: {dir_count} directories, {active_total} videos ({', '.join(parts)})")
    print(f"  Mode: {mode_str}{rerun_str}")
    print(f"  Encoder: {encoder_name} | Preset: {preset_name}")
    print(f"  Upscale: {upscale_str} | Detection: {', '.join(detect_types)}")

    if args.dry_run:
        print("  DRY RUN — detection only, no encoding\n")
    else:
        print()

    # --- Get actionable videos ---
    if is_sequential:
        actionable = conn.execute(
            "SELECT input_path, subfolder, filename FROM videos WHERE status = 'pending' ORDER BY id ASC"
        ).fetchall()
    else:
        actionable = conn.fetchall(
            "SELECT input_path, subfolder, filename FROM videos WHERE status = 'pending' ORDER BY id ASC"
        )

    pre_detected_query = (
        "SELECT input_path, subfolder, filename, duration_seconds, "
        "original_width, original_height, original_bitrate_kbps, "
        "split_type, selected_panel "
        "FROM videos WHERE status = 'split_detected' ORDER BY id ASC"
    )
    if is_sequential:
        pre_detected = conn.execute(pre_detected_query).fetchall()
    else:
        pre_detected = conn.fetchall(pre_detected_query)

    # Reorder to rotate between subfolders
    actionable = round_robin_by_subfolder(actionable)
    pre_detected = round_robin_by_subfolder(pre_detected)

    if not actionable and not pre_detected:
        print("Nothing to process — all videos are already handled.")
        conn.close()
        return

    parts_msg = []
    if actionable:
        parts_msg.append(f"{len(actionable)} pending")
    if pre_detected:
        parts_msg.append(f"{len(pre_detected)} split_detected (resume)")
    print(f"Processing {' + '.join(parts_msg)} videos...\n")

    # --- Run pipeline ---
    if is_sequential:
        stats = {"done": 0, "no_split": 0, "error": 0, "split_detected": 0, "skipped_duration": 0}

        # Process pre-detected videos first (encode only)
        for i, row in enumerate(pre_detected, 1):
            if stop_requested:
                print("\nStopped by user.\n")
                break

            input_path = row["input_path"]
            subfolder = row["subfolder"]
            filename = row["filename"]
            label = f"{subfolder}/{filename}"
            width = row["original_width"]
            height = row["original_height"]
            bitrate = row["original_bitrate_kbps"]
            split_type = row["split_type"]
            panel_idx = row["selected_panel"]
            duration = row["duration_seconds"]

            if args.dry_run:
                print(f"  [RESUME {i}/{len(pre_detected)}] {label} — {split_type}-PANEL panel {panel_idx + 1} — dry run, skip")
                continue

            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_vertical{ext}"
            output_dir = os.path.normpath(os.path.join(output_root, subfolder))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)

            target_bitrate = max(MIN_BITRATE_KBPS, int(bitrate * bitrate_multiplier))
            update_video(conn, input_path, status="processing", output_path=output_path)

            cmd, out_w, out_h = build_ffmpeg_cmd(
                input_path, output_path, width, height, split_type, panel_idx, target_bitrate,
                use_gpu, encoder_preset, upscale,
            )

            panel_w = width // split_type
            print(f"  [RESUME {i}/{len(pre_detected)}] {label} — {panel_w}x{height} → {out_w}x{out_h} @ {target_bitrate}k — encoding...")

            t0 = time.time()
            try:
                run_ffmpeg_sequential(cmd, duration)
                elapsed = time.time() - t0
                update_video(
                    conn, input_path,
                    status="done",
                    output_width=out_w,
                    output_height=out_h,
                    output_bitrate_kbps=target_bitrate,
                )
                print(f"  [RESUME {i}/{len(pre_detected)}] {label} — done ({elapsed:.1f}s)")
                stats["done"] = stats.get("done", 0) + 1
            except subprocess.CalledProcessError as e:
                elapsed = time.time() - t0
                err_msg = e.stderr.strip()[-500:] if e.stderr else str(e)
                update_video(conn, input_path, status="error", error_message=err_msg)
                print(f"  [RESUME {i}/{len(pre_detected)}] {label} — error ({elapsed:.1f}s)")
                print(f"    {err_msg[:200]}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                stats["error"] = stats.get("error", 0) + 1

        # Process pending videos (full detection + encode)
        for i, row in enumerate(actionable, 1):
            if stop_requested:
                print("\nStopped by user.\n")
                break

            status = process_video(
                conn=conn,
                input_path=row["input_path"],
                subfolder=row["subfolder"],
                filename=row["filename"],
                output_root=output_root,
                threshold=args.threshold,
                sample_interval=args.sample_interval,
                max_duration=args.max_duration,
                dry_run=args.dry_run,
                force_panel=args.force_panel,
                idx=i,
                total=len(actionable),
                use_gpu=use_gpu,
                encoder_preset=encoder_preset,
                upscale=upscale,
                bitrate_multiplier=bitrate_multiplier,
            )
            stats[status] = stats.get(status, 0) + 1
    else:
        stats = run_parallel(
            db=conn,
            actionable=actionable,
            pre_detected=pre_detected,
            output_root=output_root,
            threshold=args.threshold,
            sample_interval=args.sample_interval,
            max_duration=args.max_duration,
            dry_run=args.dry_run,
            force_panel=args.force_panel,
            num_workers=args.workers,
            num_encoders=1,
            use_gpu=use_gpu,
            encoder_preset=encoder_preset,
            upscale=upscale,
            bitrate_multiplier=bitrate_multiplier,
        )

    if stop_requested:
        print("\nStopped by user.\n")

    # --- Summary ---
    final_counts = get_status_counts(conn)
    conn.close()

    print(f"\n{'=' * 50}")
    print("Complete!")
    print(f"  Total in DB:       {sum(final_counts.values())}")
    print(f"  Done:              {final_counts.get('done', 0)}")
    print(f"  No split (skip):   {final_counts.get('no_split', 0)}")
    print(f"  Too long (skip):   {final_counts.get('skipped_duration', 0)}")
    print(f"  Errors:            {final_counts.get('error', 0)}")
    if args.dry_run:
        print(f"  Splits detected:   {final_counts.get('split_detected', 0)}")
    print(f"  DB: {db_path}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
