"""Video and frame-sampling helpers for trajectory replay."""

import os
import shutil
import subprocess

import cv2
import numpy as np


def _as_numpy_frame(frame):
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu().numpy()
    return frame


def normalize_video_frames(frames):
    frames = _as_numpy_frame(frames)
    if isinstance(frames, np.ndarray):
        return list(frames)
    return [_as_numpy_frame(frame) for frame in frames or []]


def linspace_indices(start, end, count):
    if count <= 0 or end < start:
        return []
    available = end - start + 1
    if count >= available:
        return list(range(start, end + 1))
    indices = np.rint(np.linspace(start, end, count)).astype(int).tolist()
    deduped = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    cursor = start
    while len(deduped) < count and cursor <= end:
        if cursor not in deduped:
            deduped.append(cursor)
        cursor += 1
    return sorted(deduped)


def sample_frame_indices(num_frames, sample_count=6, include_initial=True):
    num_frames = int(num_frames)
    if num_frames <= 0:
        return []
    sample_count = max(int(sample_count), 0)
    if include_initial:
        indices = [0]
        indices.extend(linspace_indices(1, num_frames - 1, sample_count))
        return indices
    return linspace_indices(0, num_frames - 1, sample_count)


def encode_video_frames(frames, video_path, fps):
    if frames is None or len(frames) == 0:
        print(f"No frames captured for video {video_path}")
        return False

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    first_frame = np.asarray(frames[0])
    if first_frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape for video encoding: {first_frame.shape}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    try:
        for frame in frames:
            frame_np = np.asarray(frame)
            frame_np = np.ascontiguousarray(frame_np[..., :3]).astype(np.uint8, copy=False)
            writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return True


def transcode_video_ffmpeg(input_path, output_stem):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None, "ffmpeg not found on PATH"

    output_candidates = [
        (
            output_stem + ".mp4",
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_stem + ".mp4",
            ],
        ),
        (
            output_stem + ".webm",
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-an",
                "-c:v",
                "libvpx",
                "-pix_fmt",
                "yuv420p",
                output_stem + ".webm",
            ],
        ),
    ]

    last_error = None
    for output_path, cmd in output_candidates:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None
        last_error = result.stderr.strip() or result.stdout.strip() or f"ffmpeg failed for {output_path}"
        if os.path.exists(output_path):
            os.remove(output_path)

    return None, last_error
