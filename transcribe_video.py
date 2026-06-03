from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL


DEFAULT_MODEL = "small"
DEFAULT_OUTPUT_FORMATS = ("txt", "srt", "json")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    title: str
    audio_path: Path
    output_paths: dict[str, Path]
    metadata: dict


def slugify(value: str, fallback: str = "video") -> str:
    value = re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE).strip("._")
    return value[:80] or fallback


def _ffmpeg_candidates() -> list[Path]:
    candidates: list[Path] = []

    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))

    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
    ]
    for raw_prefix in prefixes:
        if not raw_prefix:
            continue
        prefix = Path(raw_prefix)
        candidates.extend(
            [
                prefix / "Library" / "bin" / "ffmpeg.exe",
                prefix / "bin" / "ffmpeg",
                prefix / "Scripts" / "ffmpeg.exe",
            ]
        )

    return candidates


def _imageio_ffmpeg_path() -> Path | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return path if path.exists() else None


def find_ffmpeg(explicit_path: str | None = None) -> Path:
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.exists():
            return explicit
        found = shutil.which(explicit_path)
        if found:
            return Path(found)
        raise RuntimeError(f"ffmpeg not found at: {explicit_path}")

    for candidate in _ffmpeg_candidates():
        if candidate.exists():
            return candidate

    imageio_path = _imageio_ffmpeg_path()
    if imageio_path:
        return imageio_path

    raise RuntimeError(
        "ffmpeg was not found. Install dependencies with "
        "'python -m pip install -r requirements.txt', or pass "
        "--ffmpeg C:\\path\\to\\ffmpeg.exe."
    )


def download_audio(url: str, download_dir: Path, ffmpeg_path: Path) -> tuple[Path, str]:
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(download_dir / "%(title).80s-%(id)s.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "ffmpeg_location": str(ffmpeg_path),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",
            }
        ],
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or "video"
        prepared = Path(ydl.prepare_filename(info))
        audio_path = prepared.with_suffix(".mp3")

    if not audio_path.exists():
        candidates = sorted(download_dir.glob(f"*{info.get('id', '')}*.mp3"))
        if candidates:
            audio_path = candidates[-1]

    if not audio_path.exists():
        raise RuntimeError("Audio download finished, but no mp3 file was found.")

    return audio_path, title


def normalize_audio(source: Path, target: Path, ffmpeg_path: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(target),
    ]
    subprocess.run(command, check=True)
    return target


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
    beam_size: int,
) -> tuple[list[Segment], dict]:
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    segments = [
        Segment(start=item.start, end=item.end, text=item.text.strip())
        for item in segments_iter
    ]
    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
    }
    return segments, metadata


def format_timestamp(seconds: float, srt: bool = False) -> str:
    millis = round(seconds * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}"


def write_txt(path: Path, title: str, segments: Iterable[Segment]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"{title}\n\n")
        for segment in segments:
            file.write(
                f"[{format_timestamp(segment.start)} -> "
                f"{format_timestamp(segment.end)}] {segment.text}\n"
            )


def write_srt(path: Path, segments: Iterable[Segment]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            file.write(f"{index}\n")
            file.write(
                f"{format_timestamp(segment.start, srt=True)} --> "
                f"{format_timestamp(segment.end, srt=True)}\n"
            )
            file.write(f"{segment.text}\n\n")


def write_json(path: Path, title: str, metadata: dict, segments: list[Segment]) -> None:
    payload = {
        "title": title,
        "metadata": metadata,
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe_video_url(
    url: str,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    ffmpeg: str | None = None,
    output_formats: Iterable[str] = DEFAULT_OUTPUT_FORMATS,
    keep_raw: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    def report(message: str) -> None:
        if progress:
            progress(message)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = find_ffmpeg(ffmpeg)
    formats = {item.lower().lstrip(".") for item in output_formats}

    report(f"Using ffmpeg: {ffmpeg_path}")
    report("Downloading audio...")
    audio_path, title = download_audio(url, output_dir, ffmpeg_path)
    base_name = slugify(title)
    normalized_path = output_dir / f"{base_name}.16k.mp3"

    report("Converting audio...")
    normalize_audio(audio_path, normalized_path, ffmpeg_path)

    if not keep_raw and audio_path != normalized_path:
        audio_path.unlink(missing_ok=True)

    report("Transcribing audio...")
    segments, metadata = transcribe_audio(
        normalized_path,
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
    )

    output_paths: dict[str, Path] = {}
    if "txt" in formats:
        txt_path = output_dir / f"{base_name}.txt"
        write_txt(txt_path, title, segments)
        output_paths["txt"] = txt_path
    if "srt" in formats:
        srt_path = output_dir / f"{base_name}.srt"
        write_srt(srt_path, segments)
        output_paths["srt"] = srt_path
    if "json" in formats:
        json_path = output_dir / f"{base_name}.json"
        write_json(json_path, title, metadata, segments)
        output_paths["json"] = json_path

    report("Done.")
    return TranscriptionResult(
        title=title,
        audio_path=normalized_path,
        output_paths=output_paths,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download video audio and transcribe it locally with faster-whisper."
    )
    parser.add_argument("url", help="Video URL supported by yt-dlp, such as YouTube or Bilibili")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name, default: small")
    parser.add_argument("--language", default=None, help="Language code, such as zh/en. Empty means auto-detect")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type. CPU: int8; NVIDIA GPU can try float16",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size. Higher may be more accurate but slower")
    parser.add_argument("--outputs-dir", default="outputs", help="Transcript output directory")
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional ffmpeg executable path. Useful when conda has ffmpeg but PATH does not.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw extracted mp3. The normalized recognition mp3 is always kept.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = transcribe_video_url(
            args.url,
            Path(args.outputs_dir),
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            ffmpeg=args.ffmpeg,
            output_formats=DEFAULT_OUTPUT_FORMATS,
            keep_raw=args.keep_raw,
            progress=print,
        )

        print("Done. Output files:")
        print(f"- {result.audio_path}")
        for path in result.output_paths.values():
            print(f"- {path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
