"""
后端 API 服务（与前端分离）。

接口：
- POST /api/upload             multipart 上传视频 → 后台跑完整流水线 → {"job_id", "name"}
- GET  /api/status?id=         → {"status": queued/running/done/error, "name", "log": [...]}
- GET  /api/results            → 已分析结果列表
- GET  /api/result?name=       → 完整分析 JSON
- POST /api/correct            → 人工纠错写回（urlencoded 或 multipart）
- GET  /api/export?name=&format=md|json → 文件下载
- GET  /                      → 静态前端 frontend/index.html

启动: python web_ui.py --port 8800
只依赖 Python 标准库。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import requests

from first_layer.extractors import describe_registry
from second_layer.prompts import get_all_prompts, reset_prompts, save_prompts

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
VIDEOS_DIR = ROOT / "videos"
OUTPUTS_DIR = ROOT / "outputs"
_UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "web_uploads"

ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB
# 本地归档的类别文件夹（与 LLM 分类体系一致）
CATEGORY_FOLDERS = ("歌曲", "美食", "美文", "其他")

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_COUNTER = 0
JOBS_FILE = OUTPUTS_DIR / "_jobs_state.json"

# 任务队列 + 动态并发控制：并发数可由网页（concurrent 字段）随时调整，
# 默认取启动参数 --jobs；工作线程池固定 8 个（多出的会等空位）
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
MAX_SLOT_WORKERS = 8
_SLOT_COND = threading.Condition()
_ACTIVE_JOBS = 0
_MAX_CONCURRENT = 2


def set_concurrency(n: int) -> None:
    """调整并行分析任务数（1~8）"""
    global _MAX_CONCURRENT
    with _SLOT_COND:
        _MAX_CONCURRENT = max(1, min(MAX_SLOT_WORKERS, int(n)))
        _SLOT_COND.notify_all()


def get_concurrency() -> int:
    with _SLOT_COND:
        return _MAX_CONCURRENT


def _acquire_slot() -> None:
    global _ACTIVE_JOBS
    with _SLOT_COND:
        while _ACTIVE_JOBS >= _MAX_CONCURRENT:
            _SLOT_COND.wait()
        _ACTIVE_JOBS += 1


def _release_slot() -> None:
    global _ACTIVE_JOBS
    with _SLOT_COND:
        _ACTIVE_JOBS -= 1
        _SLOT_COND.notify_all()


def new_job(name: str, config: dict[str, Any] | None = None) -> str:
    global JOB_COUNTER
    with JOBS_LOCK:
        JOB_COUNTER += 1
        job_id = f"{int(time.time())}-{JOB_COUNTER}"
        JOBS[job_id] = {
            "id": job_id,
            "name": name,
            "status": "queued",
            "log": deque(maxlen=200),
            "config": config or {},
        }
    _persist_jobs()
    return job_id


def _persist_jobs() -> None:
    """把任务状态持久化到磁盘，服务重启后可恢复未完成任务"""
    try:
        data: dict[str, Any] = {}
        with JOBS_LOCK:
            for jid, j in JOBS.items():
                data[jid] = {
                    "name": j.get("name", ""),
                    "video_path": j.get("video_path", ""),
                    "status": j.get("status", "queued"),
                    "config": j.get("config", {}),
                    "duration": j.get("duration", 0),
                    "log": list(j.get("log", deque()))[-100:],
                }
        tmp = JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(JOBS_FILE)
    except Exception:
        pass


def _load_jobs() -> None:
    """服务启动时恢复任务：未完成且视频文件还在的重新入队"""
    try:
        if not JOBS_FILE.exists():
            return
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        global JOB_COUNTER
        for jid, j in data.items():
            status = j.get("status")
            if status in ("done", "error"):
                JOBS[jid] = {
                    "id": jid, "name": j.get("name", ""),
                    "status": status,
                    "log": deque(j.get("log", []), maxlen=200),
                    "config": j.get("config", {}),
                }
                continue
            vp = j.get("video_path", "")
            if vp and Path(vp).exists():
                job = {
                    "id": jid, "name": j.get("name", ""),
                    "status": "queued",
                    "log": deque(list(j.get("log", [])) + ["[server] 服务重启，自动恢复任务"], maxlen=200),
                    "config": j.get("config", {}),
                    "video_path": vp,
                    "duration": j.get("duration", 0),
                }
                JOBS[jid] = job
                JOB_QUEUE.put(jid)
        _persist_jobs()
    except Exception:
        pass


def run_job(job_id: str, video_path: Path) -> None:
    """后台线程：按用户选择的模型配置跑完整流水线，捕获日志"""
    job = JOBS[job_id]
    job["status"] = "running"
    _persist_jobs()
    cfg: dict[str, Any] = job.get("config") or {}
    cmd = [
        sys.executable, str(ROOT / "run_pipeline.py"),
        "--input", str(video_path),
        "--no-hf-mirror", "--no-ocr-download",
    ]
    # 手动选择的模型开关
    if not cfg.get("asr", True):
        cmd.append("--no-asr")
    if not cfg.get("ocr", True):
        cmd.append("--no-ocr")
    if not cfg.get("visual", True):
        cmd.append("--no-visual")
    if not cfg.get("audio", True):
        cmd.append("--no-audio")
    # 通用开关（注册表驱动，支持未来新增的模型 id）
    disable_ids = [x.strip() for x in cfg.get("disable", []) if x.strip()]
    if disable_ids:
        cmd += ["--disable-model", ",".join(sorted(disable_ids))]
    if cfg.get("visual_backend") in ("clip", "vlm"):
        cmd += ["--visual-backend", cfg["visual_backend"]]
    if cfg.get("audio_backend") in ("fingerprint", "shazamio", "dejavu"):
        cmd += ["--audio-backend", cfg["audio_backend"]]
    if cfg.get("llm_backend") in ("openai", "hpc_qwen"):
        cmd += ["--llm-backend", cfg["llm_backend"]]
    # 超参数
    try:
        ki = float(cfg.get("keyframe_interval") or 2.0)
    except (TypeError, ValueError):
        ki = 2.0
    if 0.2 <= ki <= 30:
        cmd += ["--keyframe-interval", f"{ki:.1f}"]
    try:
        mr = int(cfg.get("max_retries") or 2)
    except (TypeError, ValueError):
        mr = 2
    if 0 <= mr <= 10:
        cmd += ["--max-retries", str(mr)]
    try:
        temp = float(cfg.get("temperature") or 0.0)
    except (TypeError, ValueError):
        temp = 0.0
    if 0 <= temp <= 2:
        cmd += ["--temperature", str(temp)]
    # 新增超参数：ASR 规格 / OCR 语言 / 时长上限 / LLM 模型名 / LLM 超时
    if cfg.get("asr_model") in ("tiny", "base", "small", "medium", "large-v3"):
        cmd += ["--asr-model", cfg["asr_model"]]
    if cfg.get("ocr_lang") in ("ch", "cht", "en"):
        cmd += ["--ocr-lang", cfg["ocr_lang"]]
    try:
        md = float(cfg.get("max_duration_minutes") or 10.0)
    except (TypeError, ValueError):
        md = 10.0
    if 0 <= md <= 1440:
        cmd += ["--max-duration-minutes", str(md)]
    if cfg.get("llm_model"):
        cmd += ["--model", str(cfg["llm_model"])]
    try:
        mt = float(cfg.get("model_timeout") or 300.0)
    except (TypeError, ValueError):
        mt = 300.0
    if 5 <= mt <= 1800:
        cmd += ["--model-timeout", str(mt)]
    # 性能加速参数
    try:
        beam = int(cfg.get("asr_beam") or 5)
    except (TypeError, ValueError):
        beam = 5
    if beam in (1, 2, 5):
        cmd += ["--asr-beam", str(beam)]
    try:
        thr = int(cfg.get("asr_threads") or 4)
    except (TypeError, ValueError):
        thr = 4
    if 1 <= thr <= 32:
        cmd += ["--asr-threads", str(thr)]
    try:
        vmf = int(cfg.get("vlm_max_frames") or 12)
    except (TypeError, ValueError):
        vmf = 12
    if 2 <= vmf <= 60:
        cmd += ["--vlm-max-frames", str(vmf)]
    try:
        oms = int(cfg.get("ocr_max_side") or 0)
    except (TypeError, ValueError):
        oms = 0
    if oms == 0 or 320 <= oms <= 4096:
        cmd += ["--ocr-max-side", str(oms)]
    try:
        omf = int(cfg.get("ocr_max_frames") or 0)
    except (TypeError, ValueError):
        omf = 0
    if omf == 0 or 2 <= omf <= 200:
        cmd += ["--ocr-max-frames", str(omf)]
    try:
        # 子进程在 Windows 控制台默认用 GBK/cp936 输出，按系统编码解码避免乱码
        enc = locale.getpreferredencoding(False) or "utf-8"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding=enc,
            errors="replace",
            cwd=str(ROOT),
        )
        for line in proc.stdout or []:
            job["log"].append(line.rstrip())
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
        job["exit_code"] = proc.returncode
        _persist_jobs()
        if job["status"] == "done":
            # 分析完成后按类别归档：outputs/<类别>/ 与 videos/<类别>/
            try:
                _organize_by_category(Path(video_path).stem)
                job["log"].append("[server] 已按类别归档到 outputs/<类别>/ 与 videos/<类别>/")
            except Exception as e:
                job["log"].append(f"[server] 分类归档失败: {e}")
    except Exception as e:
        job["log"].append(f"[server] 启动失败: {e}")
        job["status"] = "error"
        _persist_jobs()


def start_job(job_id: str, video_path: Path) -> None:
    """把任务放入队列，由工作线程池按并发数调度执行"""
    with JOBS_LOCK:
        JOBS[job_id]["video_path"] = str(video_path)
        JOBS[job_id]["duration"] = _probe_duration(video_path)
    JOB_QUEUE.put(job_id)
    _persist_jobs()


def _job_worker() -> None:
    """工作线程：从队列取任务，拿到并发空位后执行"""
    while True:
        job_id = JOB_QUEUE.get()
        try:
            _acquire_slot()
            try:
                job = JOBS.get(job_id)
                if job is None or not Path(job.get("video_path", "")).exists():
                    continue
                run_job(job_id, Path(job["video_path"]))
            except Exception as e:
                job = JOBS.get(job_id)
                if job is not None:
                    job["log"].append(f"[server] 执行失败: {e}")
                    job["status"] = "error"
                    _persist_jobs()
            finally:
                _release_slot()
        finally:
            JOB_QUEUE.task_done()


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    """极简 multipart/form-data 解析（单文件 + 少量字段），兼容 Python 3.13（无 cgi）"""
    result: dict[str, tuple[str, bytes]] = {}
    if "boundary=" not in content_type:
        return result
    boundary = content_type.split("boundary=")[-1].strip().strip('"').encode()
    for part in body.split(b"--" + boundary):
        if not part or part in (b"--", b"\r\n"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        content = content.rstrip(b"\r\n")
        name = filename = ""
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for piece in line.split(";"):
                    piece = piece.strip()
                    if piece.lower().startswith("name="):
                        name = piece.split("=", 1)[1].strip('"')
                    elif piece.lower().startswith("filename="):
                        filename = piece.split("=", 1)[1].strip('"')
        if name:
            result[name] = (filename, content)
    return result


def list_results() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not OUTPUTS_DIR.exists():
        return items
    seen: set[str] = set()

    def sort_key(p: Path):
        # 根目录（新分析中）优先，分类子目录次之
        return (0 if p.parent == OUTPUTS_DIR else 1, str(p))

    for p in sorted(OUTPUTS_DIR.rglob("*_evidence_analysis.json"), key=sort_key):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = p.name.replace("_evidence_analysis.json", "")
        if name in seen:
            continue
        seen.add(name)
        video_file = _find_video(name)
        items.append({
            "name": name,
            "category": data.get("category", "?"),
            "reason": (data.get("reason") or "")[:120],
            "qc_passed": (data.get("qc") or {}).get("passed"),
            "organized": (data.get("organized") or {}).get("schema"),
            "has_correction": "human_correction" in data,
            "video_exists": video_file is not None,
        })
    return items


_TRANSCODE_CACHE: dict[str, Path | None] = {}


def _web_playable(src: Path) -> Path | None:
    """非 H.264 视频（如 HEVC）在 Edge 等浏览器无法播放，按需转码为 H.264 缓存。
    返回可播放文件路径；探测/转码失败返回 None（回退原文件）。"""
    import imageio_ffmpeg

    key = f"{src.resolve()}:{src.stat().st_mtime}"
    if key in _TRANSCODE_CACHE:
        return _TRANSCODE_CACHE[key]
    try:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run(
            [ff, "-i", str(src)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if "Video: h264" in probe.stderr:
            _TRANSCODE_CACHE[key] = src
            return src
        cache_dir = Path(tempfile.gettempdir()) / "web_video_cache"
        cache_dir.mkdir(exist_ok=True)
        out = cache_dir / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".mp4")
        if not out.exists():
            r = subprocess.run(
                [ff, "-y", "-i", str(src), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                 "-movflags", "+faststart", str(out)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if r.returncode != 0 or not out.exists():
                _TRANSCODE_CACHE[key] = None
                return None
        _TRANSCODE_CACHE[key] = out
        return out
    except Exception:
        _TRANSCODE_CACHE[key] = None
        return None


def _find_video(name: str) -> Path | None:
    """按名字找视频（根目录优先：新上传/分析中；其次 videos/<类别>/ 子目录）"""
    if not VIDEOS_DIR.exists():
        return None
    for p in sorted(VIDEOS_DIR.glob(name + ".*")):
        if p.is_file():
            return p
    for p in sorted(VIDEOS_DIR.rglob(name + ".*")):
        if p.is_file():
            return p
    return None


def _probe_duration(path: Path) -> float:
    """用 ffprobe 探测视频时长（秒），失败返回 0（用于队列 ETA 估算）"""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run(
            [ff, "-i", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        for line in (r.stderr or "").splitlines():
            if "Duration:" in line:
                t = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
                h, m, s = t.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


_ASR_FACTOR = {"tiny": 0.05, "base": 0.15, "small": 0.3, "medium": 0.8}


def _estimate_seconds(duration: float, config: dict[str, Any] | None) -> float:
    """粗略估算单个分析任务耗时（秒），用于队列 ETA 展示"""
    cfg = config or {}
    dur = duration if duration and duration > 0 else 30.0
    try:
        ki = float(cfg.get("keyframe_interval") or 2.0)
    except (TypeError, ValueError):
        ki = 2.0
    frames = max(1, int(dur / max(ki, 0.5)))
    asr_model = cfg.get("asr_model") or "medium"
    asr_t = dur * _ASR_FACTOR.get(asr_model, 0.8)
    vb = cfg.get("visual_backend") or "clip"
    if vb == "vlm":
        try:
            vmf = int(cfg.get("vlm_max_frames") or 12)
        except (TypeError, ValueError):
            vmf = 12
        visual_t = min(frames, vmf) * 40.0  # VLM 每帧约 40 秒
    else:
        visual_t = frames * 0.3  # CLIP 秒级
    try:
        omf = int(cfg.get("ocr_max_frames") or 0)
    except (TypeError, ValueError):
        omf = 0
    ocr_frames = min(frames, omf) if omf > 0 else frames
    ocr_t = ocr_frames * 1.5  # OCR 每帧约 1.5 秒
    # ASR 与 视觉/OCR 并行，取较大者 + 固定开销
    return max(asr_t, visual_t + ocr_t) + 8.0


def _find_analysis_path(name: str) -> Path | None:
    """按名字找分析结果 JSON（根目录优先，其次 outputs/<类别>/ 子目录）"""
    p = OUTPUTS_DIR / f"{name}_evidence_analysis.json"
    if p.exists():
        return p
    for q in sorted(OUTPUTS_DIR.rglob(f"{name}_evidence_analysis.json")):
        if q.is_file():
            return q
    return None


def get_analysis(name: str) -> dict[str, Any] | None:
    p = _find_analysis_path(name)
    if p is None:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _delete_record(name: str) -> dict[str, int]:
    """删除某条记录的分析产出与对应视频文件，并清理任务状态"""
    files = videos = 0
    for suffix in ("_evidence.json", "_evidence_analysis.json", "_evidence_analysis.md"):
        for p in OUTPUTS_DIR.rglob(name + suffix):
            try:
                p.unlink()
                files += 1
            except OSError:
                pass
    for p in VIDEOS_DIR.rglob(name + ".*"):
        try:
            p.unlink()
            videos += 1
        except OSError:
            pass
    with JOBS_LOCK:
        for jid in [j for j in JOBS if JOBS[j].get("name") == name]:
            JOBS.pop(jid, None)
    _persist_jobs()
    return {"files": files, "videos": videos}


_SYSINFO_CACHE: dict[str, Any] | None = None


def get_sysinfo() -> dict[str, Any]:
    """本机硬件信息 + 根据配置推荐的超参数（供前端预设自适应）"""
    global _SYSINFO_CACHE
    if _SYSINFO_CACHE is not None:
        return _SYSINFO_CACHE
    cores = os.cpu_count() or 4
    total_gb = avail_gb = 0.0
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        total_gb = round(m.ullTotalPhys / 1024 ** 3, 1)
        avail_gb = round(m.ullAvailPhys / 1024 ** 3, 1)
    except Exception:
        total_gb = avail_gb = 0.0
    has_gpu = False
    try:
        import torch
        has_gpu = bool(torch.cuda.is_available())
    except Exception:
        pass
    # ASR 线程：whisper 在 4~8 线程收益最大，按核数取
    rec_threads = max(2, min(cores, 8))
    # 并行任务数：按核数与内存估算（每个任务约需 2~4GB，VLM 更多）
    if cores <= 4 or (total_gb and total_gb <= 8):
        rec_concurrent = 1
    elif cores <= 8:
        rec_concurrent = 2
    elif cores <= 16:
        rec_concurrent = 3
    else:
        rec_concurrent = 4
    if total_gb > 0:
        rec_concurrent = min(rec_concurrent, max(1, int(total_gb / 6)))
    _SYSINFO_CACHE = {
        "cpu_cores": cores,
        "ram_total_gb": total_gb,
        "ram_avail_gb": avail_gb,
        "has_gpu": has_gpu,
        "rec_threads": rec_threads,
        "rec_concurrent": max(1, rec_concurrent),
    }
    return _SYSINFO_CACHE


def _organize_by_category(name: str) -> None:
    """分析完成后：把输出与视频归档到 outputs/<类别>/ 与 videos/<类别>/（本地分类保存）。
    重新识别可能改变类别：会清理其它类别目录里残留的同名文件，并把视频从原目录移到新类别。"""
    ap = _find_analysis_path(name)
    category = "其他"
    if ap is not None:
        try:
            data = json.loads(ap.read_text(encoding="utf-8"))
            cat = str(data.get("category") or "").strip()
            if cat in CATEGORY_FOLDERS:
                category = cat
        except Exception:
            pass
    errors: list[str] = []
    out_dir = OUTPUTS_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 清理其它类别目录里残留的同名产物（重新识别后类别变化，避免留下旧副本）
    for other in OUTPUTS_DIR.iterdir():
        if not other.is_dir() or other.name not in CATEGORY_FOLDERS or other.name == category:
            continue
        for suffix in ("_evidence.json", "_evidence_analysis.json", "_evidence_analysis.md"):
            stale = other / (name + suffix)
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

    # 2) 把本次新产物（根目录）移动到目标类别目录
    for suffix in ("_evidence.json", "_evidence_analysis.json", "_evidence_analysis.md"):
        src = OUTPUTS_DIR / (name + suffix)
        if not src.exists():
            continue
        try:
            dst = out_dir / src.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    # 3) 把视频从根目录或任何类别子目录移到新类别目录
    vid_dir = VIDEOS_DIR / category
    vid_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(VIDEOS_DIR.rglob(name + ".*")):
        if not src.is_file():
            continue
        if src.parent == vid_dir:
            continue
        try:
            dst = vid_dir / src.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        except Exception as e:
            errors.append(f"{src.name}: {e}")
    if errors:
        raise RuntimeError("；".join(errors))


# ---------------------------------------------------------------------------
# 信息提取：基于已分析历史，聚合歌单/菜谱/文案等结构化信息
# ---------------------------------------------------------------------------

def aggregate_analysis() -> dict[str, Any]:
    """扫描所有已分析结果，按类别提取结构化信息"""
    agg: dict[str, Any] = {"歌曲": [], "美食": [], "美文": [], "其他": [], "total": 0}
    if not OUTPUTS_DIR.exists():
        return agg
    seen: set[str] = set()
    for p in sorted(OUTPUTS_DIR.rglob("*_evidence_analysis.json")):
        name = p.name.replace("_evidence_analysis.json", "")
        if name in seen:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen.add(name)
        agg["total"] += 1
        org = data.get("organized") or {}
        schema = str(org.get("schema") or data.get("category") or "")
        prov = org.get("provenance") or {}
        entry: dict[str, Any] = {"name": name, "reason": (data.get("reason") or "")[:200]}
        if schema == "歌曲":
            for f in ("song_name", "artist", "version"):
                entry[f] = org.get(f, "")
                e = prov.get(f) or {}
                if e:
                    entry[f + "_src"] = e.get("source", "")
            agg["歌曲"].append(entry)
        elif schema == "美食":
            entry["dish_name"] = org.get("dish_name", "")
            entry["ingredients"] = list(org.get("ingredients") or [])
            entry["steps"] = [
                {"step": s.get("step"), "description": s.get("description", "")}
                if isinstance(s, dict) else str(s)
                for s in (org.get("steps") or [])
            ]
            e = prov.get("dish_name") or {}
            entry["dish_name_src"] = e.get("source", "")
            agg["美食"].append(entry)
        elif schema == "美文":
            entry["original_text"] = org.get("original_text", "")
            entry["author"] = org.get("author", "")
            e = prov.get("author") or {}
            entry["author_src"] = e.get("source", "")
            agg["美文"].append(entry)
        else:
            entry["summary"] = org.get("summary") or org.get("notes") or ""
            agg["其他"].append(entry)
    return agg


def build_aggregate_md(agg: dict[str, Any]) -> str:
    """生成合集 Markdown：歌单 / 菜谱 / 文案 / 其他摘要"""
    lines: list[str] = [
        "# 📦 抖音点赞视频 · 信息提取合集", "",
        f"共分析 {agg['total']} 个视频：歌曲 {len(agg['歌曲'])} 个 / 美食 {len(agg['美食'])} 个 / "
        f"美文 {len(agg['美文'])} 个 / 其他 {len(agg['其他'])} 个", "",
        "## 🎵 歌单",
    ]
    if agg["歌曲"]:
        for e in agg["歌曲"]:
            title = e.get("song_name") or "未知歌名"
            artist = (" - " + e["artist"]) if e.get("artist") else ""
            ver = ("（" + e["version"] + "）") if e.get("version") else ""
            src = ("，出处：" + e["song_name_src"]) if e.get("song_name_src") else ""
            lines.append(f"- 《{title}》{artist}{ver} — 来源视频：《{e['name']}》{src}")
    else:
        lines.append("- （暂无）")
    lines += ["", "## 🍜 菜谱合集"]
    if agg["美食"]:
        for e in agg["美食"]:
            src = ("，出处：" + e["dish_name_src"]) if e.get("dish_name_src") else ""
            lines.append(f"### {e.get('dish_name') or '未知菜名'}（来源视频：《{e['name']}》{src}）")
            if e.get("ingredients"):
                lines.append("**食材**：" + "、".join(e["ingredients"]))
            for s in e.get("steps") or []:
                if isinstance(s, dict) and s.get("description"):
                    lines.append(f"{s.get('step', '')}. {s['description']}".lstrip(". "))
            if e.get("notes"):
                lines.append(f"> {e['notes']}")
            lines.append("")
    else:
        lines += ["- （暂无）", ""]
    lines += ["## 📖 美文摘抄"]
    if agg["美文"]:
        for e in agg["美文"]:
            author = (" — " + e["author"]) if e.get("author") else ""
            lines.append(f"### 《{e['name']}》{author}")
            if e.get("original_text"):
                lines.append("> " + e["original_text"].replace("\n", "\n> "))
            lines.append("")
    else:
        lines += ["- （暂无）", ""]
    lines += ["## 📦 其他视频摘要"]
    if agg["其他"]:
        for e in agg["其他"]:
            lines.append(f"- **{e['name']}**：{e.get('summary') or e.get('reason') or ''}")
    else:
        lines.append("- （暂无）")
    return "\n".join(lines)


def summarize_category(category: str, backend: str = "") -> str:
    """把某一类全部视频的提取信息交给 LLM，生成中文总结报告（Markdown 文本）"""
    from second_layer.llm_analyzer import (
        BACKEND_HPC_QWEN,
        BACKEND_OPENAI,
        DEFAULT_BASE_URL,
        DEFAULT_HPC_QWEN_URL,
        DEFAULT_MODEL,
        HPC_QWEN_MESSAGE_KEY,
        _post_hpc_chat,
        load_dotenv,
    )

    agg = aggregate_analysis()
    items = agg.get(category) or []
    if not items:
        return "该类目前还没有分析记录。"
    load_dotenv()
    backend = backend or os.environ.get("LLM_BACKEND") or BACKEND_OPENAI
    items_payload = json.dumps(items, ensure_ascii=False)[:16000]
    system_prompt = (
        "你是短视频收藏分析助手。用户收藏了一批短视频，系统已逐条分析并提取出结构化信息。"
        "请根据提供的该类别全部信息，写一份简洁的中文总结（Markdown 格式，300～500 字），包含："
        "1) 整体概况（共几个视频、内容特点）；2) 关键内容清单（如歌名/歌手、菜名/食材、文案主题）；"
        "3) 用户偏好倾向分析；4) 实用建议（如何整理成歌单/菜谱/摘抄本）。"
        "只依据输入信息总结，禁止编造；用中文输出。"
    )
    user_prompt = f"类别：{category}\n该类视频的结构化提取信息（JSON）：\n{items_payload}"
    if backend == BACKEND_HPC_QWEN:
        base_url = os.environ.get("HPC_QWEN_URL") or DEFAULT_HPC_QWEN_URL
        token = os.environ.get("HPC_QWEN_TOKEN") or ""
        if not token:
            raise RuntimeError("HPC Qwen 后端需要 Bearer 令牌（.env 的 HPC_QWEN_TOKEN）")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        text = _post_hpc_chat(
            base_url, headers,
            {HPC_QWEN_MESSAGE_KEY: system_prompt + "\n\n" + user_prompt},
            timeout=180,
        )
        return text.strip() or "（模型返回空内容）"
    if backend != BACKEND_OPENAI:
        raise ValueError(f"未知 LLM 后端: {backend!r}")
    api_key = os.environ.get("LLM_API_KEY") or ""
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY（.env 未配置）")
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content.strip() or "（模型返回空内容）"


def chat_with_context(category: str, messages: list[dict], backend: str = "") -> str:
    """带上下文的 AI 问答：把选定类别的分析资料注入 system 提示，再带入对话历史"""
    from second_layer.llm_analyzer import (
        BACKEND_HPC_QWEN,
        BACKEND_OPENAI,
        DEFAULT_BASE_URL,
        DEFAULT_HPC_QWEN_URL,
        DEFAULT_MODEL,
        HPC_QWEN_MESSAGE_KEY,
        _post_hpc_chat,
        load_dotenv,
    )

    load_dotenv()
    backend = backend or os.environ.get("LLM_BACKEND") or BACKEND_OPENAI
    agg = aggregate_analysis()
    if category in CATEGORY_FOLDERS:
        items = agg.get(category) or []
        ctx_json = json.dumps(items, ensure_ascii=False)[:16000]
        ctx_desc = f"用户收藏的「{category}」类视频分析资料"
    else:
        ctx_json = json.dumps(agg, ensure_ascii=False)[:16000]
        ctx_desc = "用户收藏的全部视频分析资料（按类别）"
    system_prompt = (
        "你是用户个人短视频收藏库的 AI 助手。用户收藏了一批短视频，系统已逐条分析并提取结构化信息。"
        f"以下是你可以查阅的资料：{ctx_desc}（JSON）：\n{ctx_json}\n\n"
        "回答规则：用中文简洁回答；优先基于资料内容；资料里没有的信息可基于常识合理补充，"
        "但要说明是推测；不要编造资料里不存在的具体事实。"
    )
    recent = [m for m in messages[-20:] if m.get("content")]
    if not recent:
        return "请先输入你想问的问题。"
    if backend == BACKEND_HPC_QWEN:
        base_url = os.environ.get("HPC_QWEN_URL") or DEFAULT_HPC_QWEN_URL
        token = os.environ.get("HPC_QWEN_TOKEN") or ""
        if not token:
            raise RuntimeError("HPC Qwen 后端需要 Bearer 令牌（.env 的 HPC_QWEN_TOKEN）")
        history = "\n".join(
            ("用户：" if m.get("role") == "user" else "助手：") + str(m.get("content"))
            for m in recent
        )
        full = system_prompt + "\n\n【对话历史】\n" + history + "\n\n请作为助手继续回答。"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        text = _post_hpc_chat(base_url, headers, {HPC_QWEN_MESSAGE_KEY: full}, timeout=180)
        return text.strip() or "（模型返回空内容）"
    if backend != BACKEND_OPENAI:
        raise ValueError(f"未知 LLM 后端: {backend!r}")
    api_key = os.environ.get("LLM_API_KEY") or ""
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY（.env 未配置）")
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in recent:
        role = "assistant" if m.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": str(m.get("content"))})
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": msgs, "temperature": 0.6},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content.strip() or "（模型返回空内容）"


def parse_multipart_stream(
    rfile: Any,
    content_type: str,
    length: int,
    tmp_dir: Path,
) -> dict[str, tuple[str, Any]]:
    """流式解析 multipart/form-data：普通字段存内存，文件部分写临时文件（大视频不占内存）。
    旧版把整个 body 读进内存再 split() 会造成 2 倍内存占用，大视频并发上传会 MemoryError。"""
    result: dict[str, tuple[str, Any]] = {}
    if "boundary=" not in content_type:
        return result
    boundary = content_type.split("boundary=")[-1].strip().strip('"').encode()
    delim = b"--" + boundary
    CHUNK = 1 << 20
    keep = len(delim) - 1
    remaining = length
    buf = bytearray()

    def fill() -> None:
        nonlocal remaining
        while remaining > 0 and len(buf) < CHUNK:
            chunk = rfile.read(min(CHUNK, remaining))
            if not chunk:
                break
            buf.extend(chunk)
            remaining -= len(chunk)

    def next_part_head() -> bytes | None:
        """返回下一个 part 的 headers 原始字节；结束返回 None。
        调用前提：上一 part 已把缓冲区推进到 '\r\n'+下一part 或 '--'+结束符"""
        fill()
        if len(buf) >= 2 and buf[:2] == b"--":
            return None
        if len(buf) >= 2 and buf[:2] == b"\r\n":
            del buf[:2]
        if not buf and remaining <= 0:
            return None
        while True:
            idx = buf.find(b"\r\n\r\n")
            if idx != -1:
                head = bytes(buf[:idx])
                del buf[:idx + 4]
                return head
            if remaining <= 0:
                return None
            fill()

    # 跳过 preamble，定位并消费第一个 boundary
    while True:
        idx = buf.find(delim)
        if idx != -1:
            del buf[:idx + len(delim)]
            break
        if remaining <= 0:
            return result
        fill()
    fill()
    if len(buf) >= 2 and buf[:2] == b"\r\n":
        del buf[:2]

    while True:
        head = next_part_head()
        if head is None:
            break
        name = filename = ""
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for piece in line.split(";"):
                    piece = piece.strip()
                    if piece.lower().startswith("name="):
                        name = piece.split("=", 1)[1].strip('"')
                    elif piece.lower().startswith("filename="):
                        filename = piece.split("=", 1)[1].strip('"')
        if not name:
            continue
        if filename:
            # 文件部分：边读边写临时文件，内存只保留 1MB 缓冲区
            tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), suffix=".upload")
            with os.fdopen(fd, "wb") as out:
                while True:
                    idx = buf.find(delim)
                    if idx != -1:
                        out.write(buf[:idx])
                        del buf[:idx + len(delim)]
                        break
                    safe = len(buf) - keep
                    if safe > 0:
                        out.write(buf[:safe])
                        del buf[:safe]
                    fill()
                    if remaining <= 0 and len(buf) <= keep:
                        out.write(buf)
                        buf.clear()
                        break
            # 去掉内容末尾分隔符前的 \r\n
            try:
                with open(tmp_path, "r+b") as fh:
                    if fh.seek(0, 2) >= 2:
                        fh.seek(-2, 2)
                        if fh.read(2) == b"\r\n":
                            fh.seek(-2, 2)
                            fh.truncate()
            except OSError:
                pass
            result[name] = (filename, Path(tmp_path))
            fill()
            if len(buf) >= 2 and buf[:2] == b"\r\n":
                del buf[:2]
        else:
            parts: list[bytes] = []
            while True:
                idx = buf.find(delim)
                if idx != -1:
                    parts.append(bytes(buf[:idx]))
                    del buf[:idx + len(delim)]
                    break
                if remaining <= 0:
                    parts.append(bytes(buf))
                    buf.clear()
                    break
                fill()
            result[name] = ("", b"".join(parts).rstrip(b"\r\n"))
    return result


def parse_form_stream(rfile: Any, headers: dict[str, str], length: int,
                      tmp_dir: Path) -> dict[str, tuple[str, Any]]:
    """POST 请求体解析入口：multipart 走流式，普通表单直接读（都很小）"""
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" in content_type:
        return parse_multipart_stream(rfile, content_type, length, tmp_dir)
    body = rfile.read(length)
    return {
        k: ("", v[0].encode("utf-8"))
        for k, v in parse_qs(body.decode("utf-8", "replace")).items()
    }


def parse_form(headers: dict[str, str], body: bytes) -> dict[str, tuple[str, bytes]]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" in content_type:
        return parse_multipart(content_type, body)
    return {
        k: ("", v[0].encode("utf-8"))
        for k, v in parse_qs(body.decode("utf-8", "replace")).items()
    }


def _build_config(fields: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    """从表单字段构建分析配置（上传 /api/upload 与重新识别 /api/reanalyze 共用）"""

    def _bool(v: tuple[str, Any]) -> bool:
        return v[1].decode("utf-8", "replace").strip().lower() not in ("0", "false", "off")

    def _text(key: str, default: str) -> str:
        return fields.get(key, ("", default.encode()))[1].decode("utf-8", "replace").strip() or default

    def _num(key: str, default: float) -> float:
        try:
            return float(_text(key, str(default)))
        except ValueError:
            return default

    # 通用开关：未勾选的模型进 disable 列表（兼容旧的 asr/ocr/... 0/1 字段）
    disabled: set[str] = set()
    for x in _text("disable", "").split(","):
        if x.strip():
            disabled.add(x.strip())
    for sid in ("asr", "ocr", "visual", "audio"):
        if sid in fields and not _bool(fields[sid]):
            disabled.add(sid)
    # 各模型后端选择（backend_<id>），未知 id 原样记录供以后扩展
    backends: dict[str, str] = {}
    for k, v in fields.items():
        if k.startswith("backend_") and v[1]:
            backends[k[len("backend_"):]] = v[1].decode("utf-8", "replace").strip()

    return {
        "disable": sorted(disabled),
        "visual_backend": backends.get("visual", _text("visual_backend", "clip")),
        "audio_backend": backends.get("audio", "fingerprint"),
        "llm_backend": _text("llm_backend", "openai"),
        "backends": backends,
        "keyframe_interval": _num("keyframe_interval", 2.0),
        "temperature": _num("temperature", 0.0),
        "max_retries": int(_num("max_retries", 2)),
        "asr_model": _text("asr_model", "medium"),
        "ocr_lang": _text("ocr_lang", "ch"),
        "max_duration_minutes": _num("max_duration_minutes", 10.0),
        "llm_model": _text("llm_model", ""),
        "model_timeout": _num("model_timeout", 300.0),
        "asr_beam": int(_num("asr_beam", 5)),
        "asr_threads": int(_num("asr_threads", 4)),
        "vlm_max_frames": int(_num("vlm_max_frames", 12)),
        "ocr_max_side": int(_num("ocr_max_side", 0)),
        "ocr_max_frames": int(_num("ocr_max_frames", 0)),
        "concurrent": int(_num("concurrent", get_concurrency())),
    }


def _queue_summary() -> dict[str, Any]:
    """排队/运行中的任务列表 + 估算剩余时间（供前端队列面板）"""
    with JOBS_LOCK:
        jobs = [j for j in JOBS.values() if j.get("status") in ("queued", "running")]
        jobs = sorted(jobs, key=lambda j: (0 if j.get("status") == "running" else 1, j.get("id", "")))
    out: list[dict[str, Any]] = []
    for j in jobs:
        cfg = j.get("config") or {}
        dur = float(j.get("duration") or 0)
        est = _estimate_seconds(dur, cfg)
        out.append({
            "id": j.get("id"),
            "name": j.get("name"),
            "status": j.get("status"),
            "duration": round(dur, 1),
            "estimate": round(est, 1),
            "visual_backend": cfg.get("visual_backend") or "",
            "asr_model": cfg.get("asr_model") or "",
        })
    total = sum(x["estimate"] for x in out)
    return {"jobs": out, "count": len(out), "total_estimate": round(total, 1)}


# ---------------------------------------------------------------------------
# HTTP Handler（纯 JSON API + 静态前端）
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: dict[str, str] | None = None) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass  # 客户端中途断开（刷新页面等）属正常，忽略

    def _serve_file_with_range(self, path: Path) -> None:
        """带 Range 支持的文件服务（视频拖动进度条需要）"""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng.split("=", 1)[1]
            try:
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())

    def _send_json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _query(self) -> dict[str, str]:
        q = parse_qs(urlparse(self.path).query)
        return {k: v[0] for k, v in q.items()}

    def _serve_frontend(self) -> None:
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(200, (
                "<h1>前端文件缺失</h1><p>请确认 frontend/index.html 存在。</p>"
            ).encode(), "text/html; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        q = self._query()
        if path in ("/", "/index.html"):
            self._serve_frontend()
        elif path == "/api/results":
            self._send_json(list_results())
        elif path == "/api/models":
            self._send_json(describe_registry())
        elif path == "/api/sysinfo":
            self._send_json(get_sysinfo())
        elif path == "/api/status":
            job = JOBS.get(q.get("id", ""))
            if job is None:
                self._send_json({"error": "任务不存在"}, 404)
                return
            resp = {
                "status": job["status"],
                "name": job["name"],
                "config": job.get("config", {}),
                "log": list(job["log"]),
            }
            if job["status"] == "queued":
                # 排队位置：前面还有多少个排队任务（供前端显示）
                resp["queue_ahead"] = sum(
                    1 for j in JOBS.values()
                    if j.get("status") == "queued" and j["id"] < job["id"]
                )
            self._send_json(resp)
        elif path == "/api/video":
            video_file = _find_video(q.get("name", ""))
            if video_file is None:
                self._send_json({"error": "视频文件不存在"}, 404)
                return
            # HEVC 等编码转码为 H.264，保证 Chrome/Edge 都能播
            playable = _web_playable(video_file) or video_file
            self._serve_file_with_range(playable)
        elif path == "/api/result":
            data = get_analysis(q.get("name", ""))
            if data is None:
                self._send_json({"error": "分析结果不存在"}, 404)
                return
            self._send_json(data)
        elif path == "/api/export":
            name = q.get("name", "")
            fmt = q.get("format", "md")
            if fmt == "json":
                data = get_analysis(name)
                if data is None:
                    self._send_json({"error": "分析结果不存在"}, 404)
                    return
                body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8",
                           {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}_analysis.json"})
            else:
                md_path = None
                cand = OUTPUTS_DIR / f"{name}_evidence_analysis.md"
                if cand.exists():
                    md_path = cand
                else:
                    for q in sorted(OUTPUTS_DIR.rglob(f"{name}_evidence_analysis.md")):
                        if q.is_file():
                            md_path = q
                            break
                if md_path is None:
                    self._send_json({"error": "md 报告不存在，请先运行流水线"}, 404)
                    return
                self._send(200, md_path.read_bytes(), "text/markdown; charset=utf-8",
                           {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}_analysis.md"})
        elif path == "/api/aggregate":
            agg = aggregate_analysis()
            if q.get("format") == "md":
                body = build_aggregate_md(agg).encode("utf-8")
                self._send(200, body, "text/markdown; charset=utf-8",
                           {"Content-Disposition": "attachment; filename*=UTF-8''" + quote("信息提取合集.md")})
            else:
                self._send_json(agg)
        elif path == "/api/summary":
            cat = q.get("category", "")
            if cat not in CATEGORY_FOLDERS:
                self._send_json({"error": "类别无效"}, 400)
                return
            try:
                text = summarize_category(cat, q.get("backend", ""))
            except Exception as e:
                self._send_json({"error": f"总结失败: {e}"}, 500)
                return
            self._send_json({"category": cat, "summary": text})
        elif path == "/api/queue":
            self._send_json(_queue_summary())
        elif path == "/api/prompts":
            self._send_json(get_all_prompts())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES + 1024 * 1024:
            self._send_json({"error": "文件过大"}, 413)
            return

        if path == "/api/chat":
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "请求体不是合法 JSON"}, 400)
                return
            category = str(payload.get("category") or "")
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json({"error": "缺少对话内容"}, 400)
                return
            try:
                reply = chat_with_context(category, messages, str(payload.get("backend") or ""))
            except Exception as e:
                self._send_json({"error": f"AI 回答失败: {e}"}, 500)
                return
            self._send_json({"reply": reply})
            return

        if path == "/api/prompts":
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "请求体不是合法 JSON"}, 400)
                return
            if payload.get("action") == "reset":
                reset_prompts()
            else:
                prompts = payload.get("prompts") or payload
                save_prompts(prompts)
            self._send_json({"ok": True})
            return

        # 流式解析：大视频写临时文件、不占内存（修复批量上传时 MemoryError/缓冲区崩溃）
        fields = parse_form_stream(self.rfile, dict(self.headers), length, _UPLOAD_TMP_DIR)

        if path == "/api/upload":
            fname, content = fields.get("video", ("", b""))
            if not fname or content in (b"", ""):
                self._send_json({"error": "缺少视频文件"}, 400)
                return
            ext = Path(fname).suffix.lower()
            if ext not in ALLOWED_EXTS:
                if isinstance(content, Path):
                    content.unlink(missing_ok=True)
                self._send_json({"error": f"不支持的格式 {ext}"}, 400)
                return
            size = content.stat().st_size if isinstance(content, Path) else len(content)
            if size > MAX_UPLOAD_BYTES:
                if isinstance(content, Path):
                    content.unlink(missing_ok=True)
                self._send_json({"error": "文件过大（限 500MB）"}, 413)
                return

            config = _build_config(fields)
            # 网页可动态调整并行任务数（1~8）
            try:
                set_concurrency(int(config["concurrent"]))
            except Exception:
                pass
            VIDEOS_DIR.mkdir(exist_ok=True)
            dest = VIDEOS_DIR / fname
            if isinstance(content, Path):
                # 流式上传：临时文件直接移入 videos/
                shutil.move(str(content), str(dest))
            else:
                dest.write_bytes(content)
            name = Path(fname).stem
            job_id = new_job(name, config)
            start_job(job_id, dest)
            self._send_json({"job_id": job_id, "name": name, "config": config})

        elif path == "/api/reanalyze":
            # 重新识别：对已有视频用当前选择的模型/参数重新跑流水线（不重新上传文件）
            name = fields.get("name", ("", b""))[1].decode("utf-8", "replace").strip()
            if not name:
                self._send_json({"error": "缺少视频名称"}, 400)
                return
            video_file = _find_video(name)
            if video_file is None:
                self._send_json({"error": "视频文件不存在（可能已被删除）"}, 404)
                return
            config = _build_config(fields)
            try:
                set_concurrency(int(config["concurrent"]))
            except Exception:
                pass
            job_id = new_job(name, config)
            start_job(job_id, video_file)
            self._send_json({"job_id": job_id, "name": name, "config": config})

        elif path == "/api/correct":
            name = fields.get("name", ("", b""))[1].decode("utf-8", "replace")
            category = fields.get("category", ("", b""))[1].decode("utf-8", "replace")
            note = fields.get("note", ("", b""))[1].decode("utf-8", "replace")
            data = get_analysis(name)
            if data is None:
                self._send_json({"error": "分析结果不存在"}, 404)
                return
            data["human_correction"] = {
                "category": category,
                "note": note,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            p = _find_analysis_path(name)
            if p is None:
                self._send_json({"error": "分析结果不存在"}, 404)
                return
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._send_json({"ok": True, "name": name, "category": category})
        elif path == "/api/delete_record":
            name = fields.get("name", ("", b""))[1].decode("utf-8", "replace")
            if not name:
                self._send_json({"error": "缺少 name"}, 400)
                return
            r = _delete_record(name)
            self._send_json({"ok": True, "name": name, "files": r["files"], "videos": r["videos"]})
        elif path == "/api/clear_history":
            # 清除全部历史记录：删除 outputs 下所有分析产物与全部视频文件
            deleted = videos_deleted = 0
            try:
                for p in OUTPUTS_DIR.rglob("*"):
                    if p.is_file():
                        try:
                            p.unlink()
                            deleted += 1
                        except OSError:
                            pass
                for p in VIDEOS_DIR.rglob("*"):
                    if p.is_file():
                        try:
                            p.unlink()
                            videos_deleted += 1
                        except OSError:
                            pass
            except Exception:
                pass
            with JOBS_LOCK:
                JOBS.clear()
            try:
                if JOBS_FILE.exists():
                    JOBS_FILE.unlink()
            except OSError:
                pass
            self._send_json({"ok": True, "deleted": deleted, "videos": videos_deleted})
        elif path == "/api/cancel_job":
            # 取消排队中的任务（id=all 清空全部排队）
            jid = fields.get("id", ("", b""))[1].decode("utf-8", "replace").strip()
            if not jid:
                self._send_json({"error": "缺少任务 id"}, 400)
                return
            removed = 0
            with JOBS_LOCK:
                if jid == "all":
                    for k in [k for k, j in JOBS.items() if j.get("status") == "queued"]:
                        JOBS.pop(k, None)
                        removed += 1
                else:
                    job = JOBS.get(jid)
                    if job is None:
                        self._send_json({"error": "任务不存在"}, 404)
                        return
                    if job.get("status") != "queued":
                        self._send_json({"error": "只能取消排队中的任务（分析中请等待完成）"}, 400)
                        return
                    JOBS.pop(jid, None)
                    removed = 1
            _persist_jobs()
            self._send_json({"ok": True, "removed": removed})
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, *args: Any) -> None:
        pass  # 安静模式

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # 客户端断连等无害错误不再刷屏


def main() -> None:
    parser = argparse.ArgumentParser(description="智能短视频分析后端 API + 前端托管")
    parser.add_argument("--port", type=int, default=8800,
                        help="端口（默认 8800，避开 HPC 隧道的 8000）")
    parser.add_argument("--jobs", type=int, default=2,
                        help="默认并行分析任务数（1~8，可在网页高级参数中随时调整）")
    args = parser.parse_args()

    set_concurrency(args.jobs)

    VIDEOS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 恢复上次未完成的任务（服务重启后自动继续分析）
    _load_jobs()

    # 启动工作线程池（多出的线程等空位，实际并行数由并发上限控制）
    for i in range(MAX_SLOT_WORKERS):
        threading.Thread(target=_job_worker, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"后端已启动: http://127.0.0.1:{args.port}")
    print(f"前端页面: http://127.0.0.1:{args.port}/")
    print(f"默认并行任务数: {get_concurrency()}（网页高级参数可调 1~8）")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
