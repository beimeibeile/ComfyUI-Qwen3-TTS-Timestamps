# coding=utf-8
"""
Qwen3-TTS-Timestamps 服务端
============================
Qwen3-TTS 合成（CustomVoice / VoiceClone / VoiceDesign）+ Qwen3-ForcedAligner 词级强制对齐 -> 同源字级时间戳 -> SRT。

设计要点：
- 模型懒加载：三种 Qwen3-TTS 模型按需加载缓存（切换类型时自动卸载旧的），ForcedAligner 常驻。
- 音色注册：克隆音色（ref_audio + ref_text）注册时预计算 voice-clone prompt 并落盘，合成零重复计算。
- 对齐：transcript 与合成文本完全一致（强制对齐要求同款文本），中文按字、英文按词输出时间戳。
- 采样：seed 固定可复现；max_new_tokens/top_p/temperature/repetition_penalty 可调。
  Qwen3 无 speed/pause 概念（语速/停顿通过 instruct 控制，如"放慢语速""停顿两秒"）。
"""
import argparse
import base64
import io
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ROOT = r"D:\Ai\ComfyUI-aki-v3.2\ComfyUI\models"
QWEN3_TTS_DIR = os.path.join(DEFAULT_MODEL_ROOT, "Qwen3-TTS")
QWEN3_ASR_DIR = os.path.join(DEFAULT_MODEL_ROOT, "Qwen3-ASR")

# 三类 Qwen3-TTS 模型（本地目录名）；voice_clone 与 base 共用 Base 模型
QWEN3_MODEL_NAMES: Dict[str, str] = {
    "custom_voice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen3-TTS-12Hz-1.7B-Base",
    "base": "Qwen3-TTS-12Hz-1.7B-Base",
}
# 内部缓存键归一: voice_clone 与 base 共用同一个模型实例
_NORMAL_MODEL_TYPE = {"voice_clone": "base"}
# 对齐模型（与 ComfyUI-Qwen3-ASR 同一目录约定，可共享）
ALIGNER_NAME = "Qwen3-ForcedAligner-0.6B"

# 9 个预置音色（来自 CustomVoice 模型 get_supported_speakers()，按字母序）
PRESET_SPEAKERS: List[str] = [
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
]

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_FILE = os.path.join(SERVER_DIR, "voices.json")
VOICES_DIR = os.path.join(SERVER_DIR, "voices")

# ---------------------------------------------------------------------------
# sox 缺失补丁
# ---------------------------------------------------------------------------
# qwen_tts 的 25Hz tokenizer(XVectorExtractor) 用外部 sox 二进制做音频峰值归一化(-6 dBFS)。
# 本机未安装 sox 时该路径报错(SoX could not be found), 这里用等效 numpy 实现替换:
# sox `norm` 默认语义 = 峰值归一到目标 dBFS(无 RMS 标志), 与本实现一致。
def _patch_sox_norm():
    import shutil
    if shutil.which("sox"):
        return  # 系统已有 sox, 保持库原逻辑
    try:
        from qwen_tts.core.tokenizer_25hz.vq.speech_vq import XVectorExtractor

        def sox_norm_np(self, audio):
            a = np.asarray(audio, dtype=np.float32)
            peak = float(np.max(np.abs(a))) if a.size else 0.0
            if peak <= 0 or not np.isfinite(peak):
                return a.copy()
            target = 10.0 ** (-6.0 / 20.0)  # -6 dBFS
            return (a * (target / peak)).astype(np.float32)

        XVectorExtractor.sox_norm = sox_norm_np
        print("[engine] 本机无 sox, 已应用等效峰值归一补丁(-6 dBFS)", flush=True)
    except Exception as e:
        print(f"[engine] sox 补丁未生效(原逻辑保留): {e}", flush=True)


_patch_sox_norm()

# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class SynthRequest(BaseModel):
    text: str
    voice_type: str = "custom_voice"  # custom_voice | voice_clone | voice_design
    voice: str = ""                   # 预置音色名 / 注册音色名 / 空（voice_design 用）
    instruct: str = ""                # 方言/情绪指令（custom_voice / voice_design）
    ref_audio: Optional[str] = None   # 直接克隆用：base64 wav 或本地路径
    ref_text: Optional[str] = None    # 参考音频转写（voice_clone ICL 必需）
    language: str = "Auto"
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    device: str = ""                  # 空=服务端启动参数，cuda/cpu 需与启动一致


class RegisterRequest(BaseModel):
    name: str
    ref_audio: str                     # base64 wav
    ref_text: str = ""                 # 参考音频转写（克隆必需，建议与音频一致）
    device: str = ""


class DeleteRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# 引擎管理器（懒加载 + idle 卸载）
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, model_root: str, device: str, dtype: str, idle_timeout: float):
        self.model_root = model_root
        self.device = device
        self.dtype = dtype
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.current_type: Optional[str] = None   # custom_voice / voice_design / base
        self.tts: Any = None
        self.aligner: Any = None
        self.last_used = time.time()
        self._idle_thread: Optional[threading.Thread] = None

    # -- 加载 --
    def _load_tts(self, model_type: str):
        from qwen_tts import Qwen3TTSModel
        path = os.path.join(self.model_root, "Qwen3-TTS", QWEN3_MODEL_NAMES[model_type])
        if not (os.path.isdir(path) and os.listdir(path)):
            raise HTTPException(500, f"模型不存在或为空: {path}（请先下载 Qwen3-TTS 模型）")
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        dev = self.device
        print(f"[engine] 加载 {model_type} <- {path} ({self.device}/{self.dtype})", flush=True)
        t0 = time.time()
        m = Qwen3TTSModel.from_pretrained(
            path,
            device_map=dev,
            dtype=dt,
            attn_implementation="sdpa",
        )
        print(f"[engine] {model_type} 加载完成 {time.time()-t0:.1f}s", flush=True)
        return m

    def _load_aligner(self):
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
        path = os.path.join(self.model_root, "Qwen3-ASR", ALIGNER_NAME)
        if not (os.path.isdir(path) and os.listdir(path)):
            raise HTTPException(500, f"ForcedAligner 模型不存在: {path}（请先手动放置该模型）")
        print(f"[engine] 加载 ForcedAligner <- {path}", flush=True)
        t0 = time.time()
        a = Qwen3ForcedAligner.from_pretrained(
            path,
            device_map=self.device,
            dtype=torch.bfloat16 if self.dtype == "bf16" else torch.float16 if self.dtype == "fp16" else torch.float32,
        )
        print(f"[engine] ForcedAligner 加载完成 {time.time()-t0:.1f}s", flush=True)
        return a

    def get_tts(self, model_type: str):
        model_type = _NORMAL_MODEL_TYPE.get(model_type, model_type)  # voice_clone -> base 共享缓存
        with self.lock:
            if self.current_type != model_type:
                self.tts = None
                torch.cuda.empty_cache() if self.device.startswith("cuda") else None
                self.tts = self._load_tts(model_type)
                self.current_type = model_type
            self.last_used = time.time()
            return self.tts

    def get_aligner(self):
        with self.lock:
            if self.aligner is None:
                self.aligner = self._load_aligner()
            self.last_used = time.time()
            return self.aligner

    def unload(self):
        with self.lock:
            self.tts = None
            self.aligner = None
            self.current_type = None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        print("[engine] 模型已卸载（idle 超时）", flush=True)

    def start_idle_monitor(self):
        if self.idle_timeout <= 0:
            return
        def loop():
            while True:
                time.sleep(5)
                if self.tts is not None or self.aligner is not None:
                    if time.time() - self.last_used > self.idle_timeout:
                        self.unload()
        self._idle_thread = threading.Thread(target=loop, daemon=True)
        self._idle_thread.start()


engine: Optional[Engine] = None


# ---------------------------------------------------------------------------
# 音色注册表
# ---------------------------------------------------------------------------
def _load_voices() -> Dict[str, Any]:
    if os.path.exists(VOICES_FILE):
        try:
            with open(VOICES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_voices(voices: Dict[str, Any]):
    os.makedirs(SERVER_DIR, exist_ok=True)
    with open(VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 音频/工具函数
# ---------------------------------------------------------------------------
def _b64_to_wav_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _wav_bytes_to_np(data: bytes) -> tuple:
    import soundfile as sf
    with io.BytesIO(data) as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32), int(sr)


def _np_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _align_language(language: str) -> str:
    """ForcedAligner 语言名（与 qwen 语言名基本一致；方言文本仍按主语言对齐）"""
    l = (language or "Auto").strip()
    if l.lower() in ("auto", "中文", "chinese", "zh", "mandarin"):
        return "Chinese"
    if l.lower() in ("english", "英文", "en"):
        return "English"
    return language or "Chinese"


def build_srt(text: str, words: List[Dict[str, Any]], fallback_chars: int = 16) -> str:
    """词级时间戳 -> SRT。

    断句策略:
    - ForcedAligner 的 items 不含标点(中文按字、英文按词), 因此用原始 text 的
      标点位置作锚点, 对 words 流做【跨越检测】切句(对齐项偶尔合并/偏移也不丢句);
    - 无标点或跨度过大时按 fallback_chars 字兜底切句。
    """
    if not words:
        return ""

    # 原文断句锚点: 内容字符流中每个断句标点后的累计字符数(排除空白与标点自身)
    anchors: List[int] = []
    content = 0
    for ch in text or "":
        if ch in "，。！？；、：,.;:!?…\n\r ":
            if content > 0:
                anchors.append(content)
        else:
            content += 1
    anchors = sorted(set(a for a in anchors if a < content))

    sentences: List[Dict[str, Any]] = []
    cur_chars: List[str] = []
    cur_start: float = words[0]["start_time"]
    cur_end: float = words[0]["end_time"]

    def flush():
        nonlocal cur_chars, cur_start, cur_end
        if cur_chars:
            seg = "".join(cur_chars).strip()
            if seg:
                sentences.append({"text": seg, "start": cur_start, "end": cur_end})
        cur_chars, cur_start, cur_end = [], 0.0, 0.0

    consumed = 0
    prev_consumed = 0
    for w in words:
        t = w.get("text") or ""
        if not cur_chars:
            cur_start = w["start_time"]
        cur_chars.append(t)
        cur_end = w["end_time"]
        consumed += len(t)
        crossed = any(prev_consumed < a <= consumed for a in anchors)
        if crossed or len("".join(cur_chars)) >= fallback_chars:
            flush()
        prev_consumed = consumed

    flush()

    lines = []
    for i, s in enumerate(sentences, 1):
        lines.append(f"{i}")
        lines.append(f"{_srt_time(s['start'])} --> {_srt_time(s['end'])}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS-Timestamps", version="1.0.0")

# 浏览器前端(ComfyUI 页面 8188)跨域访问 /spks 等只读接口必须放行 CORS,
# 否则主节点 voice 下拉的前端 fetch 会被浏览器拦截(节点后端执行不受影响)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(_app):
    global engine
    if engine is None:
        engine = Engine(DEFAULT_MODEL_ROOT, "cuda" if torch.cuda.is_available() else "cpu", "bf16", 0)
    engine.start_idle_monitor()
    yield


app.router.lifespan_context = lifespan


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": engine.device if engine else "?",
        "model_type": engine.current_type if engine else None,
        "voices": len(_load_voices()),
        "presets": PRESET_SPEAKERS,
    }


@app.get("/spks")
def list_spks():
    voices = _load_voices()
    registered = [n for n in voices]  # 字符串数组, 与 /register 响应一致(前端下拉直接使用)
    # 预置音色：优先缓存；custom_voice 已加载时动态补充
    presets = PRESET_SPEAKERS
    if not presets and engine and engine.current_type == "custom_voice":
        spks = engine.tts.get_supported_speakers()
        if spks:
            PRESET_SPEAKERS.extend(spks)
            presets = PRESET_SPEAKERS
    return {"voices": registered, "presets": presets}


@app.post("/register")
def register_voice(req: RegisterRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "音色名不能为空")
    if not req.ref_audio:
        raise HTTPException(400, "缺少参考音频 ref_audio")
    try:
        wav_bytes = _b64_to_wav_bytes(req.ref_audio)
        ref_audio, ref_sr = _wav_bytes_to_np(wav_bytes)
    except Exception as e:
        raise HTTPException(400, f"参考音频解析失败: {e}")
    if len(ref_audio) / ref_sr < 1.0:
        raise HTTPException(400, "参考音频过短（至少 1 秒）")
    ref_text = req.ref_text.strip()
    if not ref_text:
        raise HTTPException(400, "克隆音色需要 ref_text（参考音频的转写文本），否则 ICL 对齐会失败")

    os.makedirs(VOICES_DIR, exist_ok=True)
    wav_path = os.path.join(VOICES_DIR, f"{name}.wav")
    with open(wav_path, "wb") as f:
        f.write(_np_to_wav_bytes(ref_audio, ref_sr))

    # 预计算 voice-clone prompt（省去每次合成重算）
    tts = engine.get_tts("base")
    items = tts.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    prompt_path = os.path.join(VOICES_DIR, f"{name}.pt")
    torch.save(items, prompt_path)  # 保存完整列表(与 generate_voice_clone 期望格式一致)

    voices = _load_voices()
    voices[name] = {
        "mode": "voice_clone",
        "ref_audio": wav_path,
        "ref_text": ref_text,
        "prompt_path": prompt_path,
    }
    _save_voices(voices)
    return {"ok": True, "name": name, "voices": [n for n in voices]}


@app.post("/delete")
def delete_voice(req: DeleteRequest):
    voices = _load_voices()
    if req.name not in voices:
        raise HTTPException(404, f"音色不存在: {req.name}")
    for suf in (".wav", ".pt"):
        p = os.path.join(VOICES_DIR, req.name + suf)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    del voices[req.name]
    _save_voices(voices)
    return {"ok": True, "name": req.name}


@app.post("/synth")
def synth(req: SynthRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "合成文本不能为空")
    if req.voice_type not in QWEN3_MODEL_NAMES:
        raise HTTPException(400, f"voice_type 无效: {req.voice_type}")

    tts = engine.get_tts(req.voice_type)
    aligner = engine.get_aligner()

    gen_kwargs: Dict[str, Any] = {}
    if req.seed is not None:
        gen_kwargs["seed"] = req.seed
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty

    # qwen_tts 库(0.1.1)未实现 seed: generate 签名无 seed 参数, 透传会被忽略 → 每次合成真随机,
    # 方言/风格"时有时无"的直接根因。这里在调用前主动设置 torch RNG, 实现同 seed 同参数完全复现。
    if req.seed is not None:
        _seed = int(req.seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)

    language = req.language or "Auto"
    try:
        t0 = time.time()
        if req.voice_type == "custom_voice":
            if not req.voice:
                raise HTTPException(400, "custom_voice 模式需要选择预置音色 voice")
            wavs, sr = tts.generate_custom_voice(
                text=text, speaker=req.voice, language=language,
                instruct=req.instruct or None, non_streaming_mode=True, **gen_kwargs,
            )
        elif req.voice_type == "voice_design":
            wavs, sr = tts.generate_voice_design(
                text=text, instruct=req.instruct or "", language=language,
                non_streaming_mode=True, **gen_kwargs,
            )
        else:  # voice_clone
            if req.ref_audio:
                # 直接克隆：本次请求临时提供参考音频
                wav_bytes = _b64_to_wav_bytes(req.ref_audio)
                ref_audio, ref_sr = _wav_bytes_to_np(wav_bytes)
                ref_text = (req.ref_text or "").strip()
                if not ref_text:
                    raise HTTPException(400, "voice_clone 模式需要 ref_text（参考音频转写）")
                wavs, sr = tts.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=(ref_audio, ref_sr), ref_text=ref_text,
                    x_vector_only_mode=False, non_streaming_mode=True, **gen_kwargs,
                )
            else:
                voices = _load_voices()
                if not req.voice or req.voice not in voices:
                    raise HTTPException(400, f"voice_clone 模式需要注册音色（已注册: {list(voices)}）")
                info = voices[req.voice]
                prompt_path = info.get("prompt_path")
                if not prompt_path or not os.path.exists(prompt_path):
                    raise HTTPException(500, f"音色 prompt 缺失: {req.voice}，请重新注册")
                prompt_item = torch.load(prompt_path, map_location=tts.device, weights_only=False)
                if not isinstance(prompt_item, list):  # 兼容旧单对象格式
                    prompt_item = [prompt_item]
                wavs, sr = tts.generate_voice_clone(
                    text=text, language=language,
                    voice_clone_prompt=prompt_item,
                    non_streaming_mode=True, **gen_kwargs,
                )
        print(f"[synth] {req.voice_type} 合成完成 {time.time()-t0:.1f}s", flush=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"合成失败: {e}")

    audio = wavs[0] if isinstance(wavs, list) else wavs
    audio = np.asarray(audio, dtype=np.float32)

    # 对齐（transcript 必须与合成文本完全一致）
    try:
        t0 = time.time()
        align_lang = _align_language(language)
        result = aligner.align(audio=(audio, int(sr)), text=text, language=align_lang)[0]
        words = [
            {"text": it.text, "start_time": it.start_time, "end_time": it.end_time}
            for it in result
        ]
        print(f"[synth] 对齐完成 {time.time()-t0:.1f}s，{len(words)} tokens", flush=True)
    except Exception as e:
        raise HTTPException(500, f"强制对齐失败: {e}（文本与音频不一致时可能失败）")

    srt = build_srt(text, words)
    duration = float(len(audio) / sr) if sr else 0.0
    wav_b64 = base64.b64encode(_np_to_wav_bytes(audio, int(sr))).decode("ascii")

    return {
        "audio_base64": wav_b64,
        "sample_rate": int(sr),
        "word_timestamps": words,
        "srt": srt,
        "duration_seconds": round(duration, 3),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global engine
    parser = argparse.ArgumentParser(description="Qwen3-TTS-Timestamps 服务端")
    parser.add_argument("--model_root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--idle_timeout", type=float, default=0, help="秒，0=不自动卸载")
    args = parser.parse_args()

    engine = Engine(args.model_root, args.device, args.dtype, args.idle_timeout)
    engine.start_idle_monitor()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
