# coding=utf-8
"""
ComfyUI-Qwen3-TTS-Timestamps 节点
=================================
统一注册规则: DBW(系列前缀) - 工程包(Qwen3TTS) - 节点
  节点1: DBW-Qwen3TTS-TTSWithTimestamps   Qwen3-TTS 合成 + ForcedAligner 词级对齐 → 同源字级时间戳
  节点2: DBW-Qwen3TTS-RegisterVoice       克隆音色注册(参考音频 + ref_text 转写)
  节点3: DBW-Qwen3TTS-DeleteVoice         删除注册音色
  节点4: DBW-Qwen3TTS-SRTFileWriter       SRT 落盘(占位符支持, 仅连线输入)

时间戳事实: qwen_tts 各生成接口仅返回 (wavs, sample_rate) 无时间戳;
  词级时间戳 = 合成完成后用 Qwen3-ForcedAligner 对成品波形强制对齐(transcript 与合成文本完全一致)。
"""
import base64
import io
import json
import os
import re
from datetime import datetime

import numpy as np
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_CATEGORY = "DBW/Qwen3TTS"
_DEFAULT_SERVER = "http://127.0.0.1:8001"

# Qwen3-TTS 支持语言(与模型一致)
_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]
# 生成模式
_VOICE_TYPES = ["custom_voice", "voice_clone", "voice_design"]

# 9 个预置音色(与服务端 PRESET_SPEAKERS 一致)
_PRESET_VOICES = [
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
]


def _voice_options():
    """voice 下拉允许值: 空串(voice_design 清空/未选) + 9 预置 + 已注册克隆音色(server/voices.json)。

    每次 INPUT_TYPES 调用时生成 → 节点定义列表始终为当前有效值,
    避免 ComfyUI 前端 combo 校验把列表外的值判为"无效输入"
    (后端校验注释明确 combo 由前端处理; 前端校验读取节点定义列表)。
    """
    opts = [""] + list(_PRESET_VOICES)
    try:
        vf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "voices.json")
        if os.path.exists(vf):
            with open(vf, "r", encoding="utf-8") as f:
                for name in json.load(f).keys():
                    if name and name not in opts:
                        opts.append(name)
    except Exception:
        pass
    return opts


# ---------------------------------------------------------------------------
# HTTP / 工具
# ---------------------------------------------------------------------------
def _post_json(server_url, payload, path="/synth", timeout=600):
    import requests
    url = (server_url or _DEFAULT_SERVER).rstrip("/") + path
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fetch_voices(server_url):
    """拉取音色列表: {voices:[注册克隆音色], presets:[预置音色]}"""
    import requests
    url = (server_url or _DEFAULT_SERVER).rstrip("/") + "/spks"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"voices": [], "presets": []}


def _decode_audio_base64(b64: str) -> torch.Tensor:
    """服务端返回标准 WAV bytes -> ComfyUI AUDIO waveform [1,1,T]

    必须按 WAV 解析(soundfile): 服务端写入 PCM_16(2 字节/样本)+44 字节头,
    若按裸 float32 流读会把头部当波形、2 字节样本错拼 4 字节 → 时长减半(加速)+波形畸变。
    """
    import io
    import soundfile as sf
    raw = base64.b64decode(b64)
    audio, _ = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 多声道取平均为单声道
    return torch.from_numpy(audio).reshape(1, 1, -1).float()


# ---------------------------------------------------------------------------
# 占位符解析(日期 + 跨节点引用) — 全新实现
# ---------------------------------------------------------------------------
_DATE_PLACEHOLDER_RE = re.compile(r"%date:([^%]+)%")
_NODE_REF_RE = re.compile(r"%([^%.]+)\.([^%]+)%")


def _build_title_map(extra_pnginfo) -> dict:
    if not extra_pnginfo:
        return {}
    work = extra_pnginfo.get("workflow", {})
    title_map = {}
    for node in work.get("nodes", []):
        title = node.get("title", "")
        title_map[node.get("id")] = title
    return title_map


def _lookup_node_value(node_name, field, prompt, title_map):
    node_id = None
    for nid, meta in (prompt or {}).items():
        if meta.get("_meta", {}).get("title") == node_name or title_map.get(int(nid) if str(nid).isdigit() else None) == node_name:
            node_id = nid
            break
    if node_id is None:
        return ""
    inputs = (prompt or {}).get(node_id, {}).get("inputs", {})
    val = inputs.get(field, "")
    if isinstance(val, dict):
        val = val.get("value", "")
    return str(val)


def _resolve_placeholders(prefix: str, prompt=None, extra_pnginfo=None) -> str:
    """解析 %date:yyyyMMdd%(支持 yyyy/MM/dd/HH/mm/ss) 与 %节点名.字段% 引用。"""
    title_map = _build_title_map(extra_pnginfo)
    now = datetime.now()

    def date_sub(m):
        fmt = m.group(1)
        out = fmt
        for token, repl in [
            ("yyyy", f"{now.year:04d}"), ("MM", f"{now.month:02d}"), ("dd", f"{now.day:02d}"),
            ("HH", f"{now.hour:02d}"), ("mm", f"{now.minute:02d}"), ("ss", f"{now.second:02d}"),
        ]:
            out = out.replace(token, repl)
        return out

    def node_sub(m):
        return _lookup_node_value(m.group(1), m.group(2), prompt, title_map)

    out = _DATE_PLACEHOLDER_RE.sub(date_sub, prefix)
    out = _NODE_REF_RE.sub(node_sub, out)
    return out


def _sanitize_path_part(part: str) -> str:
    part = part.strip()
    part = re.sub(r'[<>:"/\\|?*]', "_", part)
    return part


# ---------------------------------------------------------------------------
# 节点1: TTS With Timestamps
# ---------------------------------------------------------------------------
class Qwen3TTSWithTimestamps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_text": ("STRING", {"multiline": True, "default": "你好，欢迎使用 Qwen3-TTS 同源时间戳合成。", "placeholder": "输入待合成文本(方言/情绪通过指令控制)"}),
                "server_url": ("STRING", {"default": _DEFAULT_SERVER}),
                "voice_type": (_VOICE_TYPES, {"default": "custom_voice", "tooltip": "custom_voice=预置音色+指令; voice_clone=克隆注册音色; voice_design=文字设计音色+指令"}),
                "voice": (_voice_options(), {"tooltip": "custom_voice: 预置音色(9种); voice_clone: 已注册克隆音色; voice_design: 无需选择"}),
                "instruct": ("STRING", {"default": "", "tooltip": "方言/情绪/语速指令, 如: 四川话，情绪激动，放慢语速。\n注意: voice_clone(克隆)模式模型无指令通道, 风格只来自参考音频(前端已禁用此框);\ncustom_voice/voice_design 为软控制, 方言想稳定建议 temperature≤0.8 + 固定 seed"}),
                "language": (_LANGUAGES, {"default": "Auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "tooltip": "单次生成上限; 过大可能触发上游已知死循环 issue#68"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                # 后端握手参数: 运行时下发给 8001 服务端动态应用(照 CosyVoice TTS With Timestamps 模式),
                # 空/0=不干预(用服务端启动参数)。服务端懒加载: 模型未加载/空闲超时卸载后,
                # 本次合成请求自动触发加载(HTTP 长超时兜底等待), 无需重启服务。
                "device": (
                    ["", "auto", "cuda", "cpu"],
                    {"default": "", "tooltip": "后端模型加载设备(运行时切换, 无需重启服务): 空=不干预(用服务端启动参数);\ncpu=内存加载(不占显存); cuda=显存加载; auto=有 CUDA 自动用 GPU。\n切换会卸载并重载模型(本次合成含加载耗时)"},
                ),
                "idle_timeout": (
                    "INT",
                    {"default": 0, "min": 0, "max": 86400, "tooltip": "后端空闲自动卸载秒数: 0=不干预(用服务端启动参数, 默认 0 常驻)。\n服务常驻, 空闲超时自动卸载模型释放显存, 下次请求自动重载"},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("AUDIO", "WORD_TIMESTAMPS", "SRT", "DURATION_SECONDS")
    FUNCTION = "generate"
    CATEGORY = _CATEGORY

    def generate(self, tts_text, server_url, voice_type="custom_voice", voice="", instruct="",
                 language="Auto", seed=0, max_new_tokens=1024, top_p=1.0, temperature=0.9,
                 repetition_penalty=1.05, device="", idle_timeout=0):
        if not tts_text or not tts_text.strip():
            raise ValueError("[Qwen3TTS] tts_text 为空，请输入待合成文本")

        payload = {
            "text": tts_text,
            "voice_type": voice_type,
            "voice": voice,
            "instruct": instruct,
            "language": language,
            "seed": int(seed),
            "max_new_tokens": int(max_new_tokens),
            "top_p": float(top_p),
            "temperature": float(temperature),
            "repetition_penalty": float(repetition_penalty),
        }
        # 后端握手: 仅显式指定才下发(空/0=不干预服务端启动参数), 与 CosyVoice 节点语义一致
        if device:
            payload["device"] = device
        if int(idle_timeout) > 0:
            payload["idle_timeout"] = float(int(idle_timeout))
        result = _post_json(server_url, payload, path="/synth")
        if "audio_base64" not in result:
            raise RuntimeError(f"[Qwen3TTS] 服务端错误: {result}")

        audio_out = {
            "waveform": _decode_audio_base64(result["audio_base64"]),
            "sample_rate": int(result.get("sample_rate", 24000)),
        }
        words = result.get("word_timestamps") or []
        srt = result.get("srt") or ""
        duration = float(result.get("duration_seconds", 0.0))
        return (audio_out, json.dumps(words, ensure_ascii=False, indent=2), srt, duration)


# ---------------------------------------------------------------------------
# 节点2: RegisterVoice (克隆音色注册)
# ---------------------------------------------------------------------------
class Qwen3TTSRegisterVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "音色名(唯一), 注册后出现在 TTS 节点 voice_clone 下拉"}),
                "ref_audio": ("AUDIO", {"tooltip": "参考音频(3-15秒最佳): 来自 LoadAudio/录音等节点的 AUDIO 输出"}),
                "ref_text": ("STRING", {"default": "", "multiline": True, "tooltip": "参考音频的转写文本(克隆 ICL 必需, 必须与音频内容一致, 否则对齐会漂移)"}),
                "server_url": ("STRING", {"default": _DEFAULT_SERVER}),
            },
            "optional": {
                "device": (
                    ["", "auto", "cuda", "cpu"],
                    {"default": "", "tooltip": "后端模型加载设备(注册会触发 base 模型加载): 空=不干预(用服务端启动参数)"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESULT", "VOICE_LIST")
    FUNCTION = "register"
    CATEGORY = _CATEGORY

    def register(self, name, ref_audio, ref_text, server_url=_DEFAULT_SERVER, device=""):
        name = (name or "").strip()
        if not name:
            raise ValueError("[Qwen3TTS] 音色名不能为空")
        if not ref_text or not ref_text.strip():
            raise ValueError("[Qwen3TTS] 克隆音色必须提供 ref_text(参考音频转写), 否则 ICL 对齐会失败")

        waveform = ref_audio.get("waveform")  # [B, C, T]
        sr = int(ref_audio.get("sample_rate", 24000))
        if waveform is None:
            raise ValueError("[Qwen3TTS] 参考音频无效")
        audio_np = waveform.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # 写 wav bytes → base64 上传
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV")
        wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        payload = {"name": name, "ref_audio": wav_b64, "ref_text": ref_text.strip()}
        if device:  # 后端握手: 仅显式指定才下发
            payload["device"] = device
        result = _post_json(server_url, payload, path="/register", timeout=120)
        voices = result.get("voices", [])
        return (f"注册成功: {name}", json.dumps(voices, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 节点3: DeleteVoice
# ---------------------------------------------------------------------------
class Qwen3TTSDeleteVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": ""}),
                "server_url": ("STRING", {"default": _DEFAULT_SERVER}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESULT",)
    FUNCTION = "delete"
    CATEGORY = _CATEGORY

    def delete(self, name, server_url=_DEFAULT_SERVER):
        name = (name or "").strip()
        if not name:
            raise ValueError("[Qwen3TTS] 音色名不能为空")
        result = _post_json(server_url, {"name": name}, path="/delete", timeout=60)
        return (f"已删除: {name}",)


# ---------------------------------------------------------------------------
# 节点4: SRT File Writer (仅连线输入 + 占位符)
# ---------------------------------------------------------------------------
class Qwen3TTSSRTFileWriter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "来自 DBW-Qwen3TTS-TTSWithTimestamps 的 SRT 输出(仅连线输入, 预览请用 PreviewAny)"}),
            },
            "optional": {
                "filename_prefix": (
                    "STRING",
                    {"default": "srt/qwen3tts",
                     "tooltip": "输出路径前缀, 支持子目录与占位符:\n"
                                "1) %date:yyyyMMdd% 日期(还可用 MM/dd/HH/mm/ss);\n"
                                "2) %节点名.字段% 引用其他节点参数(如 %Qwen3 TTS With Timestamps.voice%);\n"
                                "示例: audio/TTS/qwen3tts/%date:yyyyMMdd%/%Qwen3 TTS With Timestamps.voice%_"},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("FILE_PATH", "SRT")
    FUNCTION = "write"
    CATEGORY = _CATEGORY

    def write(self, srt, filename_prefix="srt/qwen3tts", unique_id=None, prompt=None, extra_pnginfo=None):
        srt = srt or ""
        if not srt.strip():
            return ("", srt)
        try:
            from folder_paths import get_output_directory
            base_dir = get_output_directory()
        except Exception:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))

        resolved = _resolve_placeholders(filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo).replace("\\", "/").strip("/")
        parts = [_sanitize_path_part(p) for p in resolved.split("/") if p and p.strip()]
        if not parts:
            parts = ["qwen3tts"]
        name_part = parts[-1]
        out_dir = os.path.join(base_dir, *parts[:-1]) if parts[:-1] else base_dir
        os.makedirs(out_dir, exist_ok=True)

        if name_part.endswith(".srt"):
            stem, path = name_part[:-4], os.path.join(out_dir, name_part)
            i = 1
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{stem}_{i:05}.srt")
                i += 1
        else:
            path = os.path.join(out_dir, name_part + ".srt")
            i = 1
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{name_part}_{i:05}.srt")
                i += 1

        with open(path, "w", encoding="utf-8") as f:
            f.write(srt)
        return (path, srt)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS["DBW-Qwen3TTS-TTSWithTimestamps"] = Qwen3TTSWithTimestamps
NODE_CLASS_MAPPINGS["DBW-Qwen3TTS-RegisterVoice"] = Qwen3TTSRegisterVoice
NODE_CLASS_MAPPINGS["DBW-Qwen3TTS-DeleteVoice"] = Qwen3TTSDeleteVoice
NODE_CLASS_MAPPINGS["DBW-Qwen3TTS-SRTFileWriter"] = Qwen3TTSSRTFileWriter

NODE_DISPLAY_NAME_MAPPINGS["DBW-Qwen3TTS-TTSWithTimestamps"] = "Qwen3 TTS With Timestamps"
NODE_DISPLAY_NAME_MAPPINGS["DBW-Qwen3TTS-RegisterVoice"] = "Qwen3 TTS Register Voice"
NODE_DISPLAY_NAME_MAPPINGS["DBW-Qwen3TTS-DeleteVoice"] = "Qwen3 TTS Delete Voice"
NODE_DISPLAY_NAME_MAPPINGS["DBW-Qwen3TTS-SRTFileWriter"] = "Qwen3 TTS SRT File Writer"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
