# coding=utf-8
"""端到端测试: /synth custom_voice(预置音色+四川话指令) → 验证同源字级时间戳链路。
运行: python tests\e2e_synth.py (python 不在 PATH 时用你的 python.exe 完整路径)
"""
import json
import sys
import time
import base64

import requests

BASE = "http://127.0.0.1:8001"


def main():
    t0 = time.time()
    payload = {
        "text": "你好，欢迎使用 Qwen3 语音合成，这里是同源时间戳测试。",
        "voice_type": "custom_voice",
        "voice": "serena",
        "instruct": "四川话，情绪平稳。",
        "language": "Chinese",
        "seed": 42,
        "max_new_tokens": 512,
        "top_p": 1.0,
        "temperature": 0.9,
        "repetition_penalty": 1.05,
    }
    r = requests.post(BASE + "/synth", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    dur = time.time() - t0
    print(f"ELAPSED={dur:.1f}s")
    print("HAS_AUDIO=", "audio_base64" in data, "SR=", data.get("sample_rate"))
    wav = base64.b64decode(data["audio_base64"])
    print("AUDIO_BYTES=", len(wav), "DURATION_S=", data.get("duration_seconds"))
    words = data.get("word_timestamps") or []
    print("WORD_COUNT=", len(words))
    if words:
        print("WORDS_HEAD=", json.dumps(words[:5], ensure_ascii=False))
        print("WORDS_TAIL=", json.dumps(words[-3:], ensure_ascii=False))
    srt = data.get("srt") or ""
    print("SRT_LINES=", len(srt.splitlines()))
    print("SRT_HEAD:\n" + "\n".join(srt.splitlines()[:8]))
    # 校验: 文本应朗读正确(不含指令文本); 指令词不应出现在 word_timestamps 文本中
    joined = "".join(w.get("text", "") for w in words)
    print("INSTRUCT_LEAK=", ("四川话" in joined))
    if "四川话" in joined:
        sys.exit(2)
    print("E2E_OK")


if __name__ == "__main__":
    main()
