# coding=utf-8
"""voice_clone 链路测试: synth 参考段 -> /register -> /synth voice_clone -> 对齐验证。
运行: python tests\e2e_clone.py (python 不在 PATH 时用你的 python.exe 完整路径)
"""
import base64
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8001"
REF_TEXT = "你好，欢迎使用 Qwen3 语音合成，这里是克隆音色测试。"


def synth(payload):
    r = requests.post(BASE + "/synth", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    # 1) 先生成一段参考音频(custom_voice)
    ref = synth({
        "text": REF_TEXT,
        "voice_type": "custom_voice",
        "voice": "serena",
        "instruct": "",
        "language": "Chinese",
        "seed": 7,
        "max_new_tokens": 512,
    })
    ref_wav = base64.b64decode(ref["audio_base64"])
    print(f"REF_AUDIO_BYTES={len(ref_wav)} DURATION={ref.get('duration_seconds')}s")

    # 2) 注册克隆音色
    t0 = time.time()
    r = requests.post(BASE + "/register", json={
        "name": "e2e_clone_test",
        "ref_audio": base64.b64encode(ref_wav).decode("ascii"),
        "ref_text": REF_TEXT,
    }, timeout=300)
    r.raise_for_status()
    reg = r.json()
    print(f"REGISTER={reg.get('ok')} VOICES={reg.get('voices')} ELAPSED={time.time()-t0:.1f}s")

    # 3) 克隆合成(使用注册音色 prompt)
    t0 = time.time()
    out = synth({
        "text": "克隆音色测试，说话人音色与参考一致。",
        "voice_type": "voice_clone",
        "voice": "e2e_clone_test",
        "instruct": "",
        "language": "Chinese",
        "seed": 8,
        "max_new_tokens": 512,
    })
    words = out.get("word_timestamps") or []
    joined = "".join(w.get("text", "") for w in words)
    print(f"CLONE_OK DURATION={out.get('duration_seconds')}s WORDS={len(words)} ELAPSED={time.time()-t0:.1f}s")
    print("CLONE_SRT_HEAD:\n" + "\n".join((out.get("srt") or "").splitlines()[:6]))
    if "克隆" not in joined:
        print("CLONE_TEXT_FAIL:", joined)
        sys.exit(1)
    if not words:
        sys.exit(2)

    # 4) 清理测试音色
    r = requests.post(BASE + "/delete", json={"name": "e2e_clone_test"}, timeout=60)
    r.raise_for_status()
    print("DELETE_OK", r.json().get("ok"))
    print("CLONE_E2E_OK")


if __name__ == "__main__":
    main()
