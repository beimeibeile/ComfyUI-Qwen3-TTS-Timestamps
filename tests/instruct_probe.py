# coding=utf-8
"""instruct 能力探针:
1) voice_clone 是否有 instruct 通道(应无: 有/无指令结果完全一致)
2) custom_voice instruct 影响(固定 seed 下 duration/words 差异)
3) temperature 对"方言稳定性"的影响(同 seed 不同温度)
"""
import requests

SRV = "http://127.0.0.1:8001"
TEXT = "大家好，今天我们来聊聊人工智能。这是一个测试句子，用来验证指令效果。"


def synth(voice_type, voice, instruct, seed=42, temperature=0.9, **kw):
    p = {
        "text": TEXT, "voice_type": voice_type, "voice": voice, "instruct": instruct,
        "language": "Chinese", "seed": seed, "max_new_tokens": 1024, "top_p": 1.0,
        "temperature": temperature, "repetition_penalty": 1.05,
    }
    p.update(kw)
    r = requests.post(SRV + "/synth", json=p, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    # 1) voice_clone: instruct 有/无 → 应完全一致(证明指令被忽略/无通道)
    print("== 1) voice_clone instruct 通道验证 (seed=11) ==")
    a = synth("voice_clone", "周鸿祎", "四川话，情绪激动", seed=11)
    b = synth("voice_clone", "周鸿祎", "", seed=11)
    same = (
        a["duration_seconds"] == b["duration_seconds"]
        and [w["text"] for w in a["word_timestamps"]] == [w["text"] for w in b["word_timestamps"]]
    )
    print(f"   instruct='四川话，情绪激动' dur={a['duration_seconds']:.2f}s words={len(a['word_timestamps'])}")
    print(f"   instruct=''                dur={b['duration_seconds']:.2f}s words={len(b['word_timestamps'])}")
    print(f"   => instruct 被忽略(无通道): {same}")

    # 2) custom_voice: 无指令 vs 四川话(固定 seed=42, temp=0.9)
    print("== 2) custom_voice instruct 影响 (seed=42, temp=0.9) ==")
    c = synth("custom_voice", "serena", "", seed=42, temperature=0.9)
    d = synth("custom_voice", "serena", "四川话，情绪激动", seed=42, temperature=0.9)
    e = synth("custom_voice", "serena", "请用四川话朗读整段文本，语气亲切", seed=42, temperature=0.9)
    for tag, r in [("无指令", c), ("四川话，情绪激动", d), ("请用四川话朗读整段文本", e)]:
        print(f"   [{tag}] dur={r['duration_seconds']:.2f}s words={len(r['word_timestamps'])}")

    # 3) temperature 采样影响(同 seed=7, 同 instruct)
    print("== 3) temperature 采样影响 (seed=7, instruct='四川话，情绪激动') ==")
    f = synth("custom_voice", "serena", "四川话，情绪激动", seed=7, temperature=0.9)
    g = synth("custom_voice", "serena", "四川话，情绪激动", seed=7, temperature=0.6)
    print(f"   temp=0.9 dur={f['duration_seconds']:.2f}s words={len(f['word_timestamps'])}")
    print(f"   temp=0.6 dur={g['duration_seconds']:.2f}s words={len(g['word_timestamps'])}")

    print("\n说明: 方言'明显度'需人耳判断; 以上数字验证的是——指令是否进入生成、采样参数是否改变输出。")


if __name__ == "__main__":
    main()
