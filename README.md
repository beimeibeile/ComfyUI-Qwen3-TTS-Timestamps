# ComfyUI-Qwen3-TTS-Timestamps

Qwen3-TTS 1.7B 同源时间戳 ComfyUI 扩展工程（DBW 子工程）。

Qwen3-TTS 官方引擎只返回 `(wavs, sample_rate)`，**不提供时间戳**，因此本工程在合成后用
Qwen3-ForcedAligner-0.6B 对合成音频做二次强制对齐（transcript 与合成文本完全一致），
产出**同源**字级时间戳（WORD_TIMESTAMPS）与 SRT 字幕，杜绝“时间线对不上”问题。

## 架构

```
ComfyUI 节点 (nodes.py) ──HTTP──> FastAPI 服务 (server/tts_server.py, 端口 8001)
                                     ├─ Qwen3-TTS-12Hz-1.7B-CustomVoice   (指令语音 custom_voice)
                                     ├─ Qwen3-TTS-12Hz-1.7B-VoiceDesign   (文字设计音色 voice_design)
                                     ├─ Qwen3-TTS-12Hz-1.7B-Base          (克隆 voice_clone，与 base 共用实例)
                                     └─ Qwen3-ForcedAligner-0.6B  ──> word_timestamps ──> SRT
```

- 三种 voice_type 按需切换，切换时自动卸载旧模型（懒加载 + idle 超时卸载，显存友好）。
- 音色注册（RegisterVoice）扩展为「参考音频 + ref_text 转写」：注册时用 Base 模型
  `create_voice_clone_prompt` 预计算克隆 prompt 并落盘，合成时直接复用，无需每次重算。

## 目录结构

```
ComfyUI-Qwen3-TTS-Timestamps/
├── nodes.py               # 4 个 DBW-Qwen3TTS-* 节点
├── __init__.py            # WEB_DIRECTORY="./web"
├── web/js/qwen3tts.js     # 前端：voice 下拉按 voice_type 联动 + 刷新
├── server/tts_server.py   # FastAPI 服务（端口 8001）
├── server/voices/         # 注册音色数据（.pt prompt / .wav，已 gitignore）
├── tests/e2e_synth.py     # custom_voice / voice_design 端到端
├── tests/e2e_clone.py     # voice_clone 注册→合成→清理 端到端
├── start_qwen3_tts.bat    # 启动服务（GBK-936）
├── stop_qwen3_tts.bat     # 停止服务
└── pyproject.toml         # 依赖清单
```

## 节点（统一注册规则 DBW-Qwen3TTS-*）

| 节点 | 输入 | 输出 |
|---|---|---|
| `DBW-Qwen3TTS-TTSWithTimestamps` | text / server_url / voice_type / voice / instruct / language / seed / max_new_tokens / top_p / temperature / repetition_penalty | AUDIO / WORD_TIMESTAMPS / SRT / DURATION_SECONDS |
| `DBW-Qwen3TTS-RegisterVoice` | name + ref_audio(AUDIO) + ref_text(必填) | ok / voices |
| `DBW-Qwen3TTS-DeleteVoice` | name | ok / voices |
| `DBW-Qwen3TTS-SRTFileWriter` | srt(纯连线 forceInput) + filename_prefix | 写盘路径 |

- `voice_type` 联动：custom_voice → 9 个预置音色下拉；voice_clone → 已注册音色下拉；
  voice_design → 音色下拉清空禁用（用 instruct 文字设计音色）。
- **instruct 按模式联动（前端）**：voice_clone（Base 模型）**无 instruct 通道**（qwen_tts 的
  `generate_voice_clone` 不构建 instruct_ids），指令输入会被静默忽略，前端已禁用该输入框；
  custom_voice / voice_design 的 instruct 是**软控制**（非硬约束）。
- 注册/删除后前端自动刷新音色列表（executed 事件 + 20s 轮询 + 手动“刷新音色列表”按钮）。
- SRT File Writer 的 `srt` 为纯连线输入，前端不渲染文本框（直接删除控件）。

## 安装

1. **依赖**（与 ComfyUI 共用 python 环境；若 python 不在 PATH，用你的 python.exe 完整路径替换 `python`）：
   ```bat
   python -m pip install "qwen-tts==0.1.1" "qwen-asr==0.0.6" "transformers==4.57.3" fastapi uvicorn soundfile requests
   ```
2. **模型**（放在 ComfyUI/models/ 下即可，服务端只做存在性检查、不自动下载）：
   - `models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice`（约 3.83GB）
   - `models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
   - `models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base`（voice_clone 克隆用）
   - `models/Qwen3-ASR/Qwen3-ForcedAligner-0.6B`（对齐用，可 ModelScope 手动拉取）
   - 已存在即直接加载，缺失时服务端报错并提示完整路径。
   - `models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base`（voice_clone 克隆用）
   - `models/Qwen3-ASR/Qwen3-ForcedAligner-0.6B`（对齐用）
3. **节点**：将本目录复制到 `ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Timestamps/`，重启 ComfyUI。
4. **服务端**：双击 `start_qwen3_tts.bat`（或手动 `python server/tts_server.py`），
   健康检查 `http://127.0.0.1:8001/health`。

## 服务端接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 服务状态 |
| `/spks` | GET | 预置 + 注册音色列表 |
| `/register` | POST | name + ref_audio(base64 wav) + ref_text 注册克隆音色 |
| `/delete` | POST | name 删除注册音色 |
| `/synth` | POST | text/voice_type/voice/instruct/language/seed/... → audio_base64(WAV)+sample_rate+word_timestamps+srt+duration_seconds |

`/synth` 响应示例：
```json
{
  "audio_base64": "...",
  "sample_rate": 24000,
  "word_timestamps": [{"text": "你", "start_time": 0.08, "end_time": 0.16}, "..."],
  "srt": "1\n00:00:00,080 --> 00:00:00,480\n你好\n\n2\n...",
  "duration_seconds": 5.2
}
```

## 测试

```bat
python tests\e2e_synth.py   :: custom_voice/voice_design 端到端
python tests\e2e_clone.py   :: voice_clone 注册→合成→清理
```
（若 python 不在 PATH，用你的 python.exe 完整路径替换 `python`。）

实测（本机）：
- custom_voice（1.7B + 四川话指令）：约 17-18s/次（模型已缓存），21 字级时间戳，
  指令文本零泄漏进 WORD_TIMESTAMPS / SRT。
- voice_clone：注册 2.5s，克隆合成 6.7s，16 字级时间戳，SRT 按标点断句。
- voice_design：13 字级时间戳，instruct 零泄漏。

## 已知问题与事实（重要）

1. **seed 是随机种子，不是语速倍率**：本工程 seed 透传给 Qwen3-TTS 控制随机性；
   语速由模型自身决定，无独立语速参数。同 seed 同参数可复现（新进程内）。
2. **sox 补丁**：本机无 sox（qwen_tts 的 XVectorExtractor 用外部 sox 做 -6dBFS 峰值归一），
   服务端已用 numpy 等效实现替换（日志可见 `已应用等效峰值归一补丁(-6 dBFS)`）。
3. **对齐前提**：ForcedAligner 要求 transcript 与合成文本完全一致；长文本请保证
   max_new_tokens 足够，避免被截断导致对齐错位。
4. **方言/情绪走 instruct，但属软控制**：custom_voice 的 instruct（如“四川话，情绪激动”）
   可注入口音与情绪，但 1.7B 模型是**软跟随**——方言“时有时无”主要来自采样随机
   （seed 每次 randomize + temperature≈1 时波动大）。**想让方言稳定**：
   `temperature ≤ 0.8` + `seed 固定`（不 randomize），必要时指令写全（“请用四川话朗读整段文本”）。
   voice_clone 参考音频的方言/情绪会被过滤，只复制音色（Qwen3-TTS 官方行为）。
5. **克隆音色中性化 + 无指令通道**：voice_clone 仅迁移音色（声纹），参考源的方言、情绪、
   口音不会保留，且该模式**不支持任何 instruct**（模型接口无此参数，前端已禁用输入框）——
   需要方言/情绪请改用 custom_voice 或 voice_design。
6. **死循环防御**：text 为空/全空白直接报 400，不进入合成。
7. **云备选**：若本地显存不足，可改用阿里云 Qwen3-TTS API（返回同源时间戳），
   本工程专注本地 ComfyUI 场景。

## License / 说明

- 模型权重（Qwen3-TTS / Qwen3-ForcedAligner）版权归阿里 Qwen 团队，遵循各自开源协议。
- 本工程为本地 ComfyUI 扩展，节点命名遵循总仓统一注册规则 `DBW-<工程包>-<节点>`。
