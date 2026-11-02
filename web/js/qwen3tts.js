// Qwen3-TTS-Timestamps 前端扩展:
// 1) voice 下拉按 voice_type 联动: custom_voice→预置音色, voice_clone→注册克隆音色,
//    voice_design→无音色(下拉禁用);
// 2) 注册/删除音色后所有主节点下拉实时刷新(无需重启浏览器/页面)。
//
// 机制与 CosyVoice 归档工程同源(ComfyUI 0.34 / frontend 1.49.6 验证):
// - 下拉数据源是 widget.options.values; canvas 与 Vue 两种渲染在打开下拉时都会实时读取,
//   故采用【原地更新数组】(保持引用, 兼容响应式)。
// - 切换当前值必须 widget.setValue(value, {node, canvas})(受控组件)。
// - widget.disabled 是跨模式可靠的灰色禁用接口。
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const TTS_CLASSES = ["DBW-Qwen3TTS-TTSWithTimestamps"];
const REGISTER_CLASSES = ["DBW-Qwen3TTS-RegisterVoice"];
const DELETE_CLASSES = ["DBW-Qwen3TTS-DeleteVoice"];

/* ================= 1. voice 下拉刷新(voice_type 联动) ================= */

const getTTSNodes = () => {
    const graph = app.graph;
    if (!graph || !graph._nodes) return [];
    return graph._nodes.filter((n) => TTS_CLASSES.includes(n.comfyClass));
};

// 按 voice_type 填充对应音色列表; 服务异常时静默返回(由事件/轮询下次重试)
const refreshVoice = async (node) => {
    if (!node || !node.widgets) return;
    const urlW = node.widgets.find((w) => w.name === "server_url");
    const typeW = node.widgets.find((w) => w.name === "voice_type");
    const voiceW = node.widgets.find((w) => w.name === "voice");
    if (!typeW || !voiceW || !voiceW.options) return;

    const mode = typeW.value;
    // voice_design 无需选择音色 → 清空值并禁用(列表保留 [""] 保证 ComfyUI 校验通过)
    if (mode === "voice_design") {
        if (Array.isArray(voiceW.options.values)) {
            voiceW.options.values.length = 0;
            voiceW.options.values.push("");
        } else {
            voiceW.options.values = [""];
        }
        if (voiceW.value !== "") {
            if (typeof voiceW.setValue === "function") {
                voiceW.setValue("", { node, canvas: app.canvas });
            } else {
                voiceW.value = "";
            }
            if (typeof voiceW.callback === "function") voiceW.callback("");
        }
        if (voiceW.disabled !== true) voiceW.disabled = true;
        node.setDirtyCanvas?.(true, true);
        return;
    }
    if (voiceW.disabled !== false) voiceW.disabled = false;

    const base = (urlW ? urlW.value : "http://127.0.0.1:8001") || "http://127.0.0.1:8001";
    let data;
    try {
        const resp = await fetch(base.replace(/\/+$/, "") + "/spks");
        if (!resp.ok) return;
        data = await resp.json();
    } catch (e) {
        return;
    }
    if (!data || typeof data !== "object") return;
    const list = (mode === "voice_clone" ? data.voices : data.presets)
        .filter((s) => typeof s === "string" && s);
    if (!list.length) return;

    if (Array.isArray(voiceW.options.values)) {
        voiceW.options.values.length = 0;
        voiceW.options.values.push(...list);
    } else {
        voiceW.options.values = list.slice();
    }

    if (!list.includes(voiceW.value)) {
        if (typeof voiceW.setValue === "function") {
            voiceW.setValue(list[0], { node, canvas: app.canvas });
        } else {
            voiceW.value = list[0];
        }
        if (typeof voiceW.callback === "function") voiceW.callback(voiceW.value);
    }
    node.setDirtyCanvas?.(true, true);
};

const refreshAll = () => {
    for (const n of getTTSNodes()) refreshVoice(n);
};

// 防抖: 一次执行中多节点 executed 事件密集到达, 合并刷新
let _lastSchedule = 0;
const scheduleRefresh = (delay = 300) => {
    const now = Date.now();
    if (now - _lastSchedule < delay) return;
    _lastSchedule = now;
    setTimeout(refreshAll, delay);
};

/* ================= 2. 扩展注册 ================= */

app.registerExtension({
    name: "Qwen3TTS.FrontendEnhance",

    setup() {
        api.addEventListener("executed", () => scheduleRefresh(300));
        api.addEventListener("execution_success", () => scheduleRefresh(200));

        // 多时点初始化(覆盖页面加载早期 graph 未就绪)
        [800, 2000, 5000].forEach((ms) =>
            setTimeout(refreshAll, ms)
        );

        // 轻量轮询: 每 20s 同步一次, 覆盖外部增删/服务后启动(fetch 失败静默)
        setInterval(() => {
            if (getTTSNodes().length) refreshAll();
        }, 20000);
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        // Register/Delete 执行后自动同步所有主节点下拉
        if (REGISTER_CLASSES.includes(nodeData.name) || DELETE_CLASSES.includes(nodeData.name)) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                scheduleRefresh(400);
                return r;
            };
        }
        // 主节点: 执行后刷新 + 选中时刷新/重应用
        if (TTS_CLASSES.includes(nodeData.name)) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                setTimeout(() => refreshVoice(this), 300);
                return r;
            };
            const onSelected = nodeType.prototype.onSelected;
            nodeType.prototype.onSelected = function (...args) {
                const r = onSelected ? onSelected.apply(this, args) : undefined;
                refreshVoice(this);
                return r;
            };
        }
    },

    nodeCreated(node) {
        if (!TTS_CLASSES.includes(node.comfyClass)) return;

        // 手动刷新按钮
        try {
            if (typeof node.addWidget === "function") {
                node.addWidget("button", "刷新音色列表", null, () => refreshVoice(node));
            }
        } catch (e) { /* 个别渲染模式按钮不渲染, 自动刷新仍生效 */ }

        // voice_type 变化 → 即时切换音色列表
        const typeW = node.widgets && node.widgets.find((w) => w.name === "voice_type");
        if (typeW) {
            const origCb = typeW.callback;
            typeW.callback = function (...args) {
                const r = origCb ? origCb.apply(this, args) : undefined;
                refreshVoice(node);
                return r;
            };
        }
        setTimeout(() => refreshVoice(node), 800);
    },
});
