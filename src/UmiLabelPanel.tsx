import { Immutable, MessageEvent, PanelExtensionContext, Time, Topic } from "@lichtblick/suite";
import React, { ReactElement, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

type Segment = { startSec: number; endSec: number; prompt: string };
type PanelState = { segments: Segment[]; pendingStartSec?: number; promptText?: string };

function toSec(t: Time | undefined): number | undefined {
  if (!t) return undefined;
  // Lichtblick Time: { sec: number; nsec: number }
  if (typeof t.sec === "number" && typeof t.nsec === "number") {
    return t.sec + t.nsec * 1e-9;
  }
  return undefined;
}

function downloadJson(filename: string, obj: unknown) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function UmiCropPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const initial = (context.initialState as Partial<PanelState> | undefined) ?? {};

  const [topics, setTopics] = useState<undefined | Immutable<Topic[]>>();
  const [messages, setMessages] = useState<undefined | Immutable<MessageEvent[]>>();

  const [currentTimeSec, setCurrentTimeSec] = useState<number | undefined>(undefined);
  const [pendingStartSec, setPendingStartSec] = useState<number | undefined>(initial.pendingStartSec);
  const [segments, setSegments] = useState<Segment[]>(() => initial.segments ?? []);

  // ✅ promptText 定义在这里（之前缺失导致 End 直接报错）
  const [promptText, setPromptText] = useState<string>(initial.promptText ?? "");

  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();
  const [status, setStatus] = useState<string>("");

  // persist to layout
  useEffect(() => {
    context.saveState({ segments, pendingStartSec, promptText } satisfies PanelState);
  }, [context, segments, pendingStartSec, promptText]);

  useLayoutEffect(() => {
    context.onRender = (renderState, done) => {
      setRenderDone(() => done);
      setTopics(renderState.topics);
      setMessages(renderState.currentFrame);
      setCurrentTimeSec(toSec(renderState.currentTime));
    };

    context.watch("topics");
    context.watch("currentFrame");
    context.watch("currentTime");
  }, [context]);

  useEffect(() => {
    renderDone?.();
  }, [renderDone]);

  const timeText = useMemo(
    () => (currentTimeSec == null ? "n/a" : `${currentTimeSec.toFixed(3)} s`),
    [currentTimeSec],
  );

  function markStart() {
    if (currentTimeSec == null) {
      setStatus("currentTimeSec is undefined. 请先在顶部时间轴点击/拖动一次播放头。");
      return;
    }
    setPendingStartSec(currentTimeSec);
    setStatus(`Start set at ${currentTimeSec.toFixed(3)} s`);
  }

  function markEnd() {
    try {
      if (pendingStartSec == null) {
        setStatus("请先点击 Start。");
        return;
      }
      if (currentTimeSec == null) {
        setStatus("currentTimeSec is undefined. 请先在顶部时间轴点击/拖动一次播放头。");
        return;
      }

      const start = Math.min(pendingStartSec, currentTimeSec);
      const end = Math.max(pendingStartSec, currentTimeSec);

      const seg: Segment = { startSec: start, endSec: end, prompt: (promptText ?? "").trim() };

      setSegments((prev) => {
        const next = [...prev, seg];
        console.log("[umi] add segment", seg, "count", next.length);
        return next;
      });

      setPendingStartSec(undefined);
      setStatus(`Created segment: ${start.toFixed(3)} → ${end.toFixed(3)} (prompt="${seg.prompt}")`);
    } catch (e) {
      console.error("[umi] markEnd error:", e);
      setStatus(`markEnd error: ${String(e)}`);
    }
  }

  function exportSegments() {
    if (segments.length === 0) {
      setStatus("No segments to export yet.");
      return;
    }
    downloadJson("segments.json", { segments });
    setStatus(`Exported ${segments.length} segments to segments.json`);
  }

  function clearAll() {
    setSegments([]);
    setPendingStartSec(undefined);
    setStatus("Cleared.");
  }

  const canSeek = typeof context.seekPlayback === "function";
  function seekTo(t: number) {
    context.seekPlayback?.(t);
  }
  function removeRow(idx: number) {
    setSegments((prev) => prev.filter((_, i) => i !== idx));
  }

  return (
    <div style={{ padding: 12, fontFamily: "sans-serif", pointerEvents: "auto" }}>
      <h2 style={{ margin: "0 0 8px" }}>Umi Crop Labeler</h2>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 10 }}>
        <div style={{ padding: "2px 8px", border: "1px solid #ccc", borderRadius: 6 }}>
          currentTime: <b>{timeText}</b>
        </div>
        {pendingStartSec != null ? (
          <div style={{ padding: "2px 8px", border: "1px solid #ccc", borderRadius: 6 }}>
            start: <b>{pendingStartSec.toFixed(3)} s</b>
          </div>
        ) : (
          <div style={{ color: "#666" }}>在顶部时间轴定位到某个时刻，然后点 Start</div>
        )}
      </div>

      {status ? (
        <div style={{ margin: "8px 0", padding: "6px 8px", border: "1px solid #ddd", borderRadius: 6, color: "#333" }}>
          {status}
        </div>
      ) : null}

      {/* ✅ prompt 输入框（不再依赖 window.prompt/alert） */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>Prompt（End 时写入 segment）：</div>
        <input
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          placeholder="e.g., pick the bottle and put into the box"
          style={{ width: "100%", padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6 }}
        />
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
        <button onClick={markStart}>Start @ currentTime</button>
        <button onClick={markEnd}>End @ currentTime</button>
        <button onClick={exportSegments}>Export segments.json</button>
        <button onClick={clearAll}>Clear</button>
      </div>

      <h3 style={{ margin: "0 0 8px" }}>Segments ({segments.length})</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 700 }}>
          <thead>
            <tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Start (s)</th>
              <th style={thStyle}>End (s)</th>
              <th style={thStyle}>Duration (s)</th>
              <th style={thStyle}>Prompt</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {segments.length === 0 ? (
              <tr>
                <td style={tdStyle} colSpan={6}>
                  No segments yet. 先点 Start，再点 End。
                </td>
              </tr>
            ) : (
              segments.map((s, idx) => (
                <tr key={`${s.startSec}-${s.endSec}-${idx}`}>
                  <td style={tdStyle}>{idx}</td>
                  <td style={tdStyle}>{s.startSec.toFixed(3)}</td>
                  <td style={tdStyle}>{s.endSec.toFixed(3)}</td>
                  <td style={tdStyle}>{(s.endSec - s.startSec).toFixed(3)}</td>
                  <td style={{ ...tdStyle, wordBreak: "break-word" }}>{s.prompt}</td>
                  <td style={tdStyle}>
                    <button onClick={() => seekTo(s.startSec)} disabled={!canSeek} style={{ marginRight: 8 }}>
                      Seek
                    </button>
                    <button onClick={() => removeRow(idx)}>Delete</button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <details style={{ marginTop: 14 }}>
        <summary style={{ cursor: "pointer" }}>Debug</summary>
        <div style={{ marginTop: 8 }}>
          <div>topics: {(topics ?? []).length}</div>
          <div>currentFrame messages: {messages?.length ?? 0}</div>
          <div>seekPlayback supported: {String(canSeek)}</div>
          <div>currentTimeSec: {currentTimeSec ?? "undefined"}</div>
          <div>pendingStartSec: {pendingStartSec ?? "undefined"}</div>
          <div>segments count: {segments.length}</div>
        </div>
      </details>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  borderBottom: "1px solid #ccc",
  padding: "6px 8px",
  background: "#f6f6f6",
  fontWeight: 700,
};

const tdStyle: React.CSSProperties = {
  borderBottom: "1px solid #eee",
  padding: "6px 8px",
  verticalAlign: "top",
};

export function initUmiLabelPanel(context: PanelExtensionContext): () => void {
  // ✅ 强制允许接收鼠标事件（有些环境会被 overlay 影响）
  context.panelElement.style.pointerEvents = "auto";
  context.panelElement.style.userSelect = "text";

  const root = createRoot(context.panelElement);
  root.render(<UmiCropPanel context={context} />);

  return () => root.unmount();
}
