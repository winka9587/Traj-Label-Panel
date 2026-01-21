import { Immutable, MessageEvent, PanelExtensionContext, Time, Topic } from "@lichtblick/suite";
import React, { ReactElement, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

/** ===== Types ===== */
type Segment = { startSec: number; endSec: number; prompt: string };
type SegmentRow = Segment & { id: string };

type PanelState = {
  segments: Segment[]; // persisted without id
  pendingStartSec?: number;
  promptText?: string;

  // timeline window settings
  windowSec?: number;
  followCursor?: boolean;
  windowCenterSec?: number;

  // NEW
  outputDir?: string;
};

type DragHandle = "start" | "end";
type DragState = { segId: string; handle: DragHandle } | null;

const EPS = 1e-3;

function toSec(t: Time | undefined): number | undefined {
  if (!t) return undefined;
  if (typeof t.sec === "number" && typeof t.nsec === "number") return t.sec + t.nsec * 1e-9;
  return undefined;
}

function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

// allow touching boundaries: aEnd==bStart ok
function overlaps(aStart: number, aEnd: number, bStart: number, bEnd: number) {
  return aStart < bEnd - EPS && aEnd > bStart + EPS;
}

function normalizeSegment(s: Segment): Segment {
  const start = Math.min(s.startSec, s.endSec);
  const end = Math.max(s.startSec, s.endSec);
  return { ...s, startSec: start, endSec: end };
}

function normalizeRow(r: SegmentRow): SegmentRow {
  const s = normalizeSegment(r);
  return { ...s, id: r.id };
}

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** Try to derive a recording base name (without extension) from panel context.
 *  This is best-effort because not all data sources expose a filename.
 */
function getRecordingBaseName(context: PanelExtensionContext): string | undefined {
  const anyCtx = context as unknown as Record<string, unknown>;

  // Helper to safely extract string fields
  const pickString = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);

  // common candidates (best effort)
  // 1) context.dataSource?.name / .id / .fileName etc
  const dataSource = anyCtx["dataSource"] as Record<string, unknown> | undefined;
  const dsName =
    pickString(dataSource?.["name"]) ||
    pickString(dataSource?.["fileName"]) ||
    pickString(dataSource?.["filename"]) ||
    pickString(dataSource?.["id"]);

  // 2) context.playerState?.name / context.playbackState?.name etc
  const playerState = anyCtx["playerState"] as Record<string, unknown> | undefined;
  const psName =
    pickString(playerState?.["name"]) ||
    pickString(playerState?.["fileName"]) ||
    pickString(playerState?.["filename"]) ||
    pickString(playerState?.["id"]);

  // 3) context?.title
  const title = pickString(anyCtx["title"]);

  const raw = dsName || psName || title;
  if (!raw) return undefined;

  // normalize: remove path + extension
  const last = raw.split(/[\\/]/).pop() ?? raw;
  const noExt = last.replace(/\.(mcap|bag|db3|json|log)$/i, "");
  return noExt || undefined;
}

function safeFilename(s: string) {
  // keep it simple, avoid weird chars
  return s.replace(/[^\w.\-]+/g, "_");
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

/** ===== Panel ===== */
function UmiCropPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const initial = (context.initialState as Partial<PanelState> | undefined) ?? {};

  const [topics, setTopics] = useState<undefined | Immutable<Topic[]>>();
  const [messages, setMessages] = useState<undefined | Immutable<MessageEvent[]>>();

  const [currentTimeSec, setCurrentTimeSec] = useState<number | undefined>(undefined);
  const [pendingStartSec, setPendingStartSec] = useState<number | undefined>(initial.pendingStartSec);

  const [segments, setSegments] = useState<SegmentRow[]>(() => {
    const base = (initial.segments ?? []).map(normalizeSegment);
    return base.map((s) => ({ ...s, id: newId() }));
  });

  // 1) prompt default value
  const [promptText, setPromptText] = useState<string>(initial.promptText ?? "pick the bottle and put into the box");

  const [status, setStatus] = useState<string>("");
  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();

  // fixed preview window
  const [windowSec, setWindowSec] = useState<number>(initial.windowSec ?? 30);
  const [followCursor, setFollowCursor] = useState<boolean>(initial.followCursor ?? true);
  const [windowCenterSec, setWindowCenterSec] = useState<number | undefined>(initial.windowCenterSec);

  // 2) output path input w/ default
  const [outputDir, setOutputDir] = useState<string>(initial.outputDir ?? "/data/label_data/seg");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<DragState>(null);

  // table scroll refs
  const rowRefs = useRef<Record<string, HTMLTableRowElement | null>>({});

  // persist (strip id)
  useEffect(() => {
    const plain: Segment[] = segments.map((r) => {
      const n = normalizeRow(r);
      return { startSec: n.startSec, endSec: n.endSec, prompt: n.prompt };
    });
    context.saveState({
      segments: plain,
      pendingStartSec,
      promptText,
      windowSec,
      followCursor,
      windowCenterSec,
      outputDir,
    } satisfies PanelState);
  }, [context, segments, pendingStartSec, promptText, windowSec, followCursor, windowCenterSec, outputDir]);

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

  const canSeek = typeof context.seekPlayback === "function";
  const seekTo = (t: number) => context.seekPlayback?.(t);

  const timeText = useMemo(
    () => (currentTimeSec == null ? "n/a" : `${currentTimeSec.toFixed(3)} s`),
    [currentTimeSec],
  );

  const normalizedSegments = useMemo(() => {
    return segments.map(normalizeRow).sort((a, b) => a.startSec - b.startSec);
  }, [segments]);

  // which segment is "active" for current time?
  const activeSegId = useMemo(() => {
    if (currentTimeSec == null) return undefined;
    for (const r of normalizedSegments) {
      if (currentTimeSec >= r.startSec - EPS && currentTimeSec <= r.endSec + EPS) return r.id;
    }
    return undefined;
  }, [currentTimeSec, normalizedSegments]);

  // rule 5: current time in any segment => cannot create start/end
  const isCurrentInsideAnySeg = activeSegId != null;

  function canCreateAtCurrent() {
    return currentTimeSec != null && !isCurrentInsideAnySeg;
  }

  function markStart() {
    if (!canCreateAtCurrent()) {
      setStatus(currentTimeSec == null ? "currentTimeSec is undefined." : "当前时间在已有 segment 内，禁止重合。");
      return;
    }
    setPendingStartSec(currentTimeSec);
    setStatus(`Start set at ${currentTimeSec!.toFixed(3)} s`);
  }

  function markEnd() {
    if (pendingStartSec == null) {
      setStatus("请先点击 Start。");
      return;
    }
    if (!canCreateAtCurrent()) {
      setStatus("当前时间在已有 segment 内，禁止重合。");
      return;
    }

    const start = Math.min(pendingStartSec, currentTimeSec!);
    const end = Math.max(pendingStartSec, currentTimeSec!);

    for (const o of normalizedSegments) {
      if (overlaps(start, end, o.startSec, o.endSec)) {
        setStatus(`与已有 segment 重叠，创建失败（${o.startSec.toFixed(3)}→${o.endSec.toFixed(3)}）`);
        return;
      }
    }

    const seg: SegmentRow = {
      id: newId(),
      startSec: start,
      endSec: end,
      // 1) prompt default is already in state; End stores current promptText
      prompt: (promptText ?? "").trim(),
    };

    setSegments((prev) => [...prev, seg]);
    setPendingStartSec(undefined);
    setStatus(`Created segment: ${start.toFixed(3)} → ${end.toFixed(3)}`);
  }

  // 3) export json filename = same as current mcap base name (best effort)
  function exportSegments() {
    if (normalizedSegments.length === 0) {
      setStatus("No segments to export yet.");
      return;
    }

    const base = getRecordingBaseName(context) ?? "recording";
    const filename = safeFilename(`${base}.json`);

    const payload = {
      output_dir: outputDir, // (browser can't write to this path; saved for your python postprocess)
      source_name: base,
      segments: normalizedSegments.map((s) => ({
        startSec: s.startSec,
        endSec: s.endSec,
        prompt: s.prompt,
      })),
    };

    downloadJson(filename, payload);
    setStatus(`Exported ${payload.segments.length} segments to ${filename}`);
  }

  // 4) clear confirm
  function clearAll() {
    const ok = window.confirm("确定要清空所有 segments 吗？此操作不可恢复。");
    if (!ok) return;
    setSegments([]);
    setPendingStartSec(undefined);
    setStatus("Cleared.");
  }

  function removeRow(segId: string) {
    setSegments((prev) => prev.filter((s) => s.id !== segId));
  }

  /** ===== Drag constraints (no overlap, keep start<end) ===== */
  function getNeighbors(segId: string, sorted: SegmentRow[]) {
    const i = sorted.findIndex((s) => s.id === segId);
    return {
      prev: i > 0 ? sorted[i - 1] : undefined,
      next: i >= 0 && i < sorted.length - 1 ? sorted[i + 1] : undefined,
      self: i >= 0 ? sorted[i] : undefined,
    };
  }

  function applyDrag(segId: string, handle: DragHandle, newTime: number) {
    setSegments((prev) => {
      const idx = prev.findIndex((s) => s.id === segId);
      if (idx < 0) return prev;

      const cur0 = prev[idx];
      if (!cur0) return prev; // for noUncheckedIndexedAccess

      const sorted = prev.map(normalizeRow).sort((a, b) => a.startSec - b.startSec);
      const { prev: nPrev, next: nNext } = getNeighbors(segId, sorted);

      const cur = normalizeRow(cur0);

      let minAllowed = -Infinity;
      let maxAllowed = Infinity;
      if (nPrev) minAllowed = Math.max(minAllowed, nPrev.endSec + EPS);
      if (nNext) maxAllowed = Math.min(maxAllowed, nNext.startSec - EPS);

      if (handle === "start") maxAllowed = Math.min(maxAllowed, cur.endSec - EPS);
      else minAllowed = Math.max(minAllowed, cur.startSec + EPS);

      const t = clamp(newTime, minAllowed, maxAllowed);

      const proposedStart = handle === "start" ? t : cur.startSec;
      const proposedEnd = handle === "end" ? t : cur.endSec;

      for (const o of sorted) {
        if (o.id === segId) continue;
        if (overlaps(proposedStart, proposedEnd, o.startSec, o.endSec)) return prev;
      }

      const updated: SegmentRow = handle === "start" ? { ...cur, startSec: t } : { ...cur, endSec: t };

      const nextList = [...prev];
      nextList[idx] = normalizeRow(updated);
      return nextList;
    });
  }

  /** ===== Fixed window range (NO auto scaling) ===== */
  const viewRange = useMemo(() => {
    const half = Math.max(windowSec, 1) / 2;
    const center =
      followCursor && currentTimeSec != null
        ? currentTimeSec
        : windowCenterSec != null
          ? windowCenterSec
          : currentTimeSec != null
            ? currentTimeSec
            : 0;

    return { minT: center - half, maxT: center + half, center };
  }, [windowSec, followCursor, currentTimeSec, windowCenterSec]);

  function timeToX(t: number, width: number, padding: { left: number; right: number }) {
    const plotW = width - padding.left - padding.right;
    const u = (t - viewRange.minT) / Math.max(viewRange.maxT - viewRange.minT, 1e-6);
    return padding.left + u * plotW;
  }

  function xToTime(x: number, width: number, padding: { left: number; right: number }) {
    const plotW = width - padding.left - padding.right;
    const u = (x - padding.left) / plotW;
    return viewRange.minT + u * (viewRange.maxT - viewRange.minT);
  }

  /** ===== Canvas drawing ===== */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const padding = { top: 20, right: 20, bottom: 30, left: 20 };
    const plotX = padding.left;
    const plotY = padding.top;
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // grid (no numbers)
    ctx.strokeStyle = "#eee";
    ctx.lineWidth = 1;
    const gridLines = 10;
    for (let i = 0; i <= gridLines; i++) {
      const x = plotX + (i / gridLines) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, plotY);
      ctx.lineTo(x, plotY + plotH);
      ctx.stroke();
    }

    // fixed track height = 10
    const trackHeight = 10;
    const trackY = plotY + plotH / 2 - trackHeight / 2;
    const cy = trackY + trackHeight / 2;

    ctx.fillStyle = "#fafafa";
    ctx.fillRect(plotX, trackY, plotW, trackHeight);
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, trackY, plotW, trackHeight);

    for (const s of normalizedSegments) {
      const segStart = s.startSec;
      const segEnd = s.endSec;
      if (segEnd < viewRange.minT - 1 || segStart > viewRange.maxT + 1) continue;

      const x1 = timeToX(segStart, width, padding);
      const x2 = timeToX(segEnd, width, padding);

      const isActive = s.id === activeSegId;

      ctx.fillStyle = isActive ? "rgba(156, 39, 176, 0.35)" : "rgba(156, 39, 176, 0.22)";
      ctx.fillRect(Math.min(x1, x2), trackY, Math.abs(x2 - x1), trackHeight);

      ctx.strokeStyle = isActive ? "rgba(156, 39, 176, 0.95)" : "rgba(156, 39, 176, 0.75)";
      ctx.lineWidth = 2;
      ctx.strokeRect(Math.min(x1, x2), trackY, Math.abs(x2 - x1), trackHeight);

      const drawR = 7;

      ctx.fillStyle = "#F44336";
      ctx.beginPath();
      ctx.arc(x1, cy, drawR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#2196F3";
      ctx.beginPath();
      ctx.arc(x2, cy, drawR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // current time line
    if (currentTimeSec != null) {
      const cx = timeToX(currentTimeSec, width, padding);
      ctx.strokeStyle = "#F44336";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, plotY);
      ctx.lineTo(cx, plotY + plotH);
      ctx.stroke();
    }

    // pending start line
    if (pendingStartSec != null) {
      const px = timeToX(pendingStartSec, width, padding);
      ctx.strokeStyle = "#FFC107";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(px, plotY);
      ctx.lineTo(px, plotY + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [normalizedSegments, currentTimeSec, pendingStartSec, viewRange, activeSegId]);

  /** ===== Drag hit test ===== */
  function findHandleHit(mouseX: number, mouseY: number): DragState {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const width = canvas.width;
    const height = canvas.height;

    const padding = { top: 20, right: 20, bottom: 30, left: 20 };
    const plotY = padding.top;
    const plotH = height - padding.top - padding.bottom;

    const trackHeight = 10;
    const trackY = plotY + plotH / 2 - trackHeight / 2;
    const cy = trackY + trackHeight / 2;

    const hitR = 10;

    for (const s of normalizedSegments) {
      const x1 = timeToX(s.startSec, width, { left: padding.left, right: padding.right });
      const x2 = timeToX(s.endSec, width, { left: padding.left, right: padding.right });

      if (Math.hypot(mouseX - x1, mouseY - cy) <= hitR) return { segId: s.id, handle: "start" };
      if (Math.hypot(mouseX - x2, mouseY - cy) <= hitR) return { segId: s.id, handle: "end" };
    }
    return null;
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const ratio = canvas.width / rect.width;

    const x = (e.clientX - rect.left) * ratio;
    const y = (e.clientY - rect.top) * ratio;

    const hit = findHandleHit(x, y);
    if (hit) {
      dragRef.current = hit;
      canvas.setPointerCapture(e.pointerId);
      setStatus(`Dragging ${hit.handle}...`);
    }
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const drag = dragRef.current;
    if (!drag) return;

    const rect = canvas.getBoundingClientRect();
    const ratio = canvas.width / rect.width;
    const x = (e.clientX - rect.left) * ratio;

    const padding = { left: 20, right: 20 };
    const t = xToTime(x, canvas.width, padding);

    applyDrag(drag.segId, drag.handle, t);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (dragRef.current) {
      dragRef.current = null;
      setStatus("Drag done.");
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {}
    }
  }

  // wheel changes windowSec
  function onWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    setWindowSec((prev) => clamp(prev * factor, 2, 600));
  }

  // disable create buttons
  const startDisabled = !canCreateAtCurrent() || pendingStartSec != null;
  const endDisabled = pendingStartSec == null || !canCreateAtCurrent();

  // auto-scroll active row into view
  // useEffect(() => {
  //   if (!activeSegId) return;
  //   const el = rowRefs.current[activeSegId];
  //   if (el) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  // }, [activeSegId, normalizedSegments.length]);
  useEffect(() => {
    const c = context as any;

    console.log("[umi] context keys:", Object.keys(c));

    console.log("[umi] dataSource:", c.dataSource);
    console.log("[umi] playerState:", c.playerState);
    console.log("[umi] playbackState:", c.playbackState);
    console.log("[umi] session:", c.session);
    console.log("[umi] layout:", c.layout);
    console.log("[umi] app:", c.app);

    // 如果有更深层
    console.log("[umi] dataSource keys:", c.dataSource ? Object.keys(c.dataSource) : null);
  }, [context]);


  /** ===== Render ===== */
  const tableMaxHeight = 10 * 34 + 36; // ~ 10 rows

  return (
    <div style={{ padding: 12, fontFamily: "sans-serif", pointerEvents: "auto" }}>
      <h2 style={{ margin: "0 0 8px" }}>Umi Crop</h2>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
        <div style={{ padding: "2px 8px", border: "1px solid #ccc", borderRadius: 6 }}>
          currentTime: <b>{timeText}</b>
        </div>

        <div style={{ padding: "2px 8px", border: "1px solid #ccc", borderRadius: 6 }}>
          window: <b>{windowSec.toFixed(1)} s</b>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 6, color: "#555" }}>
          <input
            type="checkbox"
            checked={followCursor}
            onChange={(e) => {
              const v = e.target.checked;
              setFollowCursor(v);
              if (!v) {
                if (currentTimeSec != null) setWindowCenterSec(currentTimeSec);
                else setWindowCenterSec(viewRange.center);
              } else {
                setWindowCenterSec(undefined);
              }
            }}
          />
          follow cursor
        </label>

        {!followCursor ? (
          <button
            onClick={() => {
              if (currentTimeSec != null) setWindowCenterSec(currentTimeSec);
              setStatus("Window center set to currentTime.");
            }}
          >
            Center to current
          </button>
        ) : null}

        {pendingStartSec != null ? (
          <div style={{ padding: "2px 8px", border: "1px solid #ccc", borderRadius: 6 }}>
            pending start: <b>{pendingStartSec.toFixed(3)} s</b>
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

      {/* 1) Prompt with default */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>Prompt（End 时写入 segment）：</div>
        <input
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          style={{ width: "100%", padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6 }}
        />
      </div>

      {/* 2) Output dir with default */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>输出路径（写入 JSON，供后处理脚本使用）：</div>
        <input
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          style={{ width: "100%", padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6 }}
        />
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
        <button onClick={markStart} disabled={startDisabled} title={isCurrentInsideAnySeg ? "当前时间在 segment 内，禁止重合" : ""}>
          Start @ currentTime
        </button>
        <button onClick={markEnd} disabled={endDisabled} title={isCurrentInsideAnySeg ? "当前时间在 segment 内，禁止重合" : ""}>
          End @ currentTime
        </button>
        <button onClick={exportSegments}>Export JSON</button>
        <button onClick={clearAll} title="需要确认，防止误触">
          Clear
        </button>
      </div>

      {/* Timeline canvas */}
      <div style={{ marginBottom: 14 }}>
        <h3 style={{ margin: "0 0 8px" }}>Timeline (fixed window; wheel changes windowSec)</h3>
        <div style={{ border: "1px solid #ccc", borderRadius: 6, padding: 8, background: "#fff" }}>
          <canvas
            ref={canvasRef}
            width={900}
            height={140}
            style={{ width: "100%", maxWidth: "100%", height: "auto", display: "block", cursor: "ew-resize" }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onWheel={onWheel}
          />
        </div>
        <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
          红点=Start，蓝点=End。滚轮调整 windowSec。导出文件名会尽量与当前 mcap 同名（若数据源不提供文件名则退化为 recording.json）。
        </div>
      </div>

      {/* Table */}
      <h3 style={{ margin: "0 0 8px" }}>Segments ({normalizedSegments.length})</h3>

      <div
        style={{
          maxHeight: tableMaxHeight,
          overflowY: "auto",
          border: "1px solid #eee",
          borderRadius: 6,
        }}
      >
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 740 }}>
          <thead style={{ position: "sticky", top: 0, background: "#fff", zIndex: 1 }}>
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
            {normalizedSegments.length === 0 ? (
              <tr>
                <td style={tdStyle} colSpan={6}>
                  No segments yet. 先点 Start，再点 End。
                </td>
              </tr>
            ) : (
              normalizedSegments.map((s, idx) => {
                const isActive = s.id === activeSegId;
                return (
                  <tr
                    key={s.id}
                    ref={(el) => {
                      rowRefs.current[s.id] = el;
                    }}
                    style={{
                      background: isActive ? "#e9e9e9" : "transparent",
                      transition: "background 120ms",
                    }}
                  >
                    <td style={tdStyle}>{idx}</td>
                    <td style={tdStyle}>{s.startSec.toFixed(3)}</td>
                    <td style={tdStyle}>{s.endSec.toFixed(3)}</td>
                    <td style={tdStyle}>{(s.endSec - s.startSec).toFixed(3)}</td>
                    <td style={{ ...tdStyle, wordBreak: "break-word" }}>{s.prompt}</td>
                    <td style={tdStyle}>
                      <button
                        onClick={() => seekTo(s.startSec)}
                        disabled={!canSeek}
                        style={{ marginRight: 8 }}
                        title={!canSeek ? "Data source does not support seekPlayback" : ""}
                      >
                        Start
                      </button>
                      <button
                        onClick={() => seekTo(s.endSec)}
                        disabled={!canSeek}
                        style={{ marginRight: 8 }}
                        title={!canSeek ? "Data source does not support seekPlayback" : ""}
                      >
                        End
                      </button>
                      <button onClick={() => removeRow(s.id)}>Delete</button>
                    </td>
                  </tr>
                );
              })
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
          <div>activeSegId: {activeSegId ?? "none"}</div>
          <div>windowSec: {windowSec.toFixed(2)}</div>
          <div>followCursor: {String(followCursor)}</div>
          <div>windowCenterSec: {windowCenterSec ?? "undefined"}</div>
          <div>outputDir: {outputDir}</div>
          <div>derived base name: {getRecordingBaseName(context) ?? "(unavailable)"}</div>
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
  context.panelElement.style.pointerEvents = "auto";
  context.panelElement.style.userSelect = "text";

  const root = createRoot(context.panelElement);
  root.render(<UmiCropPanel context={context} />);

  return () => root.unmount();
}
