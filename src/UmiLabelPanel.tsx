import { Immutable, MessageEvent, PanelExtensionContext, Time, Topic } from "@lichtblick/suite";
import React, { ReactElement, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

// task config
import taskConfigJson from "./task_config.json";

/** ===== Types ===== */
type SubtaskDef = { id: string; name?: string; prompt: string };
type TaskDef = { id: string; name: string; prompt: string; subtasks?: SubtaskDef[] };
type TaskConfig = { version: number; tasks: TaskDef[] };

type SubSegment = { subtaskId: string; name?: string; startSec: number; endSec: number; prompt: string };

type Segment = {
  startSec: number;
  endSec: number;
  prompt: string;

  taskId?: string;

  // internal cut points (length = numSubtasks-1), draggable
  cutsSec?: number[];
};

type SegmentRow = Segment & { id: string };

type PanelState = {
  segments: Segment[];
  pendingStartSec?: number;
  pendingCutsSec?: number[]; // NEW: next_cut workflow
  promptText?: string;

  windowSec?: number;
  followCursor?: boolean;
  windowCenterSec?: number;

  outputDir?: string;
  selectedTaskId?: string;
};

type DragState =
  | { segId: string; kind: "start" }
  | { segId: string; kind: "end" }
  | { segId: string; kind: "cut"; cutIndex: number }
  | null;

const EPS = 1e-3;

/** ===== Utils ===== */
function toSec(t: Time | undefined): number | undefined {
  if (!t) return undefined;
  if (typeof t.sec === "number" && typeof t.nsec === "number") return t.sec + t.nsec * 1e-9;
  return undefined;
}
function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}
function overlaps(aStart: number, aEnd: number, bStart: number, bEnd: number) {
  return aStart < bEnd - EPS && aEnd > bStart + EPS;
}
function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
function normalizeSegment(s: Segment): Segment {
  const start = Math.min(s.startSec, s.endSec);
  const end = Math.max(s.startSec, s.endSec);
  const cuts = (s.cutsSec ?? []).slice().sort((a, b) => a - b);
  return { ...s, startSec: start, endSec: end, cutsSec: cuts };
}
function normalizeRow(r: SegmentRow): SegmentRow {
  const s = normalizeSegment(r);
  return { ...s, id: r.id };
}

function getRecordingBaseName(context: PanelExtensionContext): string | undefined {
  const anyCtx = context as unknown as Record<string, unknown>;
  const pickString = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);

  const dataSource = anyCtx["dataSource"] as Record<string, unknown> | undefined;
  const dsName =
    pickString(dataSource?.["name"]) ||
    pickString(dataSource?.["fileName"]) ||
    pickString(dataSource?.["filename"]) ||
    pickString(dataSource?.["id"]);

  const playerState = anyCtx["playerState"] as Record<string, unknown> | undefined;
  const psName =
    pickString(playerState?.["name"]) ||
    pickString(playerState?.["fileName"]) ||
    pickString(playerState?.["filename"]) ||
    pickString(playerState?.["id"]);

  const title = pickString(anyCtx["title"]);

  const raw = dsName || psName || title;
  if (!raw) return undefined;

  const last = raw.split(/[\\/]/).pop() ?? raw;
  const noExt = last.replace(/\.(mcap|bag|db3|json|log)$/i, "");
  return noExt || undefined;
}

function safeFilename(s: string) {
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

type ImportedSubtask = {
  subtaskId: string;
  name?: string;
  startSec: number;
  endSec: number;
  prompt: string;
};

type ImportedSegment = {
  startSec: number;
  endSec: number;
  prompt: string;
  taskId?: string;
  subtasks?: ImportedSubtask[];
  // 兼容：如果你未来导出里也直接带 cutsSec，也能读
  cutsSec?: number[];
};

type ImportedPayload = {
  output_dir?: string;
  source_name?: string;
  task_config_version?: number;
  segments?: ImportedSegment[];
};

function cutsFromImportedSegment(seg: ImportedSegment): number[] {
  // 优先用 cutsSec（如果存在）
  if (Array.isArray(seg.cutsSec) && seg.cutsSec.length) {
    return seg.cutsSec.slice().sort((a, b) => a - b);
  }

  // 否则用 subtasks 的边界反推 cuts
  const sts = seg.subtasks ?? [];
  if (sts.length >= 2) {
    // cuts = 每个 subtask 的 end（除最后一个）
    const cuts = sts.slice(0, -1).map((st) => st.endSec);
    return cuts
      .map((c) => clamp(c, seg.startSec + EPS, seg.endSec - EPS))
      .sort((a, b) => a - b);
  }

  return [];
}


/** equal init cuts (nSubtasks => nSubtasks-1 cuts) */
function initCuts(startSec: number, endSec: number, nSubtasks: number): number[] {
  const nCuts = Math.max(nSubtasks - 1, 0);
  if (nCuts <= 0) return [];
  const dur = Math.max(endSec - startSec, 0);
  if (dur < EPS) return [];
  const step = dur / nSubtasks;
  const cuts: number[] = [];
  for (let i = 1; i <= nCuts; i++) cuts.push(startSec + step * i);
  return cuts;
}

/** derive subtasks from cuts and task defs */
function buildSubtasksFromCuts(seg: SegmentRow, task?: TaskDef): SubSegment[] {
  const defs = task?.subtasks ?? [];
  if (!defs.length) return [];

  const start = seg.startSec;
  const end = seg.endSec;
  const cuts = (seg.cutsSec ?? []).slice().sort((a, b) => a - b);

  if (cuts.length !== defs.length - 1) {
    const fixed = initCuts(start, end, defs.length);
    cuts.splice(0, cuts.length, ...fixed);
  }

  const bounds = [start, ...cuts, end];
  const out: SubSegment[] = [];
  for (let i = 0; i < defs.length; i++) {
    out.push({
      subtaskId: defs[i]!.id,
      name: defs[i]!.name,
      startSec: bounds[i]!,
      endSec: bounds[i + 1]!,
      prompt: defs[i]!.prompt,
    });
  }
  return out;
}

/** colors for cut points */
const CUT_COLORS = ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#3498db", "#e74c3c"];

/** ===== Panel ===== */
function UmiCropPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const initial = (context.initialState as Partial<PanelState> | undefined) ?? {};

  const taskConfig = taskConfigJson as unknown as TaskConfig;
  const defaultTaskId = taskConfig.tasks[0]?.id ?? "";

  const [topics, setTopics] = useState<undefined | Immutable<Topic[]>>();
  const [messages, setMessages] = useState<undefined | Immutable<MessageEvent[]>>();

  const [currentTimeSec, setCurrentTimeSec] = useState<number | undefined>(undefined);

  // pending (next_cut workflow)
  const [pendingStartSec, setPendingStartSec] = useState<number | undefined>(initial.pendingStartSec);
  const [pendingCutsSec, setPendingCutsSec] = useState<number[]>(initial.pendingCutsSec ?? []);

  const [segments, setSegments] = useState<SegmentRow[]>(() => {
    const base = (initial.segments ?? []).map(normalizeSegment);
    return base.map((s) => ({ ...s, id: newId() }));
  });

  const [selectedTaskId, setSelectedTaskId] = useState<string>(initial.selectedTaskId ?? defaultTaskId);
  const selectedTask = useMemo(() => {
    return taskConfig.tasks.find((t) => t.id === selectedTaskId) ?? taskConfig.tasks[0];
  }, [taskConfig.tasks, selectedTaskId]);

  const [promptText, setPromptText] = useState<string>(
    initial.promptText ?? selectedTask?.prompt ?? "pick the bottle and put into the box",
  );

  const [status, setStatus] = useState<string>("");
  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();

  const [windowSec, setWindowSec] = useState<number>(initial.windowSec ?? 30);
  const [followCursor, setFollowCursor] = useState<boolean>(initial.followCursor ?? true);
  const [windowCenterSec, setWindowCenterSec] = useState<number | undefined>(initial.windowCenterSec);

  const [outputDir, setOutputDir] = useState<string>(initial.outputDir ?? "/data/label_data/seg");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<DragState>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);


  // update prompt on task change (optional)
  useEffect(() => {
    if (selectedTask?.prompt) setPromptText(selectedTask.prompt);
  }, [selectedTaskId]); // eslint-disable-line react-hooks/exhaustive-deps

  // persist
  useEffect(() => {
    const plain: Segment[] = segments.map((r) => {
      const n = normalizeRow(r);
      return { startSec: n.startSec, endSec: n.endSec, prompt: n.prompt, taskId: n.taskId, cutsSec: n.cutsSec };
    });

    context.saveState({
      segments: plain,
      pendingStartSec,
      pendingCutsSec,
      promptText,
      windowSec,
      followCursor,
      windowCenterSec,
      outputDir,
      selectedTaskId,
    } satisfies PanelState);
  }, [context, segments, pendingStartSec, pendingCutsSec, promptText, windowSec, followCursor, windowCenterSec, outputDir, selectedTaskId]);

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

  const timeText = useMemo(() => (currentTimeSec == null ? "n/a" : `${currentTimeSec.toFixed(3)} s`), [currentTimeSec]);

  const normalizedSegments = useMemo(() => segments.map(normalizeRow).sort((a, b) => a.startSec - b.startSec), [segments]);

  const activeSegId = useMemo(() => {
    if (currentTimeSec == null) return undefined;
    for (const r of normalizedSegments) {
      if (currentTimeSec >= r.startSec - EPS && currentTimeSec <= r.endSec + EPS) return r.id;
    }
    return undefined;
  }, [currentTimeSec, normalizedSegments]);

  const isCurrentInsideAnySeg = activeSegId != null;

  function canCreateAtCurrent() {
    return currentTimeSec != null && !isCurrentInsideAnySeg;
  }

  /** ===== View range ===== */
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

  /** ===== Buttons ===== */
  function markStart() {
    if (!canCreateAtCurrent()) {
      setStatus(currentTimeSec == null ? "currentTimeSec is undefined." : "当前时间在已有 segment 内，禁止重合。");
      return;
    }
    setPendingStartSec(currentTimeSec);
    setPendingCutsSec([]);
    setStatus(`Start set at ${currentTimeSec!.toFixed(3)} s`);
  }

  /**
   * End button behavior (NEW):
   * - If pendingStartSec exists and pendingCutsSec not complete:
   *   end now, and auto-fill remaining cuts by equal split between (lastPoint, end)
   */
  function markEnd() {
    if (pendingStartSec == null) {
      setStatus("请先点击 Start（或用 Next cut 设定 Start）。");
      return;
    }
    if (!canCreateAtCurrent()) {
      setStatus("当前时间在已有 segment 内，禁止重合。");
      return;
    }
    if (currentTimeSec == null) {
      setStatus("currentTimeSec is undefined.");
      return;
    }

    const task = selectedTask;
    const nSub = task?.subtasks?.length ?? 0;
    const needCuts = Math.max(nSub - 1, 0);

    // enforce ordering: start < cuts... < end
    const start = pendingStartSec;
    const end = currentTimeSec;

    const lastPoint = pendingCutsSec.length > 0 ? pendingCutsSec[pendingCutsSec.length - 1]! : start;
    if (end <= lastPoint + EPS) {
      setStatus(`End 必须大于上一个点：end=${end.toFixed(3)} <= last=${lastPoint.toFixed(3)}`);
      return;
    }

    // overlap check with existing segments
    for (const o of normalizedSegments) {
      if (overlaps(start, end, o.startSec, o.endSec)) {
        setStatus(`与已有 segment 重叠，创建失败（${o.startSec.toFixed(3)}→${o.endSec.toFixed(3)}）`);
        return;
      }
    }

    // build cuts: existing pending + auto-fill remaining (if any)
    const cuts = pendingCutsSec.slice(); // already increasing by Next cut rules
    const remaining = Math.max(needCuts - cuts.length, 0);
    if (remaining > 0) {
      const intervalStart = lastPoint;
      const intervalEnd = end;
      const step = (intervalEnd - intervalStart) / (remaining + 1);
      for (let i = 1; i <= remaining; i++) cuts.push(intervalStart + step * i);
    }

    // clamp and sort final cuts
    const finalCuts = cuts
      .map((c) => clamp(c, start + EPS, end - EPS))
      .slice()
      .sort((a, b) => a - b);

    const seg: SegmentRow = {
      id: newId(),
      startSec: start,
      endSec: end,
      prompt: (task?.prompt ?? promptText ?? "").trim(),
      taskId: task?.id,
      cutsSec: finalCuts,
    };

    setSegments((prev) => [...prev, seg]);
    setPendingStartSec(undefined);
    setPendingCutsSec([]);
    setStatus(`Created segment: ${start.toFixed(3)} → ${end.toFixed(3)} (cuts=${finalCuts.length}/${needCuts})`);
  }

  /**
   * Next cut: sequentially set start -> cut1 -> cut2 -> ... -> end(create)
   * If user clicks End before finishing cuts, markEnd() will auto-fill the rest.
   */
  function nextCut() {
    if (currentTimeSec == null) {
      setStatus("currentTimeSec is undefined.");
      return;
    }
    if (isCurrentInsideAnySeg) {
      setStatus("当前时间在已有 segment 内，禁止重合。");
      return;
    }

    const task = selectedTask;
    const nSub = task?.subtasks?.length ?? 0;
    const needCuts = Math.max(nSub - 1, 0);

    // if no start yet: set start
    if (pendingStartSec == null) {
      setPendingStartSec(currentTimeSec);
      setPendingCutsSec([]);
      setStatus(`Start set at ${currentTimeSec.toFixed(3)} s (NextCut mode)`);
      return;
    }

    // enforce increasing order
    const lastPoint = pendingCutsSec.length > 0 ? pendingCutsSec[pendingCutsSec.length - 1]! : pendingStartSec;
    if (currentTimeSec <= lastPoint + EPS) {
      setStatus(`需要按顺序递增标注：当前 ${currentTimeSec.toFixed(3)} <= 上一个点 ${lastPoint.toFixed(3)}`);
      return;
    }

    // if still need cuts: add a cut
    if (pendingCutsSec.length < needCuts) {
      const k = pendingCutsSec.length + 1;
      setPendingCutsSec((prev) => [...prev, currentTimeSec]);
      setStatus(`Cut ${k}/${needCuts} set at ${currentTimeSec.toFixed(3)} s`);
      return;
    }

    // cuts done; this click becomes end => create segment
    // (re-use markEnd logic by temporarily treating currentTimeSec as end)
    markEnd();
  }

  function exportSegments() {
    if (normalizedSegments.length === 0) {
      setStatus("No segments to export yet.");
      return;
    }

    const base = getRecordingBaseName(context) ?? "recording";
    const filename = safeFilename(`${base}.json`);

    const payload = {
      output_dir: outputDir,
      source_name: base,
      task_config_version: taskConfig.version,
      segments: normalizedSegments.map((s) => {
        const task = taskConfig.tasks.find((t) => t.id === s.taskId);
        const subtasks = buildSubtasksFromCuts(s, task);
        return {
          startSec: s.startSec,
          endSec: s.endSec,
          prompt: s.prompt,
          taskId: s.taskId,
          subtasks: subtasks.map((st) => ({
            subtaskId: st.subtaskId,
            name: st.name,
            startSec: st.startSec,
            endSec: st.endSec,
            prompt: st.prompt,
          })),
        };
      }),
    };

    downloadJson(filename, payload);
    setStatus(`Exported ${payload.segments.length} segments to ${filename}`);
  }

  function clearAll() {
    const ok = window.confirm("确定要清空所有 segments 吗？此操作不可恢复。");
    if (!ok) return;
    setSegments([]);
    setPendingStartSec(undefined);
    setPendingCutsSec([]);
    setStatus("Cleared.");
  }

  function removeRow(segId: string) {
    setSegments((prev) => prev.filter((s) => s.id !== segId));
  }
  // for import json
  function openImportDialog() {
    fileInputRef.current?.click();
  }

  async function importSegmentsFromFile(file: File) {
    try {
      const text = await file.text();
      const json = JSON.parse(text) as ImportedPayload;

      const list = json.segments ?? [];
      if (!Array.isArray(list) || list.length === 0) {
        setStatus("导入失败：JSON 中没有 segments。");
        return;
      }

      const rows: SegmentRow[] = list.map((s) => {
        const startSec = Number(s.startSec);
        const endSec = Number(s.endSec);
        const prompt = (s.prompt ?? "").toString();

        const cuts = cutsFromImportedSegment(s);

        const row: SegmentRow = normalizeRow({
          id: newId(),
          startSec,
          endSec,
          prompt,
          taskId: s.taskId,
          cutsSec: cuts,
        });

        return row;
      });

      // 排序 + 基本重叠检查（可选：如果你希望强制无重叠）
      const sorted = rows.sort((a, b) => a.startSec - b.startSec);

      // 你也可以选择保留现有 segments 并 append；这里默认“覆盖加载”
      setSegments(sorted);
      setPendingStartSec(undefined);
      setPendingCutsSec([]);

      // 可选：同步 outputDir
      if (typeof json.output_dir === "string" && json.output_dir.trim()) {
        setOutputDir(json.output_dir.trim());
      }

      // 可选：把 task 下拉切到第一个 segment 的 taskId（如果存在且在 config 里）
      const firstTaskId = sorted[0]?.taskId;
      if (firstTaskId && taskConfig.tasks.some((t) => t.id === firstTaskId)) {
        setSelectedTaskId(firstTaskId);
      }

      setStatus(`已导入 ${sorted.length} 个 segments：${file.name}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`导入失败：${msg}`);
    }
  }



  /** ===== Drag constraints ===== */
  function getNeighbors(segId: string, sorted: SegmentRow[]) {
    const i = sorted.findIndex((s) => s.id === segId);
    return {
      prev: i > 0 ? sorted[i - 1] : undefined,
      next: i >= 0 && i < sorted.length - 1 ? sorted[i + 1] : undefined,
      self: i >= 0 ? sorted[i] : undefined,
    };
  }


  function applyDragBoundary(segId: string, kind: "start" | "end", newTime: number) {
    setSegments((prev) => {
      const idx = prev.findIndex((s) => s.id === segId);
      if (idx < 0) return prev;

      const cur0 = prev[idx];
      if (!cur0) return prev;

      const sorted = prev.map(normalizeRow).sort((a, b) => a.startSec - b.startSec);
      const { prev: nPrev, next: nNext } = getNeighbors(segId, sorted);

      const cur = normalizeRow(cur0);

      let minAllowed = -Infinity;
      let maxAllowed = Infinity;
      if (nPrev) minAllowed = Math.max(minAllowed, nPrev.endSec + EPS);
      if (nNext) maxAllowed = Math.min(maxAllowed, nNext.startSec - EPS);

      if (kind === "start") maxAllowed = Math.min(maxAllowed, cur.endSec - EPS);
      else minAllowed = Math.max(minAllowed, cur.startSec + EPS);

      const t = clamp(newTime, minAllowed, maxAllowed);

      // 直接更新 startSec 或 endSec，而不需要创建 proposedStart 或 proposedEnd
      const updatedBase: SegmentRow = kind === "start" ? { ...cur, startSec: t } : { ...cur, endSec: t };
      const updatedNorm = normalizeRow(updatedBase);

      // 只更新边界，不修改分割点
      const nextList = [...prev];
      nextList[idx] = { ...updatedNorm };
      return nextList;
    });
  }



  function applyDragCut(segId: string, cutIndex: number, newTime: number) {
    setSegments((prev) => {
      const idx = prev.findIndex((s) => s.id === segId);
      if (idx < 0) return prev;

      const cur0 = prev[idx];
      if (!cur0) return prev;

      const cur = normalizeRow(cur0);
      const cuts = (cur.cutsSec ?? []).slice().sort((a, b) => a - b);

      if (cutIndex < 0 || cutIndex >= cuts.length) return prev;

      const lo = cutIndex === 0 ? cur.startSec + EPS : cuts[cutIndex - 1]! + EPS;
      const hi = cutIndex === cuts.length - 1 ? cur.endSec - EPS : cuts[cutIndex + 1]! - EPS;

      const t = clamp(newTime, lo, hi);
      cuts[cutIndex] = t;

      const nextList = [...prev];
      nextList[idx] = { ...cur, cutsSec: cuts.slice().sort((a, b) => a - b) };
      return nextList;
    });
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

    const trackHeight = 10;
    const trackY = plotY + plotH / 2 - trackHeight / 2;
    const cy = trackY + trackHeight / 2;

    ctx.fillStyle = "#fafafa";
    ctx.fillRect(plotX, trackY, plotW, trackHeight);
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, trackY, plotW, trackHeight);

    for (const s of normalizedSegments) {
      if (s.endSec < viewRange.minT - 1 || s.startSec > viewRange.maxT + 1) continue;

      const x1 = timeToX(s.startSec, width, padding);
      const x2 = timeToX(s.endSec, width, padding);
      const isActive = s.id === activeSegId;

      ctx.fillStyle = isActive ? "rgba(156, 39, 176, 0.35)" : "rgba(156, 39, 176, 0.22)";
      ctx.fillRect(Math.min(x1, x2), trackY, Math.abs(x2 - x1), trackHeight);

      ctx.strokeStyle = isActive ? "rgba(156, 39, 176, 0.95)" : "rgba(156, 39, 176, 0.75)";
      ctx.lineWidth = 2;
      ctx.strokeRect(Math.min(x1, x2), trackY, Math.abs(x2 - x1), trackHeight);

      // boundary points
      const drawR = 7;

      // Start red
      ctx.fillStyle = "#F44336";
      ctx.beginPath();
      ctx.arc(x1, cy, drawR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // End blue
      ctx.fillStyle = "#2196F3";
      ctx.beginPath();
      ctx.arc(x2, cy, drawR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // cut points colored
      const cuts = (s.cutsSec ?? []).slice().sort((a, b) => a - b);
      for (let i = 0; i < cuts.length; i++) {
        const t = cuts[i]!;
        if (t < viewRange.minT - 1 || t > viewRange.maxT + 1) continue;
        const cxp = timeToX(t, width, padding);

        const color = CUT_COLORS[i % CUT_COLORS.length] ?? "#2ecc71";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cxp, cy, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
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

    // pending cuts (preview) as small dots
    if (pendingStartSec != null && pendingCutsSec.length > 0) {
      const trackDotY = cy;
      for (let i = 0; i < pendingCutsSec.length; i++) {
        const t = pendingCutsSec[i]!;
        if (t < viewRange.minT - 1 || t > viewRange.maxT + 1) continue;
        const cxp = timeToX(t, width, padding);
        const color = CUT_COLORS[i % CUT_COLORS.length] ?? "#2ecc71";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cxp, trackDotY, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [normalizedSegments, currentTimeSec, pendingStartSec, pendingCutsSec, viewRange, activeSegId]);

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

    const hitR = 11;

    for (const s of normalizedSegments) {
      const x1 = timeToX(s.startSec, width, { left: padding.left, right: padding.right });
      const x2 = timeToX(s.endSec, width, { left: padding.left, right: padding.right });

      if (Math.hypot(mouseX - x1, mouseY - cy) <= hitR) return { segId: s.id, kind: "start" };
      if (Math.hypot(mouseX - x2, mouseY - cy) <= hitR) return { segId: s.id, kind: "end" };

      const cuts = (s.cutsSec ?? []).slice().sort((a, b) => a - b);
      for (let i = 0; i < cuts.length; i++) {
        const cxp = timeToX(cuts[i]!, width, { left: padding.left, right: padding.right });
        if (Math.hypot(mouseX - cxp, mouseY - cy) <= hitR) return { segId: s.id, kind: "cut", cutIndex: i };
      }
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
      setStatus(hit.kind === "cut" ? `Dragging cut[${hit.cutIndex}]...` : `Dragging ${hit.kind}...`);
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

    if (drag.kind === "start" || drag.kind === "end") {
      applyDragBoundary(drag.segId, drag.kind, t);
    } else {
      applyDragCut(drag.segId, drag.cutIndex, t);
    }
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

  function onWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    setWindowSec((prev) => clamp(prev * factor, 2, 600));
  }

  // import json
  function onImportFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    void importSegmentsFromFile(file);

    // 关键：清空 input value，否则同一个文件第二次选不会触发 change
    e.target.value = "";
  }


  /** ===== UI states ===== */
  const startDisabled = !canCreateAtCurrent() || pendingStartSec != null;
  const endDisabled = pendingStartSec == null || !canCreateAtCurrent();
  const nextCutDisabled = !canCreateAtCurrent();

  const needCutsForTask = Math.max((selectedTask?.subtasks?.length ?? 0) - 1, 0);

  /** ===== Render ===== */
  const tableMaxHeight = 10 * 34 + 36;

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
            pending start: <b>{pendingStartSec.toFixed(3)} s</b> | cuts:{" "}
            <b>{pendingCutsSec.length}</b>/<b>{needCutsForTask}</b>
          </div>
        ) : (
          <div style={{ color: "#666" }}>在时间轴定位到某个时刻，然后点 Start（或直接点 Next cut）</div>
        )}
      </div>

      {status ? (
        <div style={{ margin: "8px 0", padding: "6px 8px", border: "1px solid #ddd", borderRadius: 6, color: "#333" }}>
          {status}
        </div>
      ) : null}

      {/* Task selector */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>Task（从 task_config.json 加载）：</div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <select
            value={selectedTaskId}
            onChange={(e) => {
              const id = e.target.value;
              setSelectedTaskId(id);
              const t = taskConfig.tasks.find((x) => x.id === id);
              if (t?.prompt) setPromptText(t.prompt);
              setStatus(`Selected task: ${t?.name ?? id}`);
            }}
            style={{ padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6, minWidth: 320 }}
          >
            {taskConfig.tasks.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>

          <div style={{ color: "#666" }}>
            subtasks: <b>{selectedTask?.subtasks?.length ?? 0}</b>
          </div>
          <div style={{ color: "#888", fontSize: 12 }}>
            Next cut：按顺序点 Start→Cut…→End；也可中途直接点 End 自动补齐剩余 cut（均分 last→end）
          </div>
        </div>

        {selectedTask?.subtasks?.length ? (
          <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
            {selectedTask.subtasks.map((s, i) => (
              <div key={s.id}>
                {i + 1}. <b>{s.name ?? s.id}</b> — {s.prompt}
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {/* Prompt */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>Prompt（写入 segment）：</div>
        <input
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          style={{ width: "100%", padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6 }}
        />
      </div>

      {/* Output dir */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 6, color: "#555" }}>输出路径（写入 JSON，供后处理脚本使用）：</div>
        <input
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          style={{ width: "100%", padding: "6px 8px", border: "1px solid #ccc", borderRadius: 6 }}
        />
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <button onClick={markStart} disabled={startDisabled} title={isCurrentInsideAnySeg ? "当前时间在 segment 内，禁止重合" : ""}>
          Start @ currentTime
        </button>
        <button onClick={nextCut} disabled={nextCutDisabled} title={isCurrentInsideAnySeg ? "当前时间在 segment 内，禁止重合" : ""}>
          Next cut
        </button>
        <button onClick={markEnd} disabled={endDisabled} title={isCurrentInsideAnySeg ? "当前时间在 segment 内，禁止重合" : ""}>
          End @ currentTime
        </button>
        <button onClick={exportSegments}>Export JSON</button>
        <button onClick={openImportDialog}>Import JSON</button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          style={{ display: "none" }}
          onChange={onImportFileChange}
        />
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
          红点=Start，蓝点=End，彩色点=Subtask 分割点（可拖动）。Next cut 期间，pending cut 也会以小点预览。
        </div>
      </div>

      {/* Table */}
      <h3 style={{ margin: "0 0 8px" }}>Segments ({normalizedSegments.length})</h3>

      {/* <div style={{ maxHeight: tableMaxHeight, overflowY: "auto", border: "1px solid #eee", borderRadius: 6 }}> */}
      <div
        style={{
          maxHeight: tableMaxHeight,
          overflowY: "auto",
          overflowX: "auto", // NEW: 允许横向滚动
          border: "1px solid #eee",
          borderRadius: 6,
          background: "#fff",
        }}
      >
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 980 }}>
          <thead style={{ position: "sticky", top: 0, background: "#fff", zIndex: 1 }}>
            <tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Start (s)</th>
              <th style={thStyle}>End (s)</th>
              <th style={thStyle}>Duration (s)</th>
              <th style={thStyle}>Task</th>
              <th style={thStyle}>Prompt</th>
              <th style={thStyle}>Cuts</th>
              <th style={thStyle}>Subtasks (derived)</th>
              {/* <th style={thStyle}>Actions</th> */}
              <th style={{ ...thStyle, ...stickyRightTh }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {normalizedSegments.length === 0 ? (
              <tr>
                <td style={tdStyle} colSpan={9}>
                  No segments yet. 先点 Start，再点 End；或使用 Next cut。
                </td>
              </tr>
            ) : (
              normalizedSegments.map((s, idx) => {
                const task = taskConfig.tasks.find((t) => t.id === s.taskId);
                const taskName = task?.name ?? s.taskId ?? "-";
                const subtasks = buildSubtasksFromCuts(s, task);

                return (
                  <tr key={s.id}>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>{idx}</td>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>{s.startSec.toFixed(3)}</td>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>{s.endSec.toFixed(3)}</td>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>{(s.endSec - s.startSec).toFixed(3)}</td>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>{taskName}</td>
                    <td style={{ ...tdStyle, wordBreak: "break-word" }}>{s.prompt}</td>
                    <td style={{ ...tdStyle, fontSize: 12, color: "#555" }}>
                      {(s.cutsSec ?? []).length ? (s.cutsSec ?? []).map((c) => c.toFixed(3)).join(", ") : "-"}
                    </td>
                    <td style={{ ...tdStyle, fontSize: 12, color: "#555" }}>
                      {subtasks.length ? (
                        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                          {subtasks.map((st, i) => (
                            <div key={st.subtaskId}>
                              <span
                                style={{
                                  display: "inline-block",
                                  width: 10,
                                  height: 10,
                                  borderRadius: 999,
                                  background: (CUT_COLORS[i % CUT_COLORS.length] ?? "#2ecc71"),
                                  marginRight: 6,
                                  verticalAlign: "middle",
                                }}
                              />
                              <b>{st.name ?? st.subtaskId}</b>: {st.startSec.toFixed(3)} → {st.endSec.toFixed(3)} — {st.prompt}
                            </div>
                          ))}
                        </div>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td style={{ ...tdStyle, ...stickyRightTd, whiteSpace: "nowrap" }}>
                        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <button onClick={() => seekTo(s.startSec)} disabled={!canSeek}>Start</button>
                          <button onClick={() => seekTo(s.endSec)} disabled={!canSeek}>End</button>
                          <button onClick={() => removeRow(s.id)}>Delete</button>
                        </div>
                      {/* <button onClick={() => seekTo(s.startSec)} disabled={!canSeek} style={{ marginRight: 8 }}>
                        Start
                      </button>
                      <button onClick={() => seekTo(s.endSec)} disabled={!canSeek} style={{ marginRight: 8 }}>
                        End
                      </button>
                      <button onClick={() => removeRow(s.id)}>Delete</button> */}
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
          <div>pendingCutsSec: {pendingCutsSec.map((x) => x.toFixed(3)).join(", ") || "[]"}</div>
          <div>segments count: {segments.length}</div>
          <div>activeSegId: {activeSegId ?? "none"}</div>
          <div>windowSec: {windowSec.toFixed(2)}</div>
          <div>followCursor: {String(followCursor)}</div>
          <div>windowCenterSec: {windowCenterSec ?? "undefined"}</div>
          <div>outputDir: {outputDir}</div>
          <div>selectedTaskId: {selectedTaskId}</div>
          <div>derived base name: {getRecordingBaseName(context) ?? "(unavailable)"}</div>
          <div>task_config_version: {taskConfig.version}</div>
          <div>tasks count: {taskConfig.tasks.length}</div>
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
// 避免按钮被遮挡
const stickyRightTh: React.CSSProperties = {
  position: "sticky",
  right: 0,
  zIndex: 3,
  background: "#f6f6f6",
  boxShadow: "-8px 0 8px rgba(0,0,0,0.03)",
};

const stickyRightTd: React.CSSProperties = {
  position: "sticky",
  right: 0,
  zIndex: 2,
  background: "#fff",
  boxShadow: "-8px 0 8px rgba(0,0,0,0.03)",
};


export function initUmiLabelPanel(context: PanelExtensionContext): () => void {
  context.panelElement.style.pointerEvents = "auto";
  context.panelElement.style.userSelect = "text";

  const root = createRoot(context.panelElement);
  root.render(<UmiCropPanel context={context} />);

  return () => root.unmount();
}

