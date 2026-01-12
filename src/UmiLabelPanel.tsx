import { Immutable, MessageEvent, PanelExtensionContext, Time, Topic } from "@lichtblick/suite";
import React, { ReactElement, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

type Segment = { startSec: number; endSec: number; prompt: string };
type PanelState = { segments: Segment[]; pendingStartSec?: number; promptText?: string; selectedPoseTopic?: string };
type PoseDataPoint = { timeSec: number; x?: number; y?: number; z?: number; theta?: number };

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
  const [promptText, setPromptText] = useState<string>(initial.promptText ?? "pick the bottle and put into the box");

  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();
  const [status, setStatus] = useState<string>("");

  // 位姿数据相关
  const [poseData, setPoseData] = useState<PoseDataPoint[]>([]);
  const [selectedPoseTopic, setSelectedPoseTopic] = useState<string | undefined>(initial.selectedPoseTopic);
  const poseDataRef = useRef<PoseDataPoint[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // persist to layout
  useEffect(() => {
    context.saveState({ segments, pendingStartSec, promptText, selectedPoseTopic } satisfies PanelState);
  }, [context, segments, pendingStartSec, promptText, selectedPoseTopic]);

  useLayoutEffect(() => {
    context.onRender = (renderState, done) => {
      setRenderDone(() => done);
      setTopics(renderState.topics);
      setMessages(renderState.currentFrame);
      setCurrentTimeSec(toSec(renderState.currentTime));

      // 收集位姿数据
      if (selectedPoseTopic && renderState.currentFrame) {
        const newPosePoints: PoseDataPoint[] = [];
        for (const msg of renderState.currentFrame) {
          if (msg.topic === selectedPoseTopic) {
            const timeSec = toSec(msg.receiveTime);
            if (timeSec != null) {
              const pose = extractPoseFromMessage(msg.message);
              if (pose) {
                newPosePoints.push({ timeSec, ...pose });
              }
            }
          }
        }
        if (newPosePoints.length > 0) {
          poseDataRef.current = [...poseDataRef.current, ...newPosePoints];
          // 保持数据量在合理范围内（最多保留10000个点）
          if (poseDataRef.current.length > 10000) {
            poseDataRef.current = poseDataRef.current.slice(-10000);
          }
          setPoseData([...poseDataRef.current]);
        }
      }
    };

    context.watch("topics");
    context.watch("currentFrame");
    context.watch("currentTime");
  }, [context, selectedPoseTopic]);

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

  // 从消息中提取位姿信息
  function extractPoseFromMessage(msg: unknown): { x?: number; y?: number; z?: number; theta?: number } | null {
    try {
      const m = msg as any;
      // 尝试多种常见的位姿消息格式
      if (m?.pose?.position) {
        return {
          x: m.pose.position.x,
          y: m.pose.position.y,
          z: m.pose.position.z,
          theta: m.pose.orientation ? quaternionToYaw(m.pose.orientation) : undefined,
        };
      }
      if (m?.position) {
        return {
          x: m.position.x,
          y: m.position.y,
          z: m.position.z,
          theta: m.orientation ? quaternionToYaw(m.orientation) : undefined,
        };
      }
      if (m?.x != null || m?.y != null) {
        return {
          x: m.x,
          y: m.y,
          z: m.z,
          theta: m.theta ?? m.yaw,
        };
      }
    } catch (e) {
      console.warn("[umi] Failed to extract pose:", e);
    }
    return null;
  }

  // 四元数转yaw角
  function quaternionToYaw(q: { x?: number; y?: number; z?: number; w?: number }): number | undefined {
    if (q.w == null || q.z == null) return undefined;
    return Math.atan2(2 * (q.w * q.z + (q.x ?? 0) * (q.y ?? 0)), 1 - 2 * ((q.y ?? 0) * (q.y ?? 0) + (q.z ?? 0) * (q.z ?? 0)));
  }

  // 绘制时间轴可视化（类似plot的样式）
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // 设置绘图区域边距
    const padding = { top: 30, right: 20, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const plotX = padding.left;
    const plotY = padding.top;

    if (poseData.length === 0 && segments.length === 0) {
      ctx.fillStyle = "#999";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("暂无数据，请选择位姿topic并播放数据", width / 2, height / 2);
      return;
    }

    // 计算时间范围
    const allTimes = [
      ...poseData.map((p) => p.timeSec),
      ...segments.flatMap((s) => [s.startSec, s.endSec]),
      currentTimeSec ?? 0,
      pendingStartSec ?? 0,
    ].filter((t) => t > 0);
    if (allTimes.length === 0) return;

    const minTime = Math.min(...allTimes);
    const maxTime = Math.max(...allTimes);
    const timeRange = Math.max(maxTime - minTime, 1);

    // 绘制背景和边框
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(plotX, plotY, plotWidth, plotHeight);
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotWidth, plotHeight);

    // 绘制网格线
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 0.5;
    const gridLines = 10;
    for (let i = 0; i <= gridLines; i++) {
      const x = plotX + (i / gridLines) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, plotY);
      ctx.lineTo(x, plotY + plotHeight);
      ctx.stroke();
    }

    // 绘制时间轴标签
    ctx.fillStyle = "#666";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= gridLines; i++) {
      const t = minTime + (i / gridLines) * timeRange;
      const x = plotX + (i / gridLines) * plotWidth;
      ctx.fillText(t.toFixed(1) + "s", x, plotY + plotHeight + 5);
    }

    // 绘制位姿数据曲线
    if (poseData.length > 0) {
      // 绘制x坐标
      const xValues = poseData.map((p) => p.x).filter((x) => x != null) as number[];
      if (xValues.length > 0) {
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const xRange = Math.max(maxX - minX, 0.01);
        const xPadding = xRange * 0.1;
        const plotMinX = minX - xPadding;
        const plotMaxX = maxX + xPadding;
        const plotRangeX = plotMaxX - plotMinX;

        // Y轴标签
        ctx.fillStyle = "#4CAF50";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("X", plotX - 10, plotY + plotHeight / 3);

        ctx.strokeStyle = "#4CAF50";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (let i = 0; i < poseData.length; i++) {
          const p = poseData[i];
          if (p && p.x != null) {
            const x = plotX + ((p.timeSec - minTime) / timeRange) * plotWidth;
            const y = plotY + plotHeight - ((p.x - plotMinX) / plotRangeX) * plotHeight;
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }

      // 绘制y坐标
      const yValues = poseData.map((p) => p.y).filter((y) => y != null) as number[];
      if (yValues.length > 0) {
        const minY = Math.min(...yValues);
        const maxY = Math.max(...yValues);
        const yRange = Math.max(maxY - minY, 0.01);
        const yPadding = yRange * 0.1;
        const plotMinY = minY - yPadding;
        const plotMaxY = maxY + yPadding;
        const plotRangeY = plotMaxY - plotMinY;

        // Y轴标签
        ctx.fillStyle = "#2196F3";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("Y", plotX - 10, plotY + (plotHeight * 2) / 3);

        ctx.strokeStyle = "#2196F3";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (let i = 0; i < poseData.length; i++) {
          const p = poseData[i];
          if (p && p.y != null) {
            const x = plotX + ((p.timeSec - minTime) / timeRange) * plotWidth;
            const y = plotY + plotHeight - ((p.y - plotMinY) / plotRangeY) * plotHeight;
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }

      // 绘制theta角度
      const thetaValues = poseData.map((p) => p.theta).filter((t) => t != null) as number[];
      if (thetaValues.length > 0) {
        const minTheta = Math.min(...thetaValues);
        const maxTheta = Math.max(...thetaValues);
        const thetaRange = Math.max(maxTheta - minTheta, 0.01);
        const thetaPadding = thetaRange * 0.1;
        const plotMinTheta = minTheta - thetaPadding;
        const plotMaxTheta = maxTheta + thetaPadding;
        const plotRangeTheta = plotMaxTheta - plotMinTheta;

        // Y轴标签
        ctx.fillStyle = "#FF9800";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("θ", plotX - 10, plotY + plotHeight / 2);

        ctx.strokeStyle = "#FF9800";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (let i = 0; i < poseData.length; i++) {
          const p = poseData[i];
          if (p && p.theta != null) {
            const x = plotX + ((p.timeSec - minTime) / timeRange) * plotWidth;
            const y = plotY + plotHeight - ((p.theta - plotMinTheta) / plotRangeTheta) * plotHeight;
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }
    }

    // 绘制segments：在时间轴上显示start（红色）和end（蓝色）点，并连线
    segments.forEach((seg) => {
      const startX = plotX + ((seg.startSec - minTime) / timeRange) * plotWidth;
      const endX = plotX + ((seg.endSec - minTime) / timeRange) * plotWidth;
      const timelineY = plotY + plotHeight + 20; // 时间轴位置

      // 绘制start和end之间的连线
      ctx.strokeStyle = `rgba(156, 39, 176, ${0.6})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX, timelineY);
      ctx.lineTo(endX, timelineY);
      ctx.stroke();

      // 绘制起点（红色）
      ctx.fillStyle = "#F44336"; // 红色
      ctx.beginPath();
      ctx.arc(startX, timelineY, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 绘制终点（蓝色）
      ctx.fillStyle = "#2196F3"; // 蓝色
      ctx.beginPath();
      ctx.arc(endX, timelineY, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 绘制垂直连接线（从时间轴到图表区域）
      ctx.strokeStyle = `rgba(156, 39, 176, ${0.3})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(startX, plotY + plotHeight);
      ctx.lineTo(startX, timelineY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(endX, plotY + plotHeight);
      ctx.lineTo(endX, timelineY);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // 绘制当前时间进度线（更明显的标识）
    if (currentTimeSec != null) {
      const currentX = plotX + ((currentTimeSec - minTime) / timeRange) * plotWidth;
      if (currentX >= plotX && currentX <= plotX + plotWidth) {
        // 绘制垂直进度线
        ctx.strokeStyle = "#F44336";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(currentX, plotY);
        ctx.lineTo(currentX, plotY + plotHeight + 30);
        ctx.stroke();

        // 绘制进度指示器（三角形）
        ctx.fillStyle = "#F44336";
        ctx.beginPath();
        ctx.moveTo(currentX, plotY + plotHeight + 30);
        ctx.lineTo(currentX - 8, plotY + plotHeight + 40);
        ctx.lineTo(currentX + 8, plotY + plotHeight + 40);
        ctx.closePath();
        ctx.fill();

        // 显示当前时间文本
        ctx.fillStyle = "#F44336";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(currentTimeSec.toFixed(2) + "s", currentX, plotY + plotHeight + 45);
      }
    }

    // 绘制pending start标记
    if (pendingStartSec != null) {
      const pendingX = plotX + ((pendingStartSec - minTime) / timeRange) * plotWidth;
      if (pendingX >= plotX && pendingX <= plotX + plotWidth) {
        ctx.strokeStyle = "#FFC107";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(pendingX, plotY);
        ctx.lineTo(pendingX, plotY + plotHeight);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 绘制图例
    const legendY = plotY - 20;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    let legendX = plotX;
    const legendSpacing = 80;

    if (poseData.some((p) => p.x != null)) {
      ctx.fillStyle = "#4CAF50";
      ctx.fillRect(legendX, legendY - 5, 20, 2);
      ctx.fillStyle = "#333";
      ctx.fillText("X", legendX + 25, legendY - 4);
      legendX += legendSpacing;
    }
    if (poseData.some((p) => p.y != null)) {
      ctx.fillStyle = "#2196F3";
      ctx.fillRect(legendX, legendY - 5, 20, 2);
      ctx.fillStyle = "#333";
      ctx.fillText("Y", legendX + 25, legendY - 4);
      legendX += legendSpacing;
    }
    if (poseData.some((p) => p.theta != null)) {
      ctx.fillStyle = "#FF9800";
      ctx.fillRect(legendX, legendY - 5, 20, 2);
      ctx.fillStyle = "#333";
      ctx.fillText("θ", legendX + 25, legendY - 4);
      legendX += legendSpacing;
    }
    ctx.fillStyle = "#F44336";
    ctx.beginPath();
    ctx.arc(legendX, legendY - 4, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.fillText("Start", legendX + 10, legendY - 4);
    legendX += legendSpacing;
    ctx.fillStyle = "#2196F3";
    ctx.beginPath();
    ctx.arc(legendX, legendY - 4, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.fillText("End", legendX + 10, legendY - 4);
  }, [poseData, segments, currentTimeSec, pendingStartSec]);

  return (
    <div style={{ padding: 12, fontFamily: "sans-serif", pointerEvents: "auto" }}>
      <h2 style={{ margin: "0 0 8px" }}>Umi Crop</h2>

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

      {/* 位姿可视化 */}
      <div style={{ marginBottom: 14 }}>
        <h3 style={{ margin: "0 0 8px" }}>位姿可视化</h3>
        <div style={{ marginBottom: 8 }}>
          <label style={{ marginRight: 8, color: "#555" }}>选择位姿Topic:</label>
          <select
            value={selectedPoseTopic ?? ""}
            onChange={(e) => {
              const topic = e.target.value || undefined;
              setSelectedPoseTopic(topic);
              if (!topic) {
                poseDataRef.current = [];
                setPoseData([]);
              }
            }}
            style={{ padding: "4px 8px", border: "1px solid #ccc", borderRadius: 4, minWidth: 200 }}
          >
            <option value="">-- 未选择 --</option>
            {(topics ?? [])
              .filter((t) => t.name.toLowerCase().includes("pose") || t.name.toLowerCase().includes("odom") || t.name.toLowerCase().includes("transform"))
              .map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
          </select>
          {selectedPoseTopic && (
            <button
              onClick={() => {
                poseDataRef.current = [];
                setPoseData([]);
              }}
              style={{ marginLeft: 8, padding: "4px 8px" }}
            >
              清空数据
            </button>
          )}
        </div>
        <div style={{ border: "1px solid #ccc", borderRadius: 6, padding: 8, background: "#fff" }}>
          <canvas
            ref={canvasRef}
            width={900}
            height={350}
            style={{ width: "100%", maxWidth: "100%", height: "auto", display: "block", cursor: "crosshair" }}
            onClick={(e) => {
              if (!canvasRef.current) return;
              const rect = canvasRef.current.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const ratio = canvasRef.current.width / rect.width;
              const clickX = x * ratio;

              // 计算点击的时间（考虑padding）
              const padding = { left: 60, right: 20 };
              const plotWidth = canvasRef.current.width - padding.left - padding.right;
              const plotX = padding.left;

              if (clickX < plotX || clickX > plotX + plotWidth) return;

              const allTimes = [
                ...poseData.map((p) => p.timeSec),
                ...segments.flatMap((s) => [s.startSec, s.endSec]),
                currentTimeSec ?? 0,
              ].filter((t) => t > 0);
              if (allTimes.length === 0) return;

              const minTime = Math.min(...allTimes);
              const maxTime = Math.max(...allTimes);
              const timeRange = Math.max(maxTime - minTime, 1);
              const clickedTime = minTime + ((clickX - plotX) / plotWidth) * timeRange;

              // 跳转到点击的时间
              if (canSeek) {
                seekTo(clickedTime);
              }
            }}
          />
        </div>
        {selectedPoseTopic && (
          <div style={{ marginTop: 4, fontSize: "12px", color: "#666" }}>
            数据点: {poseData.length} | 点击时间轴可跳转到对应时间
          </div>
        )}
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
