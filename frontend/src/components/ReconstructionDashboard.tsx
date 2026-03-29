"use client";

import { FormEvent, Fragment, useEffect, useMemo, useRef, useState } from "react";

import {
  absoluteArtifactUrl,
  deleteReconstruction,
  getDatasets,
  getIterationHistory,
  getMetrics,
  getReconstructions,
  getWorkflows,
  retryReconstruction,
  startWorkflow,
  startWorkflowFromDataset,
  stopWorkflow,
  deleteWorkflow,
} from "@/lib/api";
import type { DatasetInfo, IterationHistoryResponse, IterationSummary, MetricsResponse, ReconstructionDetail, ReconstructionParams, WorkflowDetail } from "@/lib/types";


function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function prettyStatus(value: string): string {
  return value.replaceAll("_", " ");
}


/* ── Workflow pipeline steps for visualization ── */
const WORKFLOW_STEPS = [
  { key: "upload", label: "Upload", agents: [] as string[] },
  { key: "runner-1", label: "Agent A (Runner)", agents: ["runner"] },
  { key: "evaluator", label: "Agent B (Evaluator)", agents: ["evaluator"] },
  { key: "runner-retry", label: "Agent A (Retry)", agents: ["runner"] },
  { key: "done", label: "Done", agents: [] as string[] },
] as const;

function agentDisplayName(agent: string | null): string {
  if (agent === "runner") return "Agent A (Runner)";
  if (agent === "evaluator") return "Agent B (Evaluator)";
  return agent ?? "";
}

function getActiveStepIndex(w: WorkflowDetail): number {
  if (w.status === "completed") return 4;
  if (w.status === "failed") return -1;
  if (w.status === "pending") return 0;
  if (w.current_agent === "runner" && w.iteration <= 1) return 1;
  if (w.current_agent === "evaluator") return 2;
  if (w.current_agent === "runner" && w.iteration > 1) return 3;
  return 1;
}

function WorkflowPipeline({ workflow, onStop, onDelete }: { workflow: WorkflowDetail; onStop?: (id: string) => void; onDelete?: (id: string) => void }) {
  const activeIdx = getActiveStepIndex(workflow);
  const isFailed = workflow.status === "failed";
  const isActive = workflow.status === "running" || workflow.status === "pending";

  return (
    <div className="wf-card">
      <div className="wf-header">
        <div>
          <strong>{workflow.scene_name}</strong>
          <span className="wf-filename">{workflow.video_filename}</span>
        </div>
        <div className="wf-header-right">
          {isActive && onStop ? (
            <button type="button" className="stop-btn" onClick={() => onStop(workflow.id)}>
              &#x25A0; Stop
            </button>
          ) : null}
          {!isActive && onDelete ? (
            <button type="button" className="delete-btn" onClick={() => onDelete(workflow.id)}>
              &#x2715; Delete
            </button>
          ) : null}
          <span className={`status-pill status-${workflow.status === "running" ? "queued" : workflow.status}`}>
            {workflow.status}{workflow.iteration > 0 ? ` · iter ${workflow.iteration}/${workflow.max_iterations}` : ""}
          </span>
        </div>
      </div>

      <div className="wf-thresholds">
        <span>PSNR ≥ {workflow.accept_psnr_threshold}</span>
        <span>SSIM ≥ {workflow.accept_ssim_threshold}</span>
      </div>

      <div className="wf-pipeline">
        {WORKFLOW_STEPS.map((step, idx) => {
          let cls = "wf-step";
          if (isFailed && idx === activeIdx) cls += " wf-step-failed";
          else if (idx < activeIdx) cls += " wf-step-done";
          else if (idx === activeIdx) cls += " wf-step-active";

          return (
            <Fragment key={step.key}>
              {idx > 0 ? <div className={`wf-connector${idx <= activeIdx ? " wf-connector-done" : ""}`} /> : null}
              <div className={cls}>
                <div className="wf-dot" />
                <span className="wf-label">{step.label}</span>
              </div>
            </Fragment>
          );
        })}
      </div>

      {workflow.current_step ? (
        <div className="wf-detail">
          {workflow.current_agent ? <strong>{agentDisplayName(workflow.current_agent)}</strong> : null}
          {" "}{workflow.current_step}
          {workflow.last_verdict ? <span className="wf-verdict"> · verdict: {workflow.last_verdict}</span> : null}
        </div>
      ) : null}

      {workflow.error_message ? <div className="error-inline">{workflow.error_message}</div> : null}

      {workflow.last_reason ? (
        <div className="wf-reason">{workflow.last_reason}</div>
      ) : null}
    </div>
  );
}


/* ── Upload + start workflow ── */
type SourceMode = "dataset" | "video";

function WorkflowStarter({ onStarted }: { onStarted: () => void }) {
  const [mode, setMode] = useState<SourceMode>("dataset");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [sceneName, setSceneName] = useState("");
  const [maxIter, setMaxIter] = useState(3);
  const [psnrThreshold, setPsnrThreshold] = useState(25.0);
  const [ssimThreshold, setSsimThreshold] = useState(0.85);
  const [backend, setBackend] = useState<string>("fvdb");
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    getDatasets().then((ds) => {
      setDatasets(ds);
      if (ds.length > 0 && !selectedDataset) setSelectedDataset(ds[0].name);
    }).catch((err) => { console.error("getDatasets failed:", err); });
  }, []);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && /\.(mov|mp4|m4v)$/i.test(dropped.name)) {
      setFile(dropped);
      if (!sceneName) setSceneName(dropped.name.replace(/\.[^.]+$/, ""));
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setUploading(true);
    setUploadPct(0);
    try {
      if (mode === "dataset") {
        if (!selectedDataset) throw new Error("Select a dataset");
        const recName = sceneName || selectedDataset;
        await startWorkflowFromDataset(selectedDataset, recName, maxIter, psnrThreshold, ssimThreshold, backend);
      } else {
        if (!file) throw new Error("Select a video file");
        await startWorkflow(file, sceneName || file.name.replace(/\.[^.]+$/, ""), maxIter, setUploadPct, psnrThreshold, ssimThreshold);
      }
      setFile(null);
      setSceneName("");
      onStarted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start workflow");
    } finally {
      setUploading(false);
    }
  }

  const canSubmit = mode === "dataset" ? !!selectedDataset && !uploading : !!file && !uploading;

  return (
    <form className="wf-starter" onSubmit={handleSubmit}>
      {/* Mode selector tabs */}
      <div className="source-tabs">
        <button type="button" className={`source-tab${mode === "dataset" ? " source-tab-active" : ""}`} onClick={() => setMode("dataset")}>
          &#x1F4C2; Select Dataset
        </button>
        <button type="button" className={`source-tab${mode === "video" ? " source-tab-active" : ""}`} onClick={() => setMode("video")}>
          &#x1F3AC; Upload Video
        </button>
      </div>

      {mode === "dataset" ? (
        <div className="dataset-picker">
          {datasets.length === 0 ? (
            <div className="dataset-empty">No datasets found in <code>data/</code> directory</div>
          ) : (
            <div className="dataset-grid">
              {datasets.map((ds) => (
                <label
                  key={ds.name}
                  className={`dataset-card${selectedDataset === ds.name ? " dataset-card-selected" : ""}`}
                >
                  <input
                    type="radio"
                    name="dataset"
                    value={ds.name}
                    checked={selectedDataset === ds.name}
                    onChange={() => {
                      setSelectedDataset(ds.name);
                      if (!sceneName) setSceneName(ds.name);
                    }}
                  />
                  <div className="dataset-card-body">
                    <span className="dataset-icon">&#x1F5BC;</span>
                    <strong>{ds.name}</strong>
                    <span className="dataset-meta">{ds.image_count} images</span>
                    {ds.has_sparse ? (
                      <span className="dataset-badge dataset-badge-ok">COLMAP ready</span>
                    ) : (
                      <span className="dataset-badge dataset-badge-warn">needs COLMAP</span>
                    )}
                    {ds.downsampled_factors.length > 0 ? (
                      <span className="dataset-meta">downsampled: {ds.downsampled_factors.map((f) => `${f}x`).join(", ")}</span>
                    ) : null}
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div
          className={`wf-dropzone${dragActive ? " wf-dropzone-active" : ""}${file ? " wf-dropzone-has-file" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("wf-file-input")?.click()}
        >
          {file ? (
            <div className="wf-file-info">
              <span className="wf-file-icon">&#x1f3ac;</span>
              <span>{file.name}</span>
              <span className="subtle-line">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
            </div>
          ) : (
            <div className="wf-drop-prompt">
              <span className="wf-drop-icon">&#x2B06;</span>
              <span>Drop a .MOV file here or click to browse</span>
            </div>
          )}
          <input
            id="wf-file-input"
            type="file"
            accept="video/quicktime,video/mp4,.mov,.mp4,.m4v"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) {
                setFile(f);
                if (!sceneName) setSceneName(f.name.replace(/\.[^.]+$/, ""));
              }
            }}
          />
        </div>
      )}
      <div className="wf-fields">
        <label>
          <span>Scene name</span>
          <input value={sceneName} onChange={(e) => setSceneName(e.target.value)} placeholder="my-scene" />
        </label>
        <label>
          <span>Engine</span>
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            <option value="fvdb">fVDB (fast)</option>
            <option value="3dgrut">3DGRUT</option>
          </select>
        </label>
        <label>
          <span>Max iterations</span>
          <input type="number" min={1} max={10} value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} />
        </label>
        <label>
          <span>PSNR threshold</span>
          <input type="number" min={10} max={50} step={0.5} value={psnrThreshold} onChange={(e) => setPsnrThreshold(Number(e.target.value))} />
        </label>
        <label>
          <span>SSIM threshold</span>
          <input type="number" min={0.5} max={1.0} step={0.01} value={ssimThreshold} onChange={(e) => setSsimThreshold(Number(e.target.value))} />
        </label>
        <button type="submit" disabled={!canSubmit}>
          {uploading ? `Uploading ${uploadPct}%` : "Start Workflow"}
        </button>
      </div>
      {uploading ? (
        <div className="progress-track" style={{ marginTop: 8 }}>
          <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
        </div>
      ) : null}
      {error ? <p className="error-text">{error}</p> : null}
    </form>
  );
}


/* ── Metric helpers ── */
function MetricValue({ label, value, good, diff }: { label: string; value: string; good?: boolean; diff?: string }) {
  return (
    <div className="metric-item">
      <span className="metric-label">{label}</span>
      <span className={`metric-value${good === true ? " metric-good" : good === false ? " metric-warn" : ""}`}>
        {value}
        {diff ? <span className="metric-diff">{diff}</span> : null}
      </span>
    </div>
  );
}

function ParamDiff({ label, current, previous }: { label: string; current: string; previous?: string }) {
  const changed = previous != null && previous !== current;
  return (
    <div className="metric-item">
      <span className="metric-label">{label}</span>
      <span className={`metric-value${changed ? " metric-changed" : ""}`}>
        {current}
        {changed ? <span className="metric-diff">was {previous}</span> : null}
      </span>
    </div>
  );
}

function metricDiffStr(current: number | undefined, previous: number | undefined): string | undefined {
  if (current == null || previous == null) return undefined;
  const delta = current - previous;
  if (Math.abs(delta) < 0.0001) return undefined;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(4)}`;
}

function paramStr(params: ReconstructionParams, key: keyof ReconstructionParams, defaultVal: string): string {
  const v = params[key];
  if (v == null) return `${defaultVal} (default)`;
  return String(v);
}


/* ── Iteration history component ── */
function IterationTimeline({ history }: { history: IterationHistoryResponse }) {
  if (!history.iterations.length) return null;

  function fmtDelta(current: number | null, previous: number | null): string | undefined {
    if (current == null || previous == null) return undefined;
    const d = current - previous;
    if (Math.abs(d) < 0.0001) return undefined;
    return `${d > 0 ? "+" : ""}${d.toFixed(4)}`;
  }

  function deltaClass(current: number | null, previous: number | null, lowerIsBetter: boolean): string {
    if (current == null || previous == null) return "";
    const d = current - previous;
    if (Math.abs(d) < 0.0001) return "";
    const improved = lowerIsBetter ? d < 0 : d > 0;
    return improved ? " metric-good" : " metric-warn";
  }

  return (
    <div className="detail-section detail-full-width">
      <h3>Iteration History ({history.iterations.length})</h3>
      <div className="iter-timeline">
        {history.iterations.map((iter: IterationSummary, idx: number) => {
          const prev = idx > 0 ? history.iterations[idx - 1] : null;
          return (
            <div key={iter.iteration} className="iter-card">
              <div className="iter-header">
                <strong>Iteration {iter.iteration}</strong>
                {iter.verdict ? (
                  <span className={`status-pill ${iter.verdict === "ACCEPT" ? "status-completed" : "status-queued"}`}>
                    {iter.verdict}
                  </span>
                ) : null}
                {iter.ply_url ? (
                  <a className="ply-link" href={absoluteArtifactUrl(iter.ply_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>
                    &#x2B07; PLY
                  </a>
                ) : null}
              </div>
              <div className="iter-metrics">
                {iter.loss != null ? (
                  <span className={deltaClass(iter.loss, prev?.loss ?? null, true)}>
                    Loss: {iter.loss.toFixed(4)}
                    {fmtDelta(iter.loss, prev?.loss ?? null) ? <span className="metric-diff">{fmtDelta(iter.loss, prev?.loss ?? null)}</span> : null}
                  </span>
                ) : null}
                {iter.ssim != null ? (
                  <span className={deltaClass(iter.ssim, prev?.ssim ?? null, false)}>
                    SSIM: {iter.ssim.toFixed(4)}
                    {fmtDelta(iter.ssim, prev?.ssim ?? null) ? <span className="metric-diff">{fmtDelta(iter.ssim, prev?.ssim ?? null)}</span> : null}
                  </span>
                ) : null}
                {iter.num_gaussians != null ? (
                  <span>Gaussians: {iter.num_gaussians.toLocaleString()}</span>
                ) : null}
              </div>
              {/* Show changed params compared to previous */}
              {prev ? (
                <div className="iter-params-diff">
                  {(Object.keys(iter.params) as (keyof ReconstructionParams)[]).map((key) => {
                    const cur = iter.params[key];
                    const prv = prev.params[key];
                    if (cur == null && prv == null) return null;
                    if (String(cur) === String(prv)) return null;
                    return (
                      <span key={key} className="metric-changed">
                        {key.replace(/_/g, " ")}: {String(cur ?? "default")}
                        <span className="metric-diff">was {String(prv ?? "default")}</span>
                      </span>
                    );
                  })}
                </div>
              ) : null}
              {iter.reason ? <div className="iter-reason">{iter.reason}</div> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}


/* ── Detail panel ── */
function DetailPanel({
  item,
  metrics,
  prevItem,
  prevMetrics,
  iterHistory,
}: {
  item: ReconstructionDetail;
  metrics: MetricsResponse | null;
  prevItem: ReconstructionDetail | null;
  prevMetrics: MetricsResponse | null;
  iterHistory: IterationHistoryResponse | null;
}) {
  const p = item.processing_params;
  const pp = prevItem?.processing_params;
  const s = metrics?.summary;
  const ps = prevMetrics?.summary;
  const backend = p.reconstruction_backend ?? "fvdb";

  // Parse agent notes and reverse so newest is on top
  const noteLines = item.description ? item.description.split("\n").filter(Boolean).reverse() : [];

  return (
    <div className="detail-grid">
      {/* All parameters with full change tracking */}
      <div className="detail-section">
        <h3>Parameters{prevItem ? ` (vs ${prevItem.name})` : ""}</h3>
        <div className="param-grid">
          <ParamDiff label="Backend" current={paramStr(p, "reconstruction_backend", "fvdb")} previous={pp ? paramStr(pp, "reconstruction_backend", "fvdb") : undefined} />
          <ParamDiff label="Frame Rate" current={paramStr(p, "frame_rate", "2.0")} previous={pp ? paramStr(pp, "frame_rate", "2.0") : undefined} />
          <ParamDiff label="Mapper Type" current={paramStr(p, "colmap_mapper_type", "incremental")} previous={pp ? paramStr(pp, "colmap_mapper_type", "incremental") : undefined} />
          <ParamDiff label="Max Features" current={paramStr(p, "colmap_max_num_features", "8192")} previous={pp ? paramStr(pp, "colmap_max_num_features", "8192") : undefined} />
          <ParamDiff label="Matcher Overlap" current={paramStr(p, "sequential_matcher_overlap", "12")} previous={pp ? paramStr(pp, "sequential_matcher_overlap", "12") : undefined} />
          {backend === "3dgrut" ? (
            <>
              <ParamDiff label="GRUT Iterations" current={paramStr(p, "grut_n_iterations", "30000")} previous={pp ? paramStr(pp, "grut_n_iterations", "30000") : undefined} />
              <ParamDiff label="GRUT Render" current={paramStr(p, "grut_render_method", "3dgrt")} previous={pp ? paramStr(pp, "grut_render_method", "3dgrt") : undefined} />
              <ParamDiff label="GRUT Strategy" current={paramStr(p, "grut_strategy", "gs")} previous={pp ? paramStr(pp, "grut_strategy", "gs") : undefined} />
              <ParamDiff label="GRUT Downsample" current={p.grut_downsample_factor != null ? `${p.grut_downsample_factor}x` : "2x (default)"} previous={pp ? (pp.grut_downsample_factor != null ? `${pp.grut_downsample_factor}x` : "2x (default)") : undefined} />
            </>
          ) : (
            <>
              <ParamDiff label="fVDB Epochs" current={paramStr(p, "fvdb_max_epochs", "40")} previous={pp ? paramStr(pp, "fvdb_max_epochs", "40") : undefined} />
              <ParamDiff label="fVDB SH Degree" current={paramStr(p, "fvdb_sh_degree", "3")} previous={pp ? paramStr(pp, "fvdb_sh_degree", "3") : undefined} />
              <ParamDiff label="fVDB Downsample" current={p.fvdb_image_downsample_factor != null ? `${p.fvdb_image_downsample_factor}x` : "6x (default)"} previous={pp ? (pp.fvdb_image_downsample_factor != null ? `${pp.fvdb_image_downsample_factor}x` : "6x (default)") : undefined} />
            </>
          )}
          <ParamDiff label="Splat Only" current={paramStr(p, "splat_only_mode", "false")} previous={pp ? paramStr(pp, "splat_only_mode", "false") : undefined} />
        </div>
      </div>

      {/* Training metrics with diffs */}
      {s && Object.keys(s).length > 0 ? (
        <div className="detail-section">
          <h3>Training Metrics{ps ? ` (vs ${prevItem?.name})` : ""}</h3>
          <div className="param-grid">
            <MetricValue label="Loss" value={s["reconstruct/loss"]?.toFixed(4) ?? "\u2014"} good={s["reconstruct/loss"] != null ? s["reconstruct/loss"] < 0.25 : undefined} diff={metricDiffStr(s["reconstruct/loss"], ps?.["reconstruct/loss"])} />
            <MetricValue label="SSIM Loss" value={s["reconstruct/ssimloss"]?.toFixed(4) ?? "\u2014"} good={s["reconstruct/ssimloss"] != null ? s["reconstruct/ssimloss"] > 0.85 : undefined} diff={metricDiffStr(s["reconstruct/ssimloss"], ps?.["reconstruct/ssimloss"])} />
            <MetricValue label="PSNR (dB)" value={s["reconstruct/psnr"]?.toFixed(2) ?? "\u2014"} good={s["reconstruct/psnr"] != null ? s["reconstruct/psnr"] > 25 : undefined} diff={metricDiffStr(s["reconstruct/psnr"], ps?.["reconstruct/psnr"])} />
            <MetricValue label="SSIM" value={s["reconstruct/ssim"]?.toFixed(4) ?? "\u2014"} good={s["reconstruct/ssim"] != null ? s["reconstruct/ssim"] > 0.85 : undefined} diff={metricDiffStr(s["reconstruct/ssim"], ps?.["reconstruct/ssim"])} />
            <MetricValue label="L1 Loss" value={s["reconstruct/l1loss"]?.toFixed(4) ?? "\u2014"} diff={metricDiffStr(s["reconstruct/l1loss"], ps?.["reconstruct/l1loss"])} />
            <MetricValue label="Gaussians" value={s["reconstruct/num_gaussians"]?.toLocaleString() ?? "\u2014"} />
            <MetricValue label="GPU Mem" value={s["reconstruct/mem_allocated"] != null ? `${s["reconstruct/mem_allocated"].toFixed(2)} GB` : "\u2014"} />
            <MetricValue label="SH Degree" value={s["reconstruct/sh_degree"]?.toFixed(0) ?? "\u2014"} />
          </div>
        </div>
      ) : null}

      {/* Simulation-ready downloads */}
      {(item.artifact_usdz_url || item.artifact_collision_mesh_url || item.artifact_ply_url) ? (
        <div className="detail-section detail-full-width">
          <h3>Simulation-Ready Assets</h3>
          <div className="sim-assets">
            {item.artifact_usdz_url ? (
              <a className="sim-asset-btn sim-asset-usdz" href={absoluteArtifactUrl(item.artifact_usdz_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>
                <span className="sim-asset-icon">&#x1F30D;</span>
                <span className="sim-asset-info">
                  <span className="sim-asset-name">NuRec USDZ</span>
                  <span className="sim-asset-desc">Visual asset for Omniverse / Isaac Sim</span>
                </span>
                <span className="sim-asset-dl">&#x2B07;</span>
              </a>
            ) : null}
            {item.artifact_collision_mesh_url ? (
              <a className="sim-asset-btn sim-asset-collision" href={absoluteArtifactUrl(item.artifact_collision_mesh_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>
                <span className="sim-asset-icon">&#x1F9F1;</span>
                <span className="sim-asset-info">
                  <span className="sim-asset-name">Collision Mesh (OBJ)</span>
                  <span className="sim-asset-desc">Physics collision geometry for Isaac Sim</span>
                </span>
                <span className="sim-asset-dl">&#x2B07;</span>
              </a>
            ) : null}
            {item.artifact_ply_url ? (
              <a className="sim-asset-btn sim-asset-ply" href={absoluteArtifactUrl(item.artifact_ply_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>
                <span className="sim-asset-icon">&#x2B50;</span>
                <span className="sim-asset-info">
                  <span className="sim-asset-name">Gaussian Splat (PLY)</span>
                  <span className="sim-asset-desc">Raw 3D Gaussian point cloud</span>
                </span>
                <span className="sim-asset-dl">&#x2B07;</span>
              </a>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Agent notes — newest first */}
      {noteLines.length > 0 ? (
        <div className="detail-section detail-full-width">
          <h3>Agent Notes (newest first)</h3>
          <div className="detail-notes">
            {noteLines.map((line, i) => (
              <p key={i} className={line.startsWith("[Eval") ? "note-eval" : ""}>{line}</p>
            ))}
          </div>
        </div>
      ) : null}

      {/* Iteration history */}
      {iterHistory && iterHistory.iterations.length > 0 ? (
        <IterationTimeline history={iterHistory} />
      ) : null}
    </div>
  );
}


/* ── Main dashboard ── */
export default function ReconstructionDashboard() {
  const [reconstructions, setReconstructions] = useState<ReconstructionDetail[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowDetail[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [metricsMap, setMetricsMap] = useState<Record<string, MetricsResponse>>({});
  const [iterHistoryMap, setIterHistoryMap] = useState<Record<string, IterationHistoryResponse>>({});
  const [error, setError] = useState<string | null>(null);
  const fetchedMetricsRef = useRef<Set<string>>(new Set());
  const fetchedIterHistoryRef = useRef<Set<string>>(new Set());

  async function refresh() {
    const [recons, wfs] = await Promise.all([getReconstructions(), getWorkflows()]);
    setReconstructions(recons);
    setWorkflows(wfs);
  }

  useEffect(() => {
    refresh().catch((err: Error) => setError(err.message));
    const interval = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    for (const item of reconstructions) {
      if (item.status === "completed" && !fetchedMetricsRef.current.has(item.id)) {
        fetchedMetricsRef.current.add(item.id);
        getMetrics(item.id)
          .then((m) => setMetricsMap((prev) => ({ ...prev, [item.id]: m })))
          .catch(() => undefined);
      }
      if (item.status === "completed" && !fetchedIterHistoryRef.current.has(item.id)) {
        fetchedIterHistoryRef.current.add(item.id);
        getIterationHistory(item.id)
          .then((h) => setIterHistoryMap((prev) => ({ ...prev, [item.id]: h })))
          .catch(() => undefined);
      }
    }
  }, [reconstructions]);

  const prevMap = useMemo(() => {
    const map: Record<string, ReconstructionDetail> = {};
    const sorted = [...reconstructions].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    );
    const groups: Record<string, ReconstructionDetail[]> = {};
    for (const item of sorted) {
      const baseName = item.name.replace(/\s*\(run\s+\d+\)$/i, "").trim();
      if (!groups[baseName]) groups[baseName] = [];
      groups[baseName].push(item);
    }
    for (const group of Object.values(groups)) {
      for (let i = 1; i < group.length; i++) {
        map[group[i].id] = group[i - 1];
      }
    }
    return map;
  }, [reconstructions]);

  const activeCount = useMemo(
    () => reconstructions.filter((item) => !["completed", "failed"].includes(item.status)).length,
    [reconstructions],
  );

  const activeWorkflows = useMemo(
    () => workflows.filter((w) => w.status === "running" || w.status === "pending"),
    [workflows],
  );

  async function handleRetry(id: string) {
    setError(null);
    try {
      await retryReconstruction(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retry failed");
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await deleteReconstruction(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleStop(workflowId: string) {
    setError(null);
    try {
      await stopWorkflow(workflowId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stop failed");
    }
  }

  async function handleDeleteWorkflow(workflowId: string) {
    setError(null);
    try {
      await deleteWorkflow(workflowId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <main className="page-shell">

      {error ? <p className="error-text">{error}</p> : null}

      {/* Agent workflow section */}
      <section className="wf-section">
        <h2>Agent Workflow</h2>
        <p className="header-sub">Select a dataset or upload a video to start the 3D reconstruction pipeline</p>

        <WorkflowStarter onStarted={() => refresh().catch(() => undefined)} />

        {activeWorkflows.length > 0 ? (
          <div className="wf-active">
            <h3>Active Workflows</h3>
            {activeWorkflows.map((w) => (
              <WorkflowPipeline key={w.id} workflow={w} onStop={handleStop} />
            ))}
          </div>
        ) : null}

        {workflows.filter((w) => w.status === "completed" || w.status === "failed").length > 0 ? (
          <details className="wf-history">
            <summary>Past workflows ({workflows.filter((w) => w.status !== "running" && w.status !== "pending").length})</summary>
            {workflows
              .filter((w) => w.status !== "running" && w.status !== "pending")
              .map((w) => (
                <WorkflowPipeline key={w.id} workflow={w} onDelete={handleDeleteWorkflow} />
              ))}
          </details>
        ) : null}
      </section>

      {/* Reconstructions table */}
      <section className="table-card">
        <div className="table-headline">
          <h2>Reconstructions</h2>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Frames</th>
                <th>Updated</th>
                <th>Artifacts</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {reconstructions.map((item) => {
                const m = metricsMap[item.id]?.summary;
                const prev = prevMap[item.id];
                const pm = prev ? metricsMap[prev.id]?.summary : undefined;
                return (
                  <Fragment key={item.id}>
                    <tr
                      onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
                      className={expandedId === item.id ? "row-expanded" : "row-clickable"}
                    >
                      <td>
                        <strong>{item.name}</strong>
                        <div className="subtle-line">{item.source_video_filename}</div>
                        {prev ? <div className="subtle-line iteration-tag">iteration of {prev.name}</div> : null}
                      </td>
                      <td>
                        <span className={`status-pill status-${item.status}`}>{prettyStatus(item.status)}</span>
                        {item.error_message ? <div className="error-inline">{item.error_message}</div> : null}
                        {m && m["reconstruct/loss"] != null ? (
                          <div className="inline-metrics">
                            <span className={m["reconstruct/loss"] < 0.25 ? "metric-good" : "metric-warn"}>
                              Loss {m["reconstruct/loss"].toFixed(3)}
                            </span>
                            {" \u00B7 "}
                            <span className={(m["reconstruct/ssimloss"] ?? 0) > 0.85 ? "metric-good" : "metric-warn"}>
                              SSIM {(m["reconstruct/ssimloss"] ?? 0).toFixed(3)}
                            </span>
                            {pm && pm["reconstruct/loss"] != null ? (
                              <>
                                {" \u00B7 "}
                                <span className={(m["reconstruct/loss"] ?? 0) < (pm["reconstruct/loss"] ?? 0) ? "metric-good" : "metric-warn"}>
                                  \u0394 {((m["reconstruct/loss"] ?? 0) - (pm["reconstruct/loss"] ?? 0)).toFixed(3)}
                                </span>
                              </>
                            ) : null}
                          </div>
                        ) : null}
                      </td>
                      <td>
                        <div className="mini-progress">
                          <div className="progress-track">
                            <div className="progress-fill" style={{ width: `${item.processing_pct}%` }} />
                          </div>
                          <span>{item.processing_pct}%</span>
                        </div>
                        <div className="subtle-line">{item.processing_step ?? "waiting"}</div>
                      </td>
                      <td>{item.frame_count ?? "-"}</td>
                      <td>{formatTime(item.updated_at)}</td>
                      <td>
                        <div className="artifact-links">
                          {item.artifact_ply_url ? (
                            <a className="ply-link" href={absoluteArtifactUrl(item.artifact_ply_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>&#x2B07; PLY</a>
                          ) : null}
                          {item.artifact_usdz_url ? (
                            <a href={absoluteArtifactUrl(item.artifact_usdz_url) ?? undefined} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>NuRec USDZ</a>
                          ) : null}
                          {item.artifact_collision_mesh_url ? (
                            <a href={absoluteArtifactUrl(item.artifact_collision_mesh_url) ?? undefined} download onClick={(e) => e.stopPropagation()}>Collision OBJ</a>
                          ) : null}
                          {item.artifact_bundle_url ? (
                            <a href={absoluteArtifactUrl(item.artifact_bundle_url) ?? undefined} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>ZIP</a>
                          ) : null}
                          {item.artifact_log_url ? (
                            <a href={absoluteArtifactUrl(item.artifact_log_url) ?? undefined} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>LOG</a>
                          ) : null}
                        </div>
                      </td>
                      <td>
                        <div className="action-row">
                          <button type="button" onClick={(e) => { e.stopPropagation(); handleRetry(item.id); }}>
                            Retry
                          </button>
                          <button type="button" className="ghost" onClick={(e) => { e.stopPropagation(); handleDelete(item.id); }}>
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                    {expandedId === item.id ? (
                      <tr className="detail-row">
                        <td colSpan={7}>
                          <DetailPanel
                            item={item}
                            metrics={metricsMap[item.id] ?? null}
                            prevItem={prev ?? null}
                            prevMetrics={prev ? metricsMap[prev.id] ?? null : null}
                            iterHistory={iterHistoryMap[item.id] ?? null}
                          />
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
              {!reconstructions.length ? (
                <tr>
                  <td colSpan={7} className="empty-cell">No reconstructions yet. Start a workflow above.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
