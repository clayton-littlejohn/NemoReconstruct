export type ReconstructionStatusValue =
  | "uploading"
  | "queued"
  | "extracting_frames"
  | "feature_extraction"
  | "feature_matching"
  | "sparse_reconstruction"
  | "fvdb_reconstruction"
  | "exporting"
  | "completed"
  | "failed";

export type ReconstructionParams = {
  frame_rate?: number;
  sequential_matcher_overlap?: number;
  colmap_mapper_type?: string;
  colmap_max_num_features?: number;
  fvdb_max_epochs?: number;
  fvdb_sh_degree?: number;
  fvdb_image_downsample_factor?: number;
  splat_only_mode?: boolean;
};

export type ReconstructionDetail = {
  id: string;
  name: string;
  description: string | null;
  status: ReconstructionStatusValue;
  pipeline_slug: string;
  processing_step: string | null;
  processing_pct: number;
  error_message: string | null;
  source_video_filename: string;
  frame_count: number | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  processing_params: ReconstructionParams;
  artifact_ply_url: string | null;
  artifact_usdz_url: string | null;
  artifact_bundle_url: string | null;
  artifact_log_url: string | null;
  artifact_metadata_url: string | null;
};

export type ReconstructionStatus = {
  id: string;
  status: ReconstructionStatusValue;
  processing_step: string | null;
  processing_pct: number;
  error_message: string | null;
  updated_at: string;
};

export type ReconstructionArtifacts = {
  source_video_url: string;
  splat_ply_url: string | null;
  scene_usdz_url: string | null;
  sim_bundle_url: string | null;
  run_log_url: string | null;
  metadata_url: string | null;
};

export type PipelineInfo = {
  slug: string;
  name: string;
  description: string;
  source_type: string;
  output_types: string[];
  steps: string[];
  requirements: string[];
  tunable_params: Record<string, string>;
};

export type MetricsEntry = {
  epoch: number;
  metric: string;
  value: number;
};

export type MetricsResponse = {
  id: string;
  entries: MetricsEntry[];
  summary: Record<string, number>;
};

export type WorkflowDetail = {
  id: string;
  scene_name: string;
  video_filename: string;
  status: string;
  current_agent: string | null;
  current_step: string | null;
  iteration: number;
  max_iterations: number;
  last_verdict: string | null;
  last_reason: string | null;
  reconstruction_id: string | null;
  error_message: string | null;
  pid: number | null;
  created_at: string;
  updated_at: string;
};

export type IterationSummary = {
  iteration: number;
  params: ReconstructionParams;
  loss: number | null;
  ssim: number | null;
  num_gaussians: number | null;
  verdict: string | null;
  reason: string | null;
  ply_url: string | null;
  started_at: string | null;
  completed_at: string | null;
};

export type IterationHistoryResponse = {
  reconstruction_id: string;
  iterations: IterationSummary[];
};
