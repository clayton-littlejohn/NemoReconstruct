export type ReconstructionDetail = {
  id: string;
  name: string;
  description: string | null;
  status: string;
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
  status: string;
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

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: BodyInit | null;
};

export class NemoReconstructClient {
  constructor(private readonly baseUrl: string) {}

  private url(path: string): string {
    return `${this.baseUrl.replace(/\/$/, "")}${path}`;
  }

  private async request<T>(path: string, init?: RequestOptions): Promise<T> {
    const response = await fetch(this.url(path), init);
    if (!response.ok) {
      throw new Error(`API ${response.status}: ${await response.text()}`);
    }
    return response.json() as Promise<T>;
  }

  health(): Promise<{ status: string; service: string }> {
    return this.request("/health");
  }

  listPipelines(): Promise<PipelineInfo[]> {
    return this.request("/api/v1/pipelines");
  }

  listReconstructions(): Promise<ReconstructionDetail[]> {
    return this.request("/api/v1/reconstructions");
  }

  getReconstruction(id: string): Promise<ReconstructionDetail> {
    return this.request(`/api/v1/reconstructions/${id}`);
  }

  getStatus(id: string): Promise<ReconstructionStatus> {
    return this.request(`/api/v1/reconstructions/${id}/status`);
  }

  getArtifacts(id: string): Promise<ReconstructionArtifacts> {
    return this.request(`/api/v1/reconstructions/${id}/artifacts`);
  }

  retryReconstruction(id: string, params?: ReconstructionParams): Promise<ReconstructionDetail> {
    return this.request(`/api/v1/reconstructions/${id}/retry`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params ? { params } : {}),
    });
  }

  async deleteReconstruction(id: string): Promise<void> {
    const response = await fetch(this.url(`/api/v1/reconstructions/${id}`), { method: "DELETE" });
    if (!response.ok) {
      throw new Error(`API ${response.status}: ${await response.text()}`);
    }
  }

  uploadVideo(file: File, name: string, description?: string, params?: ReconstructionParams): Promise<ReconstructionDetail> {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    if (description) {
      form.append("description", description);
    }
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          form.append(key, String(value));
        }
      }
    }
    return this.request("/api/v1/reconstructions/upload", {
      method: "POST",
      body: form,
    });
  }
}