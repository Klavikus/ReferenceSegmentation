import json
from pathlib import Path

from backend.contracts.pipeline_models import ExportArtifact, ManifestSample, PipelineManifest


def manifest_to_dict(manifest: PipelineManifest) -> dict:
    return {
        "pipeline_version": manifest.pipeline_version,
        "model_version": manifest.model_version,
        "config_hash": manifest.config_hash,
        "config": {
            "target_width": manifest.config.target_width,
            "target_height": manifest.config.target_height,
            "default_pixel_to_nm": manifest.config.default_pixel_to_nm,
            "marker_generation_strategy": manifest.config.marker_generation_strategy,
            "export_storage_kind": manifest.config.export_storage_kind,
            "contour_archive_name": manifest.config.contour_archive_name,
            "pipeline_version": manifest.config.pipeline_version,
        },
        "input_roles": list(manifest.input_roles),
        "samples": [_manifest_sample_to_dict(sample) for sample in manifest.samples],
    }


def write_manifest(manifest: PipelineManifest, output_path: str | Path) -> Path:
    manifest_path = Path(output_path)
    manifest_path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _manifest_sample_to_dict(sample: ManifestSample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "run_id": sample.run_id,
        "input_role": sample.input_role,
        "original_width": sample.original_width,
        "original_height": sample.original_height,
        "input_shape": list(sample.input_shape),
        "input_dtype": sample.input_dtype,
        "input_sha256": sample.input_sha256,
        "marker_source": sample.marker_source,
        "pixel_to_nm": sample.pixel_to_nm,
        "artifacts": [_artifact_to_dict(artifact) for artifact in sample.artifacts],
    }


def _artifact_to_dict(artifact: ExportArtifact) -> dict:
    return {
        "artifact_type": artifact.artifact_type,
        "path": str(artifact.path),
        "storage_kind": artifact.storage_kind,
        "filename": artifact.filename,
        "mime_type": artifact.mime_type,
    }
