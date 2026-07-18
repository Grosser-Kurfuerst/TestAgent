"""Contracts and data tooling for the formal SFT warm-start pipeline."""

from my_agent.sft.contracts import (
    CANONICAL_SFT_SCHEMA_VERSION,
    DATASET_MANIFEST_SCHEMA_VERSION,
    ENVIRONMENT_EXCLUSION_CODES,
    EXPECTED_OUTPUT_KINDS,
    RENDERED_MANIFEST_SCHEMA_VERSION,
    RENDERED_SFT_SCHEMA_VERSION,
    SFT_RUN_MANIFEST_SCHEMA_VERSION,
    deterministic_tool_call_id,
    validate_expected_output_contract,
)
from my_agent.sft.build import (
    EXPERT_CORRECTION_SCHEMA_VERSION,
    SYNTHETIC_SAMPLE_SCHEMA_VERSION,
    ExpertCorrection,
    SFTBuildResult,
    SyntheticSFTRecord,
    build_canonical_sft_dataset,
    validate_semantic_sample,
)
from my_agent.sft.export import (
    LLAMAFACTORY_EXPORT_SCHEMA_VERSION,
    LLAMAFACTORY_LOCK_SCHEMA_VERSION,
    PARITY_REPORT_SCHEMA_VERSION,
    LlamaFactoryExportManifest,
    LlamaFactoryLock,
    export_llamafactory_dataset,
    project_semantic_sample,
)
from my_agent.sft.manifest import SFTDatasetManifest
from my_agent.sft.parity import TemplateParityReport, verify_llamafactory_parity
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample
from my_agent.sft.semantic import SemanticSFTSample


__all__ = [
    "CANONICAL_SFT_SCHEMA_VERSION",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "ENVIRONMENT_EXCLUSION_CODES",
    "EXPERT_CORRECTION_SCHEMA_VERSION",
    "EXPECTED_OUTPUT_KINDS",
    "ExpertCorrection",
    "LLAMAFACTORY_EXPORT_SCHEMA_VERSION",
    "LLAMAFACTORY_LOCK_SCHEMA_VERSION",
    "LlamaFactoryExportManifest",
    "LlamaFactoryLock",
    "PARITY_REPORT_SCHEMA_VERSION",
    "RENDERED_MANIFEST_SCHEMA_VERSION",
    "RENDERED_SFT_SCHEMA_VERSION",
    "SFT_RUN_MANIFEST_SCHEMA_VERSION",
    "SFTBuildResult",
    "SFTDatasetManifest",
    "SYNTHETIC_SAMPLE_SCHEMA_VERSION",
    "SyntheticSFTRecord",
    "TemplateParityReport",
    "RenderedSFTManifest",
    "RenderedSFTSample",
    "SemanticSFTSample",
    "build_canonical_sft_dataset",
    "export_llamafactory_dataset",
    "deterministic_tool_call_id",
    "project_semantic_sample",
    "validate_expected_output_contract",
    "validate_semantic_sample",
    "verify_llamafactory_parity",
]
