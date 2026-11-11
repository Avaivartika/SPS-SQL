"""Installable DataDesigner plugin descriptors, without engine imports."""
from __future__ import annotations

from data_designer.plugins.plugin import Plugin, PluginType


auxiliary_prediction_plugin = Plugin(
    config_qualified_name="sps.integration.config.AuxiliaryPredictionConfig",
    impl_qualified_name="sps.integration.postprocessing.AuxiliaryPredictionGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

post_correction_plugin = Plugin(
    config_qualified_name="sps.integration.config.PostCorrectionConfig",
    impl_qualified_name="sps.integration.postprocessing.PostCorrectionGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

schema_context_plugin = Plugin(
    config_qualified_name="sps.integration.config.SchemaContextConfig",
    impl_qualified_name="sps.integration.schema_context.SchemaContextGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

raw_selection_plugin = Plugin(
    config_qualified_name="sps.integration.config.RawSelectionConfig",
    impl_qualified_name="sps.integration.raw_selection.RawSelectionGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

candidate_archive_plugin = Plugin(
    config_qualified_name="sps.integration.config.CandidateArchiveConfig",
    impl_qualified_name="sps.integration.generators.CandidateArchiveGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

population_plugin = Plugin(
    config_qualified_name="sps.integration.config.PopulationConfig",
    impl_qualified_name="sps.integration.populations.PopulationGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

cem_plugin = Plugin(
    config_qualified_name="sps.integration.config.CEMUpdateConfig",
    impl_qualified_name="sps.integration.generators.CEMUpdateGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
