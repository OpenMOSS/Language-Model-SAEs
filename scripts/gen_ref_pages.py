"""Generate the code reference pages and navigation.

Instead of dumping every module, this generates curated reference pages
organized by category, covering only the important public API classes
and functions.
"""

from pathlib import Path

import mkdocs_gen_files

CATEGORIES: dict[str, dict[str, list[str]]] = {
    "Models": {
        "description": "Sparse dictionary model architectures and their configuration classes.",
        "items": [
            "llamascopium.SparseDictionary",
            "llamascopium.SparseDictionaryConfig",
            "llamascopium.SAEConfig",
            "llamascopium.SparseAutoEncoder",
            "llamascopium.CrosscoderConfig",
            "llamascopium.Crosscoder",
            "llamascopium.CLTConfig",
            "llamascopium.CrossLayerTranscoder",
            "llamascopium.LorsaConfig",
            "llamascopium.LowRankSparseAttention",
            "llamascopium.MOLTConfig",
            "llamascopium.MixtureOfLinearTransform",
        ],
    },
    "Training": {
        "description": "Training infrastructure: trainer, optimizer configs, initialization, and logging.",
        "items": [
            "llamascopium.TrainerConfig",
            "llamascopium.Trainer",
            "llamascopium.WandbConfig",
            "llamascopium.InitializerConfig",
            "llamascopium.Initializer",
        ],
    },
    "Evaluation": {
        "description": "Evaluation pipeline for trained sparse dictionaries.",
        "items": [
            "llamascopium.EvalConfig",
            "llamascopium.Evaluator",
        ],
    },
    "Activation": {
        "description": "Activation extraction, caching, and processing.",
        "items": [
            "llamascopium.ActivationFactoryConfig",
            "llamascopium.ActivationFactory",
            "llamascopium.ActivationFactoryTarget",
            "llamascopium.ActivationFactoryDatasetSource",
            "llamascopium.ActivationFactoryActivationsSource",
            "llamascopium.BufferShuffleConfig",
            "llamascopium.ActivationWriterConfig",
            "llamascopium.ActivationWriter",
        ],
    },
    "Analysis": {
        "description": "Post-training feature analysis and interpretability tools.",
        "items": [
            "llamascopium.FeatureAnalyzerConfig",
            "llamascopium.FeatureAnalyzer",
            "llamascopium.DirectLogitAttributorConfig",
            "llamascopium.DirectLogitAttributor",
        ],
    },
    "Runners": {
        "description": "High-level runner functions and their settings for common workflows.",
        "items": [
            "llamascopium.PretrainedSAE",
            "llamascopium.TrainSAESettings",
            "llamascopium.train_sae",
            "llamascopium.TrainCLTSettings",
            "llamascopium.train_clt",
            "llamascopium.TrainCrosscoderSettings",
            "llamascopium.train_crosscoder",
            "llamascopium.TrainLorsaSettings",
            "llamascopium.train_lorsa",
            "llamascopium.TrainMOLTSettings",
            "llamascopium.train_molt",
            "llamascopium.EvaluateSAESettings",
            "llamascopium.evaluate_sae",
            "llamascopium.EvaluateCrosscoderSettings",
            "llamascopium.evaluate_crosscoder",
            "llamascopium.AnalyzeSAESettings",
            "llamascopium.analyze_sae",
            "llamascopium.AnalyzeCrosscoderSettings",
            "llamascopium.analyze_crosscoder",
            "llamascopium.GenerateActivationsSettings",
            "llamascopium.generate_activations",
            "llamascopium.AutoInterpSettings",
            "llamascopium.auto_interp",
            "llamascopium.SweepSAESettings",
            "llamascopium.SweepingItem",
            "llamascopium.sweep_sae",
            "llamascopium.DirectLogitAttributeSettings",
            "llamascopium.direct_logit_attribute",
            "llamascopium.CheckActivationConsistencySettings",
            "llamascopium.check_activation_consistency",
        ],
    },
    "Infrastructure": {
        "description": "Language model backend, dataset, and database configuration.",
        "items": [
            "llamascopium.LanguageModelConfig",
            "llamascopium.TransformerLensLanguageModel",
            "llamascopium.HuggingFaceLanguageModel",
            "llamascopium.DatasetConfig",
            "llamascopium.MongoDBConfig",
            "llamascopium.MongoClient",
        ],
    },
}

nav = mkdocs_gen_files.Nav()

for category_name, category in CATEGORIES.items():
    filename = category_name.lower().replace(" ", "_") + ".md"
    doc_path = Path("reference", filename)

    lines: list[str] = []
    lines.append(f"# {category_name}\n")
    lines.append(f"{category['description']}\n")

    for item in category["items"]:
        lines.append(f"::: {item}\n")

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write("\n".join(lines))

    nav[(category_name,)] = filename

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
