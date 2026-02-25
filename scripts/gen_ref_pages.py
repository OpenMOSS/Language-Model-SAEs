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
            "lm_saes.BaseSAEConfig",
            "lm_saes.SAEConfig",
            "lm_saes.SparseAutoEncoder",
            "lm_saes.CrossCoderConfig",
            "lm_saes.CrossCoder",
            "lm_saes.CLTConfig",
            "lm_saes.CrossLayerTranscoder",
            "lm_saes.LorsaConfig",
            "lm_saes.LowRankSparseAttention",
            "lm_saes.MOLTConfig",
            "lm_saes.MixtureOfLinearTransform",
        ],
    },
    "Training": {
        "description": "Training infrastructure: trainer, optimizer configs, initialization, and logging.",
        "items": [
            "lm_saes.TrainerConfig",
            "lm_saes.Trainer",
            "lm_saes.WandbConfig",
            "lm_saes.InitializerConfig",
            "lm_saes.Initializer",
        ],
    },
    "Evaluation": {
        "description": "Evaluation pipeline for trained sparse dictionaries.",
        "items": [
            "lm_saes.EvalConfig",
            "lm_saes.Evaluator",
        ],
    },
    "Activation": {
        "description": "Activation extraction, caching, and processing.",
        "items": [
            "lm_saes.ActivationFactoryConfig",
            "lm_saes.ActivationFactory",
            "lm_saes.ActivationFactoryTarget",
            "lm_saes.ActivationFactoryDatasetSource",
            "lm_saes.ActivationFactoryActivationsSource",
            "lm_saes.BufferShuffleConfig",
            "lm_saes.ActivationWriterConfig",
            "lm_saes.ActivationWriter",
        ],
    },
    "Analysis": {
        "description": "Post-training feature analysis and interpretability tools.",
        "items": [
            "lm_saes.FeatureAnalyzerConfig",
            "lm_saes.FeatureAnalyzer",
            "lm_saes.DirectLogitAttributorConfig",
            "lm_saes.DirectLogitAttributor",
        ],
    },
    "Runners": {
        "description": "High-level runner functions and their settings for common workflows.",
        "items": [
            "lm_saes.PretrainedSAE",
            "lm_saes.TrainSAESettings",
            "lm_saes.train_sae",
            "lm_saes.TrainCLTSettings",
            "lm_saes.train_clt",
            "lm_saes.TrainCrossCoderSettings",
            "lm_saes.train_crosscoder",
            "lm_saes.TrainLorsaSettings",
            "lm_saes.train_lorsa",
            "lm_saes.TrainMOLTSettings",
            "lm_saes.train_molt",
            "lm_saes.EvaluateSAESettings",
            "lm_saes.evaluate_sae",
            "lm_saes.EvaluateCrossCoderSettings",
            "lm_saes.evaluate_crosscoder",
            "lm_saes.AnalyzeSAESettings",
            "lm_saes.analyze_sae",
            "lm_saes.AnalyzeCrossCoderSettings",
            "lm_saes.analyze_crosscoder",
            "lm_saes.GenerateActivationsSettings",
            "lm_saes.generate_activations",
            "lm_saes.AutoInterpSettings",
            "lm_saes.auto_interp",
            "lm_saes.SweepSAESettings",
            "lm_saes.SweepingItem",
            "lm_saes.sweep_sae",
            "lm_saes.DirectLogitAttributeSettings",
            "lm_saes.direct_logit_attribute",
            "lm_saes.CheckActivationConsistencySettings",
            "lm_saes.check_activation_consistency",
        ],
    },
    "Infrastructure": {
        "description": "Language model backend, dataset, and database configuration.",
        "items": [
            "lm_saes.LanguageModelConfig",
            "lm_saes.TransformerLensLanguageModel",
            "lm_saes.DatasetConfig",
            "lm_saes.MongoDBConfig",
            "lm_saes.MongoClient",
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
