# Analyze Sparse Autoencoders

What can a trained Sparse Autoencoder tell us? As an approach to **Interpretability**, we definitely want to see what each individual latent of a Sparse Autoencoder (i.e., **feature**) means.

`Language-Model-SAEs` incorporates a bunch of methods to explore the functionality of each individual feature, primarily on on what context a feature activates. If an SAE is trained well, you can naturally observe that there's a type of commonality among these contexts. The language model extracts information from these context and expresses it by the feature's activation. Other types of analytical methods are also supported, including Direct Logit Attribution and Automated Interpretation.

## Setup Prerequisites

A [MongoDB](https://www.mongodb.com/) instance is required to save all the analyses and speed up feature-level queries. To install MongoDB on your system and launch an instance, we refer you to read the official [documentation](https://www.mongodb.com/docs/v7.0/administration/install-on-linux/) of MongoDB.

Alternatively, to launch MongoDB with [Docker](https://www.docker.com/), run the following command:

```bash
docker run -d --name mongodb --restart always -p 27017:27017 mongo:latest
```

## Analyze a trained Sparse Autoencoder

A main entrypoint of feaature analyzing is provided for basic feature statistical information, including the activation context at different magnitudes.

To analyze a trained Sparse Autoencoder, you can run the following variants:

=== "Runner"

    Create the `AnalyzeSAESettings` and call `analyze_sae` with it. 

    ```python
    import torch
    from lm_saes import (
        AnalyzeSAESettings,
        analyze_sae,
        PretrainedSAE,
        DatasetConfig,
        ActivationFactoryConfig,
        ActivationFactoryDatasetSource,
        ActivationFactoryTarget,
        FeatureAnalyzerConfig,
        LanguageModelConfig,
    )

    settings = AnalyzeSAESettings(
        sae=PretrainedSAE(
            pretrained_name_or_path="results",
        ),
        sae_name="pythia-160m-sae",
        sae_series="pythia-sae",
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        ),
        model_name="pythia-160m",
        datasets={
            "SlimPajama-3B": DatasetConfig(
                dataset_name_or_path="Hzfinfdu/SlimPajama-3B",
            ) 
        },
        activation_factory=ActivationFactoryConfig(
            sources=[ActivationFactoryDatasetSource(name="SlimPajama-3B")],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=32,
            context_size=1024,
        ),
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=100_000_000,
        )
        mongo=MongoDBConfig(),
    )

    analyze_sae(settings)
    ```

=== "CLI"

    CLI-based workflow requires a configuration file containing the settings consistent with `AnalyzeSAESettings`. 

    Create a TOML configuration file (e.g., `analyze_config.toml`) with the following content:

    ```toml
    sae_name = "pythia-160m-sae"
    sae_series = "pythia-sae"
    model_name = "pythia-160m"
    output_dir = "analysis_results"

    [sae]
    pretrained_name_or_path = "results"

    [model]
    model_name = "EleutherAI/pythia-160m"
    device = "cuda"
    dtype = "torch.float16"

    [datasets."SlimPajama-3B"]
    dataset_name_or_path = "Hzfinfdu/SlimPajama-3B"

    [activation_factory]
    target = "activations-2d"
    hook_points = ["blocks.6.hook_resid_post"]
    batch_size = 32
    context_size = 1024

    [[activation_factory.sources]]
    type = "dataset"
    name = "SlimPajama-3B"

    [mongo]
    mongo_uri = "localhost"
    
    [analyzer]
    total_analyzing_tokens = 10_000_000
    ```

    Then run the analysis with:

    ```bash
    lm-saes analyze analyze_config.toml
    ```

=== "Full Script"

    For more granular control, you can use the `FeatureAnalyzer` directly.

    ```python
    import datasets
    import torch
    from lm_saes import (
        ActivationFactory,
        ActivationFactoryConfig,
        ActivationFactoryDatasetSource,
        ActivationFactoryTarget,
        LanguageModelConfig,
        FeatureAnalyzer,
        FeatureAnalyzerConfig,
        TransformerLensLanguageModel,
        AbstractSparseAutoEncoder,
    )

    # Load Model & Dataset
    model = TransformerLensLanguageModel(
        LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        )
    )

    dataset = datasets.load_dataset(
        "Hzfinfdu/SlimPajama-3B",
        split="train",
    )

    # Generate Activations
    activation_factory = ActivationFactory(
        ActivationFactoryConfig(
            sources=[ActivationFactoryDatasetSource(name="SlimPajama-3B")],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=32,
            context_size=1024,
        )
    )

    # Load trained SAE from disk
    sae = AbstractSparseAutoEncoder.from_pretrained("results", device="cuda")

    # Analyze it
    analyzer = FeatureAnalyzer(
        FeatureAnalyzerConfig(total_analyzing_tokens=100_000_000)
    )

    result = analyzer.analyze_chunk(
        activation_factory,
        sae=sae,
    )
    ```

Note that a key difference of activation generation between training and analyzing is: we want activations with their complete contexts in analyzing. These tokens are only meaningful (to human) when the surrounding contexts are present. In comparison, SAEs are unaware of the contexts of activations in training, but just treat activations at different context positions as equal. Thus, we here generate activations with `ActivationFactoryTarget.ACTIVATIONS_2D` in `ActivationFactoryConfig`. This stops our generation process breaking down the with-context activations and shuffling them.

## Direct Logit Attribution

Direct Logit Attribution (DLA) helps understand how each feature directly contributes to the model's output logits. It computes the projection of the feature's decoder weight onto the unembedding matrix.

DLA is like an opposite of the top activation contexts: the top activation contexts are the most related **inputs** to a certain feature which makes it activate, while the DLA concerns about the most related **output** that the feature likely induces. Higher layer features are likely to have more direct effect on the output side and show clearer inclination in their DLA logits.

To perform DLA, you can use the `direct_logit_attribute` runner:

```python
from lm_saes import DirectLogitAttributeSettings, direct_logit_attribute, DirectLogitAttributorConfig, PretrainedSAE

settings = DirectLogitAttributeSettings(
    sae=PretrainedSAE(pretrained_name_or_path="results"),
    sae_name="pythia-160m-sae",
    sae_series="pythia-sae",
    model_name="EleutherAI/pythia-160m",
    direct_logit_attributor=DirectLogitAttributorConfig(
        top_k=10,
    ),
    mongo=MongoDBConfig(),
)

direct_logit_attribute(settings)
```

## Automated Interpretation

`Language-Model-SAEs` supports automated interpretation of features using LLMs. The interpretation are mostly generated through investigating the top activation context of each feature. While not perfect, it can help human to quickly gain a brief cognition of the feature.

To run automated interpretation, you can use the `auto_interp` runner:

```python
from lm_saes import AutoInterpSettings, auto_interp, AutoInterpConfig, LanguageModelConfig, MongoDBConfig

settings = AutoInterpSettings(
    sae_name="pythia-160m-sae",
    sae_series="pythia-sae",
    model=LanguageModelConfig(
        model_name="EleutherAI/pythia-160m",
        device="cuda",
        dtype="torch.float16",
    ),
    model_name="pythia-160m",
    auto_interp=AutoInterpConfig(
        openai_api_key="your-api-key",
        openai_model="gpt-4o",
    ),
    mongo=MongoDBConfig(),
)

auto_interp(settings)
```

