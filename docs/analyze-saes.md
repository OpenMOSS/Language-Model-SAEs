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

## Visualize Feature Analysis

We have successfully retrieved top activation contexts of each feature. But we definitely do not want to look at each token and each feature's activation value on it. Luckily, `Language-Model-SAEs` provide two methods to visualize the feature analyses.

### CLI Feature Preview

You can preview top activation contexts of a certain feature via the CLI. After analyzing an SAE, you can run:

```bash
lm-saes show feature <sae-name> <feature-index>
```

to preview the feature with its analyses. Here's an example output:

<div class="cli-output" markdown="0">
<pre>$ lm-saes show feature qwen3-1.7b-plt-8x-topk64-layer13 7893</pre>

<div class="cli-panel cli-panel-info">
<div class="cli-panel-header">Feature Info</div>
<div class="cli-panel-content">
<div><span class="cli-panel-title">Feature #7893</span> @ qwen3-1.7b-plt-8x-topk64-layer13</div>
<div class="cli-panel-subtitle">termination condition</div>
</div>
</div>

<div class="cli-panel cli-panel-stats">
<div class="cli-panel-header">Statistics</div>
<div class="cli-panel-content">
<div><span class="cli-label">Activation Times:</span> 1,207,198</div>
<div><span class="cli-label">Max Activation:</span> 5.0000</div>
<div><span class="cli-label">Analyzed Tokens:</span> 472,725,125</div>
</div>
</div>

<div class="cli-legend">
<strong>Top 5 Activation Samples</strong>&nbsp;&nbsp;(<span class="tok-weak">Weak</span> | <span class="tok-medium">Medium</span> | <span class="tok-strong">Strong</span>)
</div>

<div class="cli-sample">
<div class="cli-sample-header">Sample 1 (max act: 5.0000)</div>
<div class="cli-sample-content"> remove 1, 2, or 3 stones from the pile. The player who<span class="tok-medium"> removes</span><span class="tok-weak"> the</span><span class="tok-strong"> last</span><span class="tok-medium"> stone</span> wins. If the first
player goes first, what is their winning strategy?&lt;|im_end|&gt;
&lt;|im_start|&gt;</div>
</div>

<div class="cli-sample">
<div class="cli-sample-header">Sample 2 (max act: 4.7500)</div>
<div class="cli-sample-content">imax algorithm with alpha-beta pruning to search for the best move. It recursively<span class="tok-medium"> explores</span><span class="tok-weak"> the</span>
game tree to<span class="tok-weak"> a</span> specified<span class="tok-medium"> depth</span> and returns the evaluation score.

The `get_best_move` function iterates over all</div>
</div>

<div class="cli-sample">
<div class="cli-sample-header">Sample 3 (max act: 4.7188)</div>
<div class="cli-sample-content"><span class="tok-medium"> returns</span><span class="tok-weak"> a</span> pagination token
        //<span class="tok-weak"> that</span> matches the most recent token provided to the service.
        <span class="tok-strong">Stop</span><span class="tok-medium">On</span>DuplicateToken bool
}

// DescribeFleetAttributesPaginator is a paginator for DescribeFleetAttributes</div>
</div>

<div class="cli-sample">
<div class="cli-sample-header">Sample 4 (max act: 4.6875)</div>
<div class="cli-sample-content">::wostream&amp; os, T const* str, size_t const max_size, bool const<span class="tok-medium"> stop_at_null</span>)
    {
        for(size_t i = 0; i &lt; max_size; ++</div>
</div>

<div class="cli-sample">
<div class="cli-sample-header">Sample 5 (max act: 4.4375)</div>
<div class="cli-sample-content"> mathematical expression. The valid characters are letters, numbers,
                                              and
                                                                 underscore. The first character
must be a lowercase letter.
                                           &lt;/p&gt;

                                           &lt;p&gt;&lt;em</div>
</div>
</div>

The highlighted tokens show where the feature activates, with colors indicating activation strength: <span class="tok-weak">weak</span>, <span class="tok-medium">medium</span>, and <span class="tok-strong">strong</span>. In this example, Feature #7893 appears to detect "termination condition" patterns—contexts related to stopping, ending, or terminal states in algorithms and data structures.

### Web UI

For a more comprehensive exploration experience, you can launch the web server to browse all features interactively. The server provides a visual interface for exploring feature analyses. To use the Web UI, you can either manually launch the Python backend and React frontend, or launch them through [Docker Compose](https://docs.docker.com/compose/).

=== "Manual"

    1.  **Launch Backend**: Start the FastAPI server using `uvicorn`. You may need to create a `.env` file in the `server` directory first (see `server/.env.example`).
        ```bash
        uvicorn server.app:app --port 24577 --env-file server/.env
        ```

    2.  **Launch Frontend**: The frontend uses [Bun](https://bun.sh/) for dependency management. Install dependencies and start the development server.
        ```bash
        cd ui
        bun install
        # Copy .env.example and configure BACKEND_URL if necessary
        cp .env.example .env
        bun dev --port 24576
        ```

    After both are running, you can access the Web UI at `http://localhost:24576`.

=== "Docker Compose"

    You can launch the entire stack (MongoDB, Backend, and Frontend) using Docker Compose. Create a `docker-compose.yml` file with the following content:

    ```yaml
    services:
      mongodb:
        image: mongo:latest
        restart: always
        ports:
          - "27017:27017"
        volumes:
          - mongodb_data:/data/db

      backend:
        image: ghcr.io/openmoss/language-model-saes-backend:latest
        restart: always
        ports:
          - "24577:24577"
        environment:
          - MONGO_URI=mongodb://mongodb:27017/
          - MONGO_DB=mechinterp
        # volumes:
        #   - ./models:/models
        #   - ./datasets:/datasets
        #   - ./saes:/saes
        depends_on:
          - mongodb

      frontend:
        image: ghcr.io/openmoss/language-model-saes-frontend:latest
        restart: always
        ports:
          - "24576:24576"
        environment:
          - BACKEND_URL=http://backend:24577
        depends_on:
          - backend

    volumes:
      mongodb_data:
    ```

    Note the above configuration contains a container for MongoDB. If you have launched your MongoDB instance/container elsewhere, configure it properly through the `MONGO_URI` environmental variable in `backend`.

    Then run:

    ```bash
    docker compose up
    ```

    The Web UI will be available at `http://localhost:24576`.

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

