## v2.0.0b8 (2026-01-18)

## v2.0.0b7 (2026-01-18)

### Feat

- **ui/circuits**: add remix button to fill NewGraphDialog with initial configs
- **circuit**: implement circuit generation status tracking and dynamic pruning
- **ui/circuits**: enhance SAE set creation dialog with filtering and search functionality
- **ui/dictionary**: add feature index input and update dictionary select styling
- **ui/circuits**: add BiasNodeSchema and update NodeSchema
- **ui/circuits**: add isFromQkTracing field for filtering (untested)
- **ui**: support dictionary selection filtering
- **ui/circuits**: support logical subgraph
- **ui/circuits**: trace features
- **ui/circuits**: add QK tracing section to node connections
- **ui/circuit**: show new graph dialog when no circuit available
- **ui**: support grouping circuits
- **train**: add auxiliary loss for topk (#164)
- **lorsa**: add auto expand when load from pretrain
- add capability of tracing from features to attribution graphs (#163)

### Fix

- **feature_interpreter**: handle None interpretation in feature analysis
- **ui/circuits**: specify timeout for circuit generation in undici
- **attn_scores_attribution**: correct component expansion for QK calculations
- **ui/circuits**: use ctrl/cmd to multiselect features
- **topk**: improve topk in single card situation
- **graph**: assign sae_name based on feature type in append_qk_tracing_results function
- **circuit**: support tracing in bf16 precision
- **autointerp**: comment out assertion for interpretation text in interpret_feature function
- handle 0-d tensors in feature tracing detection (#170)
- **auxk**: fix function signature for compute_loss
- **auxk**: add valid token mask for dead statistics
- **lorsa**: remove useless overide method for lorsa
- **circuit**: fix probe equiv for ln for qk norms, fix a gradient control bug
- **ui**: fix preview logic. Now consistent with tracing results
- **ui**: fix preview logic. Now consistent with tracing results
- **examples**: adapt to new from_pretrained method
- **runners**: support setting from_pretrained args
- pass device in from_pretrained

### Refactor

- **admin**: use transaction to update sae/sae set name
- **ui/circuits**: improve circuit query options and navigation handling
- **replacement_model**: enhance tokenization by passing tokenizer to ensure_tokenized function
- **replacement_model**: simplify input tokenization by utilizing ensure_tokenized function
- **ui**: remove debug console logs
- **circuit**: enhance node mapping and tracing result handling
- **graph**: remove argument_graph_file.py
- **graph**: remove debug print statements in append_qk_tracing_results function
- **graph**: reorganize graph-related classes and functions
- **attribution**: improve readability and structure in _run_attribution function
- **lorsa**: replace qk_exp_factor with ov_group_size for attention calculations
- integrate local/HF/SAELens loading into `from_pretrained` (#159)
- relocate configs and enforce import rules (#158)
- move distributed testing utils into the main library
- remove distributed state dict loading; directly use nn.Module.load_state_dict instead

## v2.0.0b6 (2025-12-30)

### Fix

- **examples**: add amp_dtype configuration
- **examples**: remove use of Path

## v2.0.0b5 (2025-12-30)

### Feat

- **ui**: support iframe
- **ui**: dictionary inference (WIP)
- **ui**: dictionary inference (WIP)
- **ui**: display activation value
- **ui**: support modify explanations
- **ui**: add bookmark page
- **ui**: add admin dashboard
- support uploading to hf and downloading from hf (#155)
- **ui**: display logits/embed tokens
- **ui**: preview next token
- **ui**: adjust new graph dialog layout
- **ui**: support applying chat template
- **ui**: display time in graph selector
- **ui**: support adding sae set
- **ui**: improve circuit visualization
- **ui**: improve circuit visualization

### Fix

- **ui**: avoid infinite useEffect in LinkGraph
- **feature_analyzer**: convert DTensor to local before calling item() in FeatureAnalyzer
- **lorsa**: fix lorsa distribute load
- **ui**: feature schema
- **ui**: bookmark count
- **server**: update circuit creation logic to handle cases without Lorsas and improve error handling

### Refactor

- **cli**: use typer to reconstruct the CLI
- **app**: split app into smaller modules
- **ui**: replace navigation with Link component for improved routing
- **ui**: manage svg rendering with react directly, instead of through d3.js; make d3.js only calculate the positions

### Perf

- **ui**: improve data persistency
- **ui**: topologically sort nodes
- **ui**: speed up (and track) graph feature retrieval
- **ui**: use rbush to speed up index; better cache results
- **ui**: add index to node/edge to improve rendering/filtering

## v2.0.0b4 (2025-12-18)

### Feat

- **analyze**: make FeatureAnalyzer aware of mask
- **ui**: hover & click nearest node
- **transcoder**: init transcoder with MLP.
- circuit tracing with backend interaction
- **optim**: add custom gradient norm computation and clipping for distributed training
- **training**: support training lorsa with varying lengths of training sequences. This will lead to total number of training tokens inaccurate (#150)
- **metrics**: add GradientNormMetric and extend Record with reduction modes
- **lorsa**: Init lorsa with the active subspace of V.
- **ui**: simply move the original circuit page to ui-ssr
- **backend**: add DTensor support to TransformerLensLanguageModel
- **circuit**: Major revision. 1. Support circuit tracing with plt+lorsa and plt only. wrap list of plts into Trancoder Set, following circuit tracer. 2. update QK tracing. Now we can see feature-feature pairwise attribution. Efficiency might require revisiting. 3. refactor attribution sturcture. Breaking down several heavy files. Ready to be further improved, mainly in reducing numerous if use_lorsa branches
- **ui**: adjust accent color
- **ui**: remove in card progress bar
- **server**: support preloading models/saes
- **ui**: feature list in feature page (WIP)
- **ui**: dictionary page
- **ui**: dictionary page (WIP)
- **ui**: interpretation with real data
- **ui**: set scrollbar-gutter to ensure space reserved for scrollbar to prevent layout shiftingg
- **ui**: support paged queries of samples
- **autointerp**: refactor to async & support lorsa
- conversion methods between lm-saes and saelens
- **train**: add checkpoint resume support for crosscoder, clt, lorsa and molt runners
- support resuming wandb run from training checkpoint
- **distributed**: add get_process_group utility, trying to fix checkpoint saving during sweeps.
- **generate**: add override_dtype setting to control activation dtype in GenerateActivationsSettings
- **autointerp**: refactor to async & support lorsa
- **analyze**: add functionality to save analysis results to file and update analysis with logits; enhance settings for output directory and feature analysis name

### Fix

- **runner**: type mismatch
- **tc**: fix type problem
- **tc**: fix type problem
- **trainer**: remove assertion for clip_grad_norm in distributed training
- **attribution**: add details to some comments
- **ui**: minor layout issues
- **ui**: correctly display truncated z pattern
- **ui**: better display dead feature
- **backend**: use TokenizerFast for trace token origins
- **server**: synchronized decorator type issue
- **misc**: we do not want to filter out eos. It might be end of chats and include useful information
- compute_loss DTensor loss shape
- **training**: also transform batch['mask'] to Tensor from DTensor in… (#152)
- **lorsa**: avoid triggering DTensor bug in torch==2.9.0
- **lorsa**: fix set decoder norm for lorsa
- **lorsa**: fix lorsa init
- **server**: expose lru_cache ability from synchronized decorater
- **attribution**: fix missing gradient flow configuration for lorsa QKnorm (#149)
- **optim**: add DTensor support for SparseAdam, redistribute grad to match parameter's placements when grad is DTensor
- **ui**: fix visible range comparison
- **metric**: support inconsistent batch size
- **ui**: fix feature list height
- **ui**: reinitialize useFeatures hook when concerned feature index out of range
- **ui**: feature list loading previous page causes wrong scroll position
- pin torch==2.8.0 for dtensor compatibility. - Pin torch version to 2.8.0 to avoid dtensor-related errors in 2.9.0 - Remove unused d_model field from LanguageModelConfig - Add GPU memory usage display in training progress bar - Move batch refresh to end of training loop iteration
- **database**: deal with none value
- **TL**: add support for whole qwen3 family & fix inconsistency in tie-word-embed
- type errors due to torch updates on local_map
- **trainer**: correct token count calculation for 2D activation in LORSA training
- **trainer**: use ctx.get() for optional coefficients to prevent KeyError
- **activation**: use local_map for mask computation on DTensor to ensure correct device placement
- **activation**: make mask/attention_mask on the correct device
- replace `.item()` with `item(...)` to ensure distributed consistency.
- **basedpyright**: exclude scripts folder
- basedpyright & ruff issues
- fix sweep SAE distributed training and convert Path to str in training configs
- **log**: use torch.any to detect inf value in total_variance_mean, to fix crosscoder case.
- **crosscoder**: make tokens DTensor in analysis
- **log**: fix incorrect log_info update
- ensure WORLD_SIZE is correctly compared as an integer in training scripts
- **prepend_bos**: fix shape unmatch
- **distributed**: update broadcast_object_list calls to use group_src parameter
- **batchtopk**: fix batchtopk in dp mode
- **distributed**: fix the sort of import
- **analyze**: improve implements for list operations
- **analyze**: define functions common to Tensor and DTensor, improve implements for list operations
- basedpyright issues (WIP)
- **analysis**: fix tp for analyze_chunk

### Refactor

- **ui**: remove standalone CircuitVisualization component
- **ui**: split data and visual states; move up feature fetching logic
- use tanstack start for frontend; make a more neuronpedia-like ui (#146)
- use Metric classes to run evaluation (#133)
- use Metric classes for disentangled metric computation
- use TensorSpecs rather than logging method dispatch for logging with different SAE variants (#130)

### Perf

- **ui**: better visual display for circuit (WIP)
- **ui**: fetch sample range on demand

## v2.0.0b2 (2025-11-04)

## v2.0.0b1 (2025-11-04)

### Feat

- **crosscoder**: add log for (non) activated decoder norms in activated feature
- add tanh-quad frequency_scale to TrainerConfig
- circuit tracing + z_pattern
- general loading for HuggingFace models
- add z_pattern
- add z_pattern
- **dla**: add DLA test for clt and lorsa
- **circuit**: add intervention for replacement model; add tests for attribution and intervention.
- update ev compute
- **analyze**: support molt analysis
- **trainer**: add per rank_group logging logic to molt
- **molt**: implement distributed molt
- **molt**: most of molt should be done right
- **molt**: a staged version of molt
- **activation**: chunk assignment in tp+dp setting
- **analysis**: reimplement DirectLogitAttributor and related configurations
- **analysis**: reimplement DirectLogitAttributor and related configurations
- **feature_analyzer**: implement DDP support for feature analyzer. Add DDP capabilities to the feature analyzer. Note: TP compatibility is not guaranteed and may require additional work.
- **feature_analyzer**: add mask ratio statistics update and logging
- **api**: add metric filtering and retrieval for features
- **language_model**: enhance activation processing and configuration
- **language_model**: enhance LLaDA model integration and preprocessing
- **activation**: support checking activation consistency
- **activation**: implements distributed loading across tp dimension
- **activation**: implements distributed loading across tp dimension
- **TransformerLens**: fix precision mismatch for ln hook_normalized for rms norm
- **trainer**: support param groups
- **l_p**: add pre-act loss for jumprelu and fix some bugs in sae.py and lorsa.py
- **qk trace**: implement compute_attention_scores_attribution in attribution
- autointerp of neuronpedia
- add autointerp and logits to graph json
- DLA for lorsa and clt; autointerp for lorsa and clt.
- improve lorsa and clt training
- **topk**: add conversion to jrelu
- **circuit; frontend**: add support to show feature card in linkgraph
- **ui**: update circuit tracing (not done but quiet close)
- **kernels**: support encoder bacakward acceleration kernels
- **clt**: add binary search for singleGPU and fix some bug regarding kernel opening logic
- **kernels**: add a shitty implementation of encode with triton kernel
- **clt**: add a binary_search method to accelerate batchtopk
- **clt**: improve batchtopk by divide and conquer; precision and efficiency will be done later
- **clt**: major change. clt done right. will fix dirty parts later
- **activation**: implements distributed loading across tp dimension
- **activation**: implements distributed loading across tp dimension
- **TransformerLens**: fix precision mismatch for ln hook_normalized for rms norm
- **config**: add prepend_bos option to LanguageModelConfig
- **timer**: integrate timer to activation writer
- **evaluator**: add evaluation functionality for  CrossCoders
- **trainer**: log current l1 coefficient
- **trainer**: remove l0_based decoder weight learning rate
- **config**: add LLaDA model configuration and implementation
- **clt**: start to seem good in training. fix all bugs in fwd and bwd (temp)
- **clt**: passed all fwd test in distributed settings. requires training on real data
- **trainer**: add update_decoder_lr_with_l0 flag and enhance learning rate adjustment
- **trainer**: add expected_l0 parameter and update decoder learning rate dynamically
- **abstract_sae**: enhance JumpReLU with precision promotion
- **crosscoder**: support head parallelism with world size < head number
- **logging**: implement centralized logging system across modules
- **vis**: enhance feature retrieval and analysis handling
- **bookmarks**: implement bookmarking functionality for features
- **abstract_sae**: enhance state dict handling in AbstractSparseAutoEncoder
- **feature-interpretation**: enhance feature interpretation component and update API calls
- **abstract_sae**: add support for DCP checkpoint format in save/load methods
- **trainer**: add per-head logging (ev & l0) for crosscoder
- **interpreter**: improve automatic feature interpretation with multiprocessing
- **timer**: integrate timing functionality into key methods for performance monitoring
- **interpreter**: support for auto interp without loading LLM ckpt
- **interpreter**: enhance feature interpretation with optional interpretation field and parallel processing
- **feature**: add analysis retrieval and enhance feature details in UI
- **sae**: enhance parameter loading and device management in JumpReLU and CrossCoder
- **crosscoder**: distribute tensor while loading
- **config**: add non-activating subsample parameters and enhance FeatureAnalyzer with non-activating example sampling
- **crosscoder**: support inputs of different shapes on each head
- **sae**: support tanh-quad loss
- decouple batching and activation 1d
- **analysis**: support converting from crosscoder head parallel to analyzer feature dimension parallel
- implement autointerp
- **crosscoder**: add distributed support
- **ui**: visible segments
- feature analysis for crosscoder
- share activation factory in sweep experiment
- **crosscoder**: reimplement crosscoder to internalize n_heads
- **mixcoder**: fix bugs when using apply_decoder_bias_to_pre_encoder
- **visualization**: support act frequency
- **analysis**: record analyzed token count
- **backend**: re-support TransformerLens models
- **visualization**: add modality specific info
- **analysis**: add modality-specific metrics
- **language_model**: set padding method to max_length
- **analysis**: support mixcoder
- support multi-lingual mixcoder
- **trainer**: add more loginfo for mixcoder training
- **backend**: support qwen2.5 base
- **language_model**: support qwen2.5 vl with hf_backend
- **backend**: add language model base class
- **kernels**: support spmm triton kernel for topk greater
- **kernels**: support spmm triton kernel for topk greater
- **kernels**: support spmm triton kernel for topk saes. Topk SAEs do not require precise gradients passed to feature acts so acceleration can be greater
- **mixcoder**: changed loss calculation method and added more log in trainer
- **spmm_decode**: implement sparse mm decode few & bwd with triton
- **anthropic jumprelu**: ready for training
- **anthropic jumprelu**: training is ready
- **anthropic jumprelu**: initial implementation
- **runner**: support train with non-pre-generated activations
- **trainer**: add some new extra log info for mixcoder
- **trainer**: add some new extra log info for mixcoder
- **trainer**: add extra log info for mixcoder
- **runner**: support mixcoder training (#78)
- **activation**: add tqdm in loading cached activation
- **mixcoder**: implemented mixcoder
- **sae**: support saving/loading dataset_average_activation_norm to/from SAE state dict
- **sae**: change input format of forward method
- **entrypoint**: support train/analyze runner and create/remove dataset record
- **dataset**: support removing analysis & sae
- **analysis**: remove internal batching
- **activation**: support writing activations without batching
- **runner**: add AnalyzeSAERunner
- **config**: automatically load model & dataset config from database
- **activation**: batched model generation
- **activation**: add some tqdms to monitor activation generation
- **activation**: add model_name to activation meta
- **runner**: add train sae runner with TrainSAESettings; update SAE initialization and training logic
- **activation**: support load from cached activatio in ActivationFactory
- **config**: remove FlattenableModel inheritance
- **entrypoint**: support generate activations
- **runner**: generate activations
- **activation**: add start shard
- **activation**: support ddp activation generation
- add ActivationWriter
- **activation**: rename "info" to "meta"
- **activation**: add CachedActivationLoader
- complete ActivationFactory
- **activation**: implement basic activation pipeline

### Fix

- basedpyright issue (WIP)
- **backend**: unify to tokens function
- basedpyright issue (WIP)
- TL
- crosscoder training
- lorsa post qk init
- **crosscoder**: dp gradient should be partial
- import
- **crosscoder**: eliminate the impact of activation function from f(x)/x to f(x)
- **abstract_sae**: use local_map to conduct mean reduction in tanh-quad computation
- import
- attribution graph and intervention pnly for CLT. and ruff
- attribution graph and intervention pnly for CLT.
- attribution
- analyze for SAE
- ruff
- analyze for clt
- **trainer**: correctly apply jumprelu lr; make new ev appliable to crosscoder
- **circuit**: change absolute path
- **train**: support sae for data prallel
- **molt**: fix a major bug in _decode_distributed and now dist molt is done right
- **molt**: fix a init bug in dist molt
- **trainer**: adapt log_info for situations where l_s doesn't exist
- **runner**: dp runner
- **activation**: tensor parallel
- recycle import and jumprelu pre-act loss
- **activation**: adjust total count for activation processing in CachedActivationLoader
- **activation**: fix batch size validation for data parallelism
- **distributed**: fix several dp placements errors. revert the CachedActivationLoader to load the entire chunk.
- **distributed**: enhance data parallelism support
- **writer**: avoid including activations from other hook points during chunk data handling in ActivationWriter and improve assertion clarity in CachedActivationLoader.
- **activation**: correct chunk buffer condition in CachedActivationLoader.
- **ci**: fix dependency installation
- **ci**: ignore all import errors; temporarily remove unit tests checking; fix ci dependency installation
- **activation**: update device_mesh type hint and improve DistributedSampler usage
- **trainer**: remove logics about n_forward_passes_since_fired
- code comments. del useless file.
- autointerp of neuronpedia about 'say ...' feature
- autointerp of neuronpedia
- circuit trace clt error node idx
- **clt; lorsa**: minor fixes on scalable training
- **ui**: fix bugs in lorsa zpattern
- **circuit**: minor improvements in attribution
- **kernels**: return a minimal zero tensor as grad in case no feature is activated for some rank
- **clt**: replace isinstance(,torch.sparse.Tensor) with torch.is_sparse
- **math**: batch_kthvalue_clt_binary_search donot need grad
- **misc**: fix a misalignment bug in 7fd72fc  introduced by 7fd72fc
- **activation**: update device_mesh type hint and improve DistributedSampler usage
- **trainer**: remove logics about n_forward_passes_since_fired
- **misc**: remove abundant mixcoder logics in config; bring back time in abstract_sae
- type errors
- **language_model**: enable trust_remote_code in model loading
- **sae**: remove sae data parallel runner
- **sae**: placements of norms
- **sae**: add device_mesh parameter to __init__
- **trainer**: ensure metrics logging only occurs on primary rank
- **bookmarks**: correct query parameter in bookmarks feature link
- **abstract_sae**: correct tensor shape initialization in state dict
- **trainer**: update crosscoder logging keys to use hook points instead of batch keys
- **trainer**: logs timer stats when enabled
- **abstract_sae**: correct indentation for label tensor conversion
- **pyproject.toml**: fix torch version
- **crosscoder**: correct slice calculation for local rank in tensor distribution
- **trainer**: move compile to training step
- **analysis**: insert analysis only once
- **backend**: minor type errors
- **activation**: generate activation w.r.t. specific context size
- **crosscoder**: various issues in crosscoder head parallel
- **crosscoder**: fix distributed training
- **analysis**: add start_idx in tp analysis
- minor compatibility issues with crosscoder
- replace SparseAutoEncoder with AbstractSparseAutoEncoder
- **initializer**: normalize activations in initialization search
- **visualization**: typos
- **language_model**: add support for multiple images in single input
- **server**: batchify raw data for tracing
- fix some bugs in training and analysis
- **backend**: padding & truncation
- **mixcoder**: pass in modalities to encode & decode
- **activation**: transform & batchify all tensor field
- **activation**: intercept activation source
- **activation**: generate activation in no_grad mode
- image format from HuggingFace dataset
- **server**: correctly retrieve image
- **kernel**: fix speed degradation of TopK kernel
- **cached_acts**: re-implement changes mistakenly removed in 71ff9f9
- **kernels**: fix bugs in kernel tests
- **server**: correctly retrieve non-sharded dataset
- **activation**: preserve tokens type during dtype conversion
- **activation**: preserve tokens type during dtype conversion
- **activation**: preserve tokens type during dtype conversion
- **analyze**: misc things stopping analysis working
- **trainer**: convert modality_ev from bf16 to fp32
- **kernel**: fix type error
- **sae**: fix type error
- **cc**: fix cc inheriting logics of activation func
- **tests;ruff**: remove unused variables and functions. fix errors in tests
- **cached_activation**: fix errors in implementation of overriding cached activations
- **triton**: add triton in pyproject
- **fp32 threshold**: enforce jumprelu threshold uses fp32
- **fp32 threshold**: enforce jumprelu threshold uses fp32
- **mixcoder**: fix topk activation func
- **config**: fix lr warmup & cooldown step default value
- **config**: set prefetch in ActivationFactoryActivationsSource to be optional
- **training**: fully utilize GPU cores with torch dataloaders for CachedActivations
- **test_sae**: fix ground truth for sae fwd when sae does not apply dec bias to pre dec by default
- **crosscoder**: fix minor bugs and logs in cc training
- **crosscoder**: fix minor bugs and logs in cc training
- **crosscoder**: fix minor bugs and logs in cc training
- **ui**: minor type error
- **topk activation**: add keepdim=True to enable broadcasting; make d… (#73)
- **runner**: fixed the issue where the wandb logger was not properly initialized during SAE training
- **misc**: fix calculate_activation_norm method
- **entrypoint**: load dataset & model config
- **server**: database interaction
- **analysis**: early skipping condition
- **activation**: inconsistent runner & component config
- master node condition
- module export
- **activation**: remove meta requirements for cached activation
- **activation**: mask tokens
- **trainer**: fix warmup process of lr & topk
- **sae**: fix the key error of sae.dataset_average_norm
- **config**: set default ActivationFactoryDatasetSource.prepend_bos to False
- **sae**: fix some bugs, log device_mesh in SAE now, and add trainer tests
- **database**: add sae path
- **trainer**: fix type error
- **activation**: record shard info in metadata
- **resouce_loader**: num of shards when specifying start_shard
- **activation**: pipeline order
- **activation**: misc issues in generating activations
- **SparseAutoencoder**: overlapping method sign in overloading compute_loss

### Refactor

- move model-specific logs to model class
- **molt**: remove sparisty score and activation func acts directly on hidden_pre
- **utils**: remove unused util modules; reorganize distributed module
- **abstract_sae**: prepare_input should now also return decoder_kwargs
- **molt**: refactor rank assignments logic for disentangling dist inference and dist training
- **molt**: refactor decode into einsum oprations
- **molt**: refactor decode and achieve a comparable efficiency to transcoder
- **molt**: refactor decode for a better effciency and vram usage
- **molt**: refactor rank distribution logic and a fix a bug in decode einops
- **activation**: improve distributed loading logic and tensor gathering
- **writer**: streamline chunk data structure in activation writer to include extra information.
- **activation**: simplify activation output structure and type hints
- **clt**: disentangle distributed and single-gpu logic of encode and decode
- **kernels**: move sparsity check into kernels for decoder regarding the unequal division of activations across devices
- **kernels**: move sparsity check into kernels regarding the unequal division of activations across devices
- **kernels**: remove vmap implementation for encoder kernels
- **distributed**: remove heuristic topk implementation which is no longer needed
- **timer**: enhance TimerNode and Timer structure for hierarchical path management
- **resource_loaders**: update imports for LLaDALanguageModel
- **language_model**: rename LLaDA class to LLaDALanguageModel
- **timer**: enhance hierarchical timer functionality
- **clt**: remove debug print statements from CrossLayerTranscoder
- **initializer**: streamline initialization logic for encoder norms
- **sae**: directly manipulate DTensor parameters
- **initializer, runners**: remove MixCoder references and clean up code
- **entrypoint**: update import paths for runner modules
- **initializer**: remove constant fire times bias initialization and rename method
- **analysis**: update import for feature interpretation module
- **autointerp**: remove auto_interp module and integrate functionality into runner
- **distributed**: replace dictionary-based dimension maps with DimMap class
- **runners**: reorganize runner module and update imports
- decouple hook_point_in/hook_point_out with BaseSAEConfig
- explicitly normalizing activations
- move public interface of SparseAutoEncoder into AbstractSparseAutoEncoder
- **activation**: directly get activation from raw data
- **activation**: directly get activation from raw data
- **activation**: replace HookedTransformer with LanguageModel
- **activation**: reorganize Dataloader-based cached activation reading
- **sae**: decouple encoder & decoder from sae methods
- **server**: comply to new api
- **analysis**: replace sample_feature_activation with FeatureAnalyzer
- **config**: compute BaseSAEConfig.d_sae on the fly; set default for BaseSAEConfig.hook_point_out based on BaseSAEConfig.hook_point_in
- **database**: drop GridFS; use a 3-level architecture (sae, analysis, feature); use pydantic to parse database retrieval result
- **config**: use pydantic models
- **trainer**: fix bugs in sae and initializer and add codes for trainer
- **activation**: add low level processors of ActivationPipeline

### Perf

- **abstract_sae**: reduce tanh-quad memory cost in dp by manually execution of DTensor mean
- **crosscoder**: put data in the first mesh dim
- **crosscoder**: add encoding and decoding methods with optional einsum support
- **vis**: refactor feature components for improved performance and usability
- **crosscoder, analysis**: improve tensor redistribution strategy to reduce GPU memory usage
- **abstract_sae, crosscoder**: fully distributed initialization & redistribute accumulated_hidden_pre in crosscoder encode
- **trainer**: remove unnecessary tensor gathering
- **crosscoder**: directly initialize parameters and parallelism in target devices
- **trainer**: compile the model
- **analysis**: change distribute_tensor to DTensor.from_local for non-leaf tensor
- **runner**: remove useless rebatching
- **runner**: logging load/save config from/to database
- **activation**: support parallel writing activation
- **activation**: support parallel & background loading cached activation

## v1.1.0 (2024-12-25)

### Feat

- **ui**: add showImageHighlights and showImageGrid buttons
- **ui**: show sample with origins
- **ActivationSource**: filter pad token from activation dataset
- **server**: set field feature_acts_all in feature as optional
- **analysis**: refactor token source & analysis
- **ActivationSource**: support cached multi-dataset activation
- support tracing token origins
- **act gen**: add option to center a batch of activation generated

### Fix

- **ActivationSource**: remove act from the buffer after taking it out of the buffer
- **server**: load datasets & model
- **ActivationSource**: set correct device in loading cached activation
- generate activations
- types and format
- **server**: steering
- minor type issues

### Refactor

- make all types work with basedpyright

## v1.0.0 (2024-11-01)

### Fix

- **example**: remove training examples from an older version
- **example**: fix error in loading example
- **server**: specify env file for starting server
- **tensor_parallel**: avoid passing during_init to indicate pre-tp condition
- **example**: remove device-specific path
- **mypy**: fix mypy issues
- **example**: fix wrong default params for llamascope

## v0.1.0 (2024-09-12)

### Feat

- **ui/circuit**: show attention pattern
- **server**: catch oom error
- **server**: trace attention score
- **ui/circuit**: trace attention score; remove existed tracing
- **ui/circuit**: show information of the selected node/edge
- **server**: update tracing return schema
- **ui/circuit**: trace intermediate nodes
- **server**: add tracing api
- **ui/model-page**: add basic circuit
- **sae_training**: change save ckpt interval to log scale
- **ui/model-page**: steering
- **ui/server**: model generate with sae and steering
- **ui/model-page**: add detail of selected tokens
- **ft4supp**: support ft4supp adjusted for AprilTrick update SAEs
- **sae**: add a utils func to merge pre-enc bias into enc bias
- **ft4supp**: support ft4supp adjusted for AprilTrick update SAEs
- **ui/model-page**: generation section
- **ui/model-page**: create model generation interface
- **ui**: create model page
- **analysis**: support tensor parallel
- **sae**: add a utils func to merge pre-enc bias into enc bias
- **ui**: update style of section navigator
- **model**: support llama3_1
- **model**: support llama3_1
- **config**: add decay ratio
- **config**: support warmup step set to a proportion of overall steps
- fix type error
- offload LLM parameters after last hook, and support warm up/cool down ratio in config
- **sae**: Implement ckpt saving in tensor parallel environment.
- Implement tensor parallelism in SAE using device mesh
- **circuit**: add specific functionality for attributing transformers
- **circuit**: add basic attributors
- **HookedRootModule**: add mount_hooked_modules
- **runner**: load tokenizer manually
- **runner**: load tokenizer manually
- **entrypoint**: add entry point for lm_saes
- **HookedRootModule**: implement run_with_cache_until & remove fake tensors
- **HookedRootModule**: fix & add test case for fake tensors
- **training**: support buffer filling from multiple data sources and configure pack and sample probability for each dataset
- **sae**: implement from_pretrained from huggingface hub
- change ckpts to safetensors; decouple RunnerConfig; add from_pretrained for local parameters
- **server**: support all TL models in server app
- **runner**: replace hardcoded 'gpt2' with cfg.model_name variable
- **analysis**: accelerate analysis with chunked d_sae and stop_at_layer and a pre-check before sorting
- **hook_points**: Enable early stopping by converting parameters to fake tensors
- **runner**: replace hardcoded 'gpt2' with cfg.model_name variable
- **transformer_lens**: add ref cache
- add support for bf16
- **circuit**: minor changes
- **runner**: add model_from_pretrained_path param
- **circuit**: contribution graph
- **analysis**: support analysis on self-trained models
- **visualizer**: display token position
- **db**: remove dictionary
- **sae**: unbind sae input and label
- **core**: support analysis and visualization for self-trained models
- **sae**: support training saes in local models
- **visualizer**: auto fetch dictionary
- **visualizer**: add navbar
- **visualizer**: multiple dictionary samples
- **visualizer**: dictionary sample display
- **visualizer**: dictionary custom input
- **visualizer**: add feature logits
- **visualizer**: add feature logits
- **visualizer**: attn score
- **stats**: compute attn score
- **visualizer**: auto interp
- **auto_interp**: optimize logic and improve code readability
- **autointerp**: implement automatic interpretation
- **visualizer**: search params for dictionary and feature index
- **visualizer**: feature activation histogram
- **visualizer**: pagination
- **visualizer**: custom input
- **visualizer**: subsample
- **visualizer**: fetch random living feature
- **FeatureActivations**: subsample
- **visualizer**: add result dir
- enable non-strict loading of model state dicts
- sae pruning
- record wandb id
- adjust result saving structure
- **visualization**: dictionary selection
- FastAPI server for feature visualization
- **sample_feature_activations**: save feature index
- save feature activations as huggingface dataset
- **sample_feature_activations**: compute feature act hists
- save feature acts with datasets & compute feature act bins
- **TokenSource**: directly remove bos token
- **TokenSource**: disable concating tokens
- add glu encoder bias
- add gau encoder
- batch-wise act norm
- **scheduler**: add cool down
- add eval runner
- **evals**: explained variance and l0
- **SAE**: config to remove decoder bias
- count useful feature
- support non-exactly decoder norm
- remove thomson potential
- exponential warmup scheduler
- sample feature activations
- add config for lp and Adam betas
- load checkpoint
- load hf model from local files
- **ActivationSource**: load cached activations
- remove unused field when generating activations
- generate activations on disk
- add cache_dir config
- add l2 norm error metric
- update activation source
- activation source
- switch to their activation store

### Fix

- **server**: trace sae feature
- **eval**: not offload params on eval
- **config**: fix linear saving mode bugs and mypy issues
- **runner, server**: use HookedTransformer.from_pretrained_no_processing under all circumstances
- **TransformerLens**: fix mlp dtype and missing attn code
- **sae**: support grid searching for best init
- **ui/sample**: fix folded start
- **runner**: add accidentally missing from_init_searching
- **runner**: add accidentally missing from_init_searching
- **runner**: set offload after the last hook (previously it was the first)
- **textdataset**: set default prepend_bos to True
- **textdataset**: set default prepend_bos to True
- **ft4supp**: supp final ver.
- **misc**: remove unnecessary changes
- **sae**: do not init device mesh in single device mode
- **ui/model-page**: model generation
- **sae**: move transform decoder_norm to save_pretrained
- **sae**: transform decoder_norm and encoder_norm to dtensor under tensor parallel settings
- **sae**: do not init device mesh in single device mode
- **training**: stable GPU usage
- **training**: stable GPU usage
- **training**: stable GPU usage
- **runner**: remove redundant code
- **training**: stable training
- **training**: change back to clip grad norm
- **ui**: import error
- **ui/preview**: add color to "..." to avoid confuse
- **ui/preview**: disable flex to avoid showing long token
- **ui/preview**: assign unique value to accordion item; fix(ui/preview): enable showing 10 samples within one page
- **runner**: remove redundant code
- **training**: use clip grad value instead of norm
- **sae**: fix post process in transform_to_unit_decoder_norm
- **typing**: fix mypy issues
- **config**: ignore mypy checking for norm init.
- **sae**: merge a standalone SAE init func with static method
- typo error
- typo
- typo
- **sae**: fix transform_to_unit_decoder_norm in tensor parallel
- **prune_sae**: support bfloat16
- convert decoder bias to local tensor while using tensor parallel
- fix some bugs encountered during the initialization of SAE and the retrieval of next token in a tensor parallel environment.
- **sae**: fix merge bugs
- fix bugs of prepend bos during eval and sampling  (#30)
- **evals**: performs logits mask when computing ce score. Ignoring pad tokens
- **activation gen**: remove bos. ce score is greatly improved
- resolve DDP-related synchronization bug
- **HookPoint**: use register_full_backward_hook in bwd hooks
- **sae_training**: support bf16
- **frontend**: implement a byte-to-unicode function instead of using the hf implementation in tokenizer
- **examples**: fix programmatic runners
- type checking workflow
- replace deprecated import from core package with lm_saes package
- **runner**: add dtype parameter during model initialization
- **analysis**: optimize GPU memory usage after finishing each chunk
- **analysis**: fix numpy to list in runner.py
- **analysis**: optimize GPU memory usage after finishing each chunk
- **activation**: remove context info in activation_source
- **runner**: add dtype parameter during model initialization
- **sae**: set default use_decoder_bias to False
- **sae**: bring back decoder bias
- **SAE**: output non-normalized aux data in compute_loss
- remove decoder bias
- **visualizer**: load model
- **visualizer**: load sae
- **visualizer**: preserving leading & trailing space of tokens
- **visualizer**: update token group indices on samples changed
- **visualizer**: minor style fixes
- minor bugs in database io
- **runner**: minor bugs
- **database**: minor bugs
- **visualizer**: prevent fetchFeature with empty searchParams
- **analysis**: disable feature_acts gathering when subsampling
- **SAE**: apply decoder bias to x_hat at correct place
- **FeatureActivations**: minor bugs
- **notebooks**: use correctly pruned sae
- **sae**: apply feature activation mask & scale
- **prune_sae**: assign mask
- **SAE**: parameter groups
- pack tokens which cannot be directly decoded
- **server**: minor bugs
- **TokenSource**: skip tokens with inadequate seq_len
- feature activation notebook
- **sample_feature_activations**: analyzing steps
- **TokenSource**: disable concating tokens
- **ActivationStore**: shuffle activations
- **SAE**: batch wise act norm
- **eval**: ev computation
- **SAE**: reset decoder norm
- minor bugs in eval
- **sample_feature_activations**: rand distribution
- exponential warmup
- **sample_feature_activations**: nan elt value
- sample feature activations
- load hf model
- **eval**: remove attention mask in computing ce loss
- **activation_dataset**: chunk size
- minor bugs in generate_activations
- use default_factory to create default list
- run name

### Refactor

- **ui/circuit**: use tracings as circuit props
- **ui**: integrate functionalities into Sample component
- **config**: merge exp_name into exp_result_dir as exp_result_path and added a path field to the database
- **circuit**: decouple attributor with saes
- **config**: change to compositional config
- **analysis**: move feature direct logits contribution to core.analysis
- **visualizer**: move database module to core
- **analysis**: analysis to database
- migrate result to MongoDB
- **visualizer**: FeatureCard
- sample feature activations

### Perf

- **ui/circuit**: select edges
- comply with strict mypy typing
- early stop activation caching with run_with_cache_until
- **activation_source**: disable gradients during inference
- **SAE**: decouple encoding and decoding procedure of SAE
- **training**: merge finetuning into training process
- **visualizer**: change plotly style
- remove unused config
- **ft4supp**: remove unused code & save hyperparams
- **visualizer**: hint for loading dictionary first time
- **visualizer**: minor style changes
