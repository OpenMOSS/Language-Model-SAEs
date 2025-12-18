# Example setups of Language-Model-SAEs

The standard SAE-based pipeline of mechanistically interpreting internal representations of language models contains the following steps: Generating activations (optional) -> Training SAEs -> Analyzing SAEs -> Visualizing analyses.

Here present example setups of generating, training and analyzing, with variants of SAE architectures, activation functions and whether to use pre-generated activations.

## Use on-the-fly model activations

SAE training requires stream of model activations at certain hook points (i.e. specefic location of model internal representation). Model activations can either be cached ahead-of-time on the disk, or produced on the fly. 

For on-the-fly model activation usage, the _Generating activations_ step can be skipped, and thus the overall pipeline is simplified. You can refer to [train_pythia_sae_topk](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/train_pythia_sae_topk.py) and [analyze_pythia_sae](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/analyze_pythia_sae.py) and other scripts without a `with_pre_generated_activations` suffix to launch the experiments on Pythia. Note the analyzing requires a MongoDB instance (default to `mongodb://localhost:27017`) running to save the analyzing results.

## Use cached activations

Cached activations are more common usage in practical SAE training and analyzing. It enables effective hyperparameter sweeping with reuse of generated activations, and also enables parallelled training and analyzing (DP/TP). However, it requires a non-trivial amount of disk space, e.g., caching 800M tokens of one layer activation of Pythia 160M requires about 6TB space.

To launch experiments with cached activations, you should first generate activations with 1d shape (`(batch, d_model)`, for training use), and 2d shape (`(batch, n_context, d_model)`, for analyzing use), by running [generate_pythia_activation_1d](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/generate_pythia_activation_1d.py) and [generate_pythia_activation_2d](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/generate_pythia_activation_2d.py). Then, you can use [train_pythia_sae_with_pre_generated_activations](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/train_pythia_sae_with_pre_generated_activations.py) and [analyze_pythia_sae_with_pre_generated_activations](https://github.com/OpenMOSS/Language-Model-SAEs/blob/main/examples/analyze_pythia_sae_with_pre_generated_activations.py) to run training and analyzing respectively, with a pre-generated activation path specified. Note the analyzing still requires a MongoDB instance running.