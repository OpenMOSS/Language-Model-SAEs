import os
from typing import Annotated, Any, Literal, Union, cast

import numpy as np
from pydantic import BaseModel, Field
import torch

from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_gpt2_weights

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import plotly.express as px
import plotly.graph_objects as go

from lm_saes.analysis.auto_interp import check_description, generate_description
from lm_saes.circuit.context import apply_sae, detach_on
from lm_saes.config import AutoInterpConfig, LanguageModelConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.bytes import bytes_to_unicode
from lm_saes.utils.hooks import detach_hook

result_dir = os.environ.get("RESULT_DIR", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"), os.environ.get("MONGO_DB", "mechinterp"))
dictionary_series = os.environ.get("DICTIONARY_SERIES", None)

sae_cache = {}
lm_cache = {}


def get_model(dictionary_name: str) -> HookedTransformer:
	path = client.get_dictionary_path(dictionary_name, dictionary_series=dictionary_series)
	if path is "":
		path = f"{result_dir}/{dictionary_name}"
	cfg = LanguageModelConfig.from_pretrained_sae(path)
	if (cfg.model_name, cfg.model_from_pretrained_path) not in lm_cache:
		hf_model = AutoModelForCausalLM.from_pretrained(
			(
				cfg.model_name
				if cfg.model_from_pretrained_path is None
				else cfg.model_from_pretrained_path
			),
			cache_dir=cfg.cache_dir,
			local_files_only=cfg.local_files_only,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			(
				cfg.model_name
				if cfg.model_from_pretrained_path is None
				else cfg.model_from_pretrained_path
			),
			trust_remote_code=True,
			use_fast=False,
			add_bos_token=True,
		)
		model = HookedTransformer.from_pretrained_no_processing(
			cfg.model_name,
			device=device,
			cache_dir=cfg.cache_dir,
			hf_model=hf_model,
			tokenizer=tokenizer,
			dtype=hf_model.dtype,
		)
		model.eval()
		lm_cache[(cfg.model_name, cfg.model_from_pretrained_path)] = model
	return lm_cache[(cfg.model_name, cfg.model_from_pretrained_path)]


def get_sae(dictionary_name: str) -> SparseAutoEncoder:
	path = client.get_dictionary(dictionary_name, dictionary_series=dictionary_series)['path'] or f"{result_dir}/{dictionary_name}"
	if dictionary_name not in sae_cache:
		sae = SparseAutoEncoder.from_pretrained(path, device=device)
		sae.eval()
		sae_cache[dictionary_name] = sae
	return sae_cache[dictionary_name]


def make_serializable(obj):
	if isinstance(obj, torch.Tensor):
		return obj.tolist()
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, dict):
		return {k: make_serializable(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [make_serializable(v) for v in obj]
	return obj

@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
	return Response(content=str(exc), status_code=400)


@app.get("/dictionaries")
def list_dictionaries():
	return client.list_dictionaries(dictionary_series=dictionary_series)


@app.get("/dictionaries/{dictionary_name}/features/{feature_index}")
def get_feature(dictionary_name: str, feature_index: str | int):
	tokenizer = get_model(dictionary_name).tokenizer
	if isinstance(feature_index, str):
		if feature_index == "random":
			feature = client.get_random_alive_feature(dictionary_name, dictionary_series=dictionary_series)
		else:
			try:
				feature_index = int(feature_index)
			except ValueError:
				return Response(
					content=f"Feature index {feature_index} is not a valid integer",
					status_code=400,
				)
	if isinstance(feature_index, int):
		feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)

	if feature is None:
		return Response(
			content=f"Feature {feature_index} not found in dictionary {dictionary_name}",
			status_code=404,
		)

	sample_groups = []
	for analysis in feature["analysis"]:
		samples = [
			{
				"context": [
					bytearray([byte_decoder[c] for c in t])
					for t in tokenizer.convert_ids_to_tokens(analysis["contexts"][i])
				],
				"feature_acts": analysis["feature_acts"][i],
			}
			for i in range(len(analysis["feature_acts"]))
		]
		sample_groups.append(
			{
				"analysis_name": analysis["name"],
				"samples": samples,
			}
		)

	feature_activation_histogram = px.histogram(feature["feature_acts_all"], width=600, nbins=50)

	feature_activation_histogram = go.Histogram(
		x=feature["feature_acts_all"],
		nbinsx=50,
		hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
		marker_color="#636EFA",
		showlegend=False,
	).to_plotly_json()

	if "logits" in feature:
		logits_bin_edges = np.array(feature["logits"]["histogram"]["edges"])
		logits_histogram = go.Bar(
			x=(logits_bin_edges[:-1] + logits_bin_edges[1:]) / 2,
			customdata=np.dstack([logits_bin_edges[:-1], logits_bin_edges[1:]]).squeeze(),
			y=np.array(feature["logits"]["histogram"]["counts"]),
			hovertemplate="Count: %{y}<br>Range: %{customdata[0]} - %{customdata[1]}<extra></extra>",
			marker_color=["#EF553B" for _ in range((len(logits_bin_edges) - 1) // 2)] + ["#636EFA" for _ in range(
				(len(logits_bin_edges) - 1) // 2)],
			showlegend=False,
		).to_plotly_json()

	return Response(
		content=msgpack.packb(
			make_serializable(
				{
					"feature_index": feature["index"],
					"dictionary_name": dictionary_name,
					"feature_activation_histogram": [feature_activation_histogram],
					"act_times": feature["act_times"],
					"max_feature_act": feature["max_feature_acts"],
					"sample_groups": sample_groups,
					"logits": {
						"top_positive": list(reversed(feature["logits"]["top_positive"])),
						"top_negative": feature["logits"]["top_negative"],
						"histogram": [logits_histogram],
					} if "logits" in feature else None,
					"interpretation": feature["interpretation"] if "interpretation" in feature else None,
				}
			)
		),
		media_type="application/x-msgpack",
	)


@app.get("/dictionaries/{dictionary_name}")
def get_dictionary(dictionary_name: str):
	feature_activation_times = client.get_feature_act_times(dictionary_name, dictionary_series=dictionary_series)
	if feature_activation_times is None:
		return Response(
			content=f"Dictionary {dictionary_name} not found", status_code=404
		)
	log_act_times = np.log10(np.array(list(feature_activation_times.values())))
	feature_activation_times_histogram = go.Histogram(
		x=log_act_times,
		nbinsx=100,
		hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
		marker_color="#636EFA",
		showlegend=False,
	).to_plotly_json()

	alive_feature_count = client.get_alive_feature_count(dictionary_name, dictionary_series=dictionary_series)
	if alive_feature_count is None:
		return Response(
			content=f"Dictionary {dictionary_name} not found", status_code=404
		)

	return Response(
		content=msgpack.packb(
			make_serializable(
				{
					"dictionary_name": dictionary_name,
					"feature_activation_times_histogram": [feature_activation_times_histogram],
					"alive_feature_count": alive_feature_count,
				}
			)
		),
		media_type="application/x-msgpack",
	)


@app.post("/dictionaries/{dictionary_name}/features/{feature_index}/custom")
def feature_activation_custom_input(
	dictionary_name: str, feature_index: int, input_text: str
):
	try:
		sae = get_sae(dictionary_name)
	except FileNotFoundError:
		return Response(
			content=f"Dictionary {dictionary_name} not found", status_code=404
		)

	if feature_index < 0 or feature_index >= sae.cfg.d_sae:
		return Response(
			content=f"Feature index {feature_index} is out of range", status_code=400
		)

	model = get_model(dictionary_name)
	with torch.no_grad():
		input = model.to_tokens(input_text, prepend_bos=False)
		_, cache = model.run_with_cache_until(input, names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out], until=sae.cfg.hook_point_out)

		feature_acts = sae.encode(cache[sae.cfg.hook_point_in][0], label=cache[sae.cfg.hook_point_out][0])
		sample = {
			"context": [
				bytearray([byte_decoder[c] for c in t])
				for t in model.tokenizer.convert_ids_to_tokens(input[0])
			],
			"feature_acts": feature_acts[:, feature_index].tolist(),
		}

	return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")


@app.post("/dictionaries/{dictionary_name}/custom")
def dictionary_custom_input(dictionary_name: str, input_text: str):
	try:
		sae = get_sae(dictionary_name)
	except FileNotFoundError:
		return Response(
			content=f"Dictionary {dictionary_name} not found", status_code=404
		)

	max_feature_acts = client.get_max_feature_acts(dictionary_name, dictionary_series=dictionary_series)

	model = get_model(dictionary_name)

	with torch.no_grad():
		input = model.to_tokens(input_text, prepend_bos=False)
		_, cache = model.run_with_cache_until(input, names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out], until=sae.cfg.hook_point_out)

		feature_acts = sae.encode(cache[sae.cfg.hook_point_in][0], label=cache[sae.cfg.hook_point_out][0])
		sample = {
			"context": [
				bytearray([byte_decoder[c] for c in t])
				for t in model.tokenizer.convert_ids_to_tokens(input[0])
			],
			"feature_acts_indices": [
				feature_acts[i].nonzero(as_tuple=True)[0].tolist()
				for i in range(feature_acts.shape[0])
			],
			"feature_acts": [
				feature_acts[i][feature_acts[i].nonzero(as_tuple=True)[0]].tolist()
				for i in range(feature_acts.shape[0])
			],
			"max_feature_acts": [
				[max_feature_acts[j] for j in feature_acts[i].nonzero(as_tuple=True)[0].tolist()]
				for i in range(feature_acts.shape[0])
			]
		}

	return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")

class SteeringConfig(BaseModel):
	sae: str
	feature_index: int
	steering_type: Literal["times", "add", "set", "ablate"]
	steering_value: float | None = None

class FeatureNode(BaseModel):
	type: Literal["feature"]
	sae: str
	feature_index: int
	position: int

class LogitsNode(BaseModel):
	type: Literal["logits"]
	position: int
	token_id: int

Node = Annotated[Union[FeatureNode, LogitsNode], Field(discriminator="type")]

class ModelGenerateRequest(BaseModel):
	input_text: str | list[int]
	max_new_tokens: int = 128
	top_k: int = 50
	top_p: float = 0.95
	return_logits_top_k: int = 5
	saes: list[str] = []
	steerings: list[SteeringConfig] = []
	tracings: list[Node] = []
	tracing_threshold: float = 0.1

@app.post("/model/generate")
def model_generate(request: ModelGenerateRequest):
	dictionaries = client.list_dictionaries(dictionary_series=dictionary_series)
	assert len(dictionaries) > 0, "No dictionaries found. Model name cannot be inferred."
	model = get_model(dictionaries[0])
	saes = [(get_sae(name), name) for name in request.saes]
	max_feature_acts = {
		name: client.get_max_feature_acts(name, dictionary_series=dictionary_series)
		for _, name in saes
	}
	assert all(steering.sae in request.saes for steering in request.steerings), "Steering SAE not found"
	assert all(tracing.sae in request.saes for tracing in request.tracings if isinstance(tracing, FeatureNode)), "Tracing SAE not found"
	if len(request.tracings) > 0:
		assert request.max_new_tokens == 0, "Tracing is only supported for max_new_tokens=0"
	
	def generate_steering_hook(steering: SteeringConfig):
		def steering_hook(tensor: torch.Tensor, hook: HookPoint):
			assert len(tensor.shape) == 3
			tensor = tensor.clone()
			if steering.steering_type == "times":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] *= steering.steering_value
			elif steering.steering_type == "ablate":
				tensor[:, :, steering.feature_index] = 0
			elif steering.steering_type == "add":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] += steering.steering_value
			elif steering.steering_type == "set":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] = steering.steering_value
			return tensor
		sae = get_sae(steering.sae)
		return f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", steering_hook
	
	steerings_hooks = [generate_steering_hook(steering) for steering in request.steerings]

	with apply_sae(model, [sae for sae, _ in saes]):
		with model.hooks(steerings_hooks):
			with detach_on(model, [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes]):
				input = model.to_tokens(request.input_text, prepend_bos=False) if isinstance(request.input_text, str) else torch.tensor([request.input_text], device=device)
				if request.max_new_tokens > 0:
					with torch.no_grad():
						output = model.generate(input, max_new_tokens=request.max_new_tokens, top_k=request.top_k, top_p=request.top_p)
						input = output.clone()
				name_filter = [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes] + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre" for sae, _ in saes] + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post" for sae, _ in saes]
				logits, cache = model.run_with_ref_cache(input, names_filter=name_filter)
				logits_topk = [torch.topk(l, request.return_logits_top_k) for l in logits[0]]

				tracing_results = []
				for tracing in request.tracings:
					model.zero_grad()
					if isinstance(tracing, LogitsNode):
						assert tracing.position < logits.shape[1], "Position out of range"
						assert tracing.token_id < logits.shape[2], "Token id out of range"
						logits[:, tracing.position, tracing.token_id].backward(retain_graph=True)
					else:
						sae = get_sae(tracing.sae)
						assert tracing.position < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[1], "Position out of range"
						assert tracing.feature_index < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[2], "Feature index out of range"
						cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0][:, tracing.position, tracing.feature_index].backward(retain_graph=True)
					contributors = []
					for sae, name in saes:
						if cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad is None:
							continue
						attributions = cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad[0] * cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"][0]
						for index in (attributions > request.tracing_threshold).nonzero():
							index = tuple(index.tolist())
							contributors.append(
								{
									"sae": name,
									"position": index[0],
									"feature_index": index[1],
									"attribution": attributions[index].item(),
									"activation": cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"][0][index].item(),
								}
							)
						cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad.zero_()
					tracing_results.append({"node": tracing.model_dump(), "contributors": contributors})

				result = {
					"context": [
						bytearray([byte_decoder[c] for c in t])
						for t in model.tokenizer.convert_ids_to_tokens(input[0])
					],
					"token_ids": input[0].tolist(),
					"logits": {
						"logits": [l.values.tolist() for l in logits_topk],
						"tokens": [
							[
								bytearray([byte_decoder[c] for c in t])
								for t in model.tokenizer.convert_ids_to_tokens(l.indices)
							] for l in logits_topk
						],
						"token_ids": [
							l.indices.tolist()
							for l in logits_topk
						],
					},
					"input_mask": [1 for _ in range(len(input[0]))] + [0 for _ in range(len(input[0]) - len(input[0]))],
					"sae_info": [
						{
							"name": name,
							"feature_acts_indices": [
								cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i].nonzero(as_tuple=True)[0].tolist()
								for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
							],
							"feature_acts": [
								cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i][cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i].nonzero(as_tuple=True)[0]].tolist()
								for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
							],
							"max_feature_acts": [
								[max_feature_acts[name][j] for j in cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i].nonzero(as_tuple=True)[0].tolist()]
								for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
							]
						} for sae, name in saes
					],
					"tracings": tracing_results,
				}
	return Response(content=msgpack.packb(result), media_type="application/x-msgpack")

class ModelTraceRequest(BaseModel):
	input_text: str | list[int]
	saes: list[str] = []
	steerings: list[SteeringConfig] = []
	tracings: list[Node] = []
	tracing_threshold: float = 0.1

@app.post("/model/trace")
def model_trace(request: ModelTraceRequest):
	dictionaries = client.list_dictionaries(dictionary_series=dictionary_series)
	assert len(dictionaries) > 0, "No dictionaries found. Model name cannot be inferred."
	model = get_model(dictionaries[0])
	saes = [(get_sae(name), name) for name in request.saes]
	assert all(steering.sae in request.saes for steering in request.steerings), "Steering SAE not found"
	assert all(tracing.sae in request.saes for tracing in request.tracings if isinstance(tracing, FeatureNode)), "Tracing SAE not found"
	
	def generate_steering_hook(steering: SteeringConfig):
		def steering_hook(tensor: torch.Tensor, hook: HookPoint):
			assert len(tensor.shape) == 3
			tensor = tensor.clone()
			if steering.steering_type == "times":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] *= steering.steering_value
			elif steering.steering_type == "ablate":
				tensor[:, :, steering.feature_index] = 0
			elif steering.steering_type == "add":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] += steering.steering_value
			elif steering.steering_type == "set":
				assert steering.steering_value is not None
				tensor[:, :, steering.feature_index] = steering.steering_value
			return tensor
		sae = get_sae(steering.sae)
		return f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", steering_hook
	
	steerings_hooks = [generate_steering_hook(steering) for steering in request.steerings]

	with apply_sae(model, [sae for sae, _ in saes]):
		with model.hooks(steerings_hooks):
			with detach_on(model, [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes]):
				input = model.to_tokens(request.input_text, prepend_bos=False) if isinstance(request.input_text, str) else torch.tensor([request.input_text], device=device)
				name_filter = [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes] + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre" for sae, _ in saes] + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post" for sae, _ in saes]
				logits, cache = model.run_with_ref_cache(input, names_filter=name_filter)
				tracing_results = []
				for tracing in request.tracings:
					model.zero_grad()
					if isinstance(tracing, LogitsNode):
						assert tracing.position < logits.shape[1], "Position out of range"
						assert tracing.token_id < logits.shape[2], "Token id out of range"
						logits[:, tracing.position, tracing.token_id].backward(retain_graph=True)
					else:
						sae = get_sae(tracing.sae)
						assert tracing.position < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[1], "Position out of range"
						assert tracing.feature_index < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[2], "Feature index out of range"
						cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0][:, tracing.position, tracing.feature_index].backward(retain_graph=True)
					contributors = []
					for sae, name in saes:
						if cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad is None:
							continue
						attributions = cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad[0] * cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"][0]
						for index in (attributions > request.tracing_threshold).nonzero():
							index = tuple(index.tolist())
							contributors.append(
								{
									"sae": name,
									"position": index[0],
									"feature_index": index[1],
									"attribution": attributions[index].item(),
									"activation": cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"][0][index].item(),
								}
							)
						cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"].grad.zero_()
					tracing_results.append({"node": tracing.model_dump(), "contributors": contributors})

				result = {
					"context": [
						bytearray([byte_decoder[c] for c in t])
						for t in model.tokenizer.convert_ids_to_tokens(input[0])
					],
					"token_ids": input[0].tolist(),
					"tracings": tracing_results,
				}
	return Response(content=msgpack.packb(result), media_type="application/x-msgpack")


@app.post("/dictionaries/{dictionary_name}/features/{feature_index}/interpret")
def feature_interpretation(
	dictionary_name: str,
	feature_index: int,
	type: str,
	custom_interpretation: str | None = None,
):
	model = get_model(dictionary_name)
	path = client.get_dictionary(dictionary_name, dictionary_series=dictionary_series)['path'] or f"{result_dir}/{dictionary_name}"
	if type == "custom":
		interpretation = {
			"text": custom_interpretation,
			"validation": [
				{
					"method": "manual",
					"passed": True,
				}
			],
		}
	elif type == "auto":
		cfg = AutoInterpConfig(
			**{
				"sae": SAEConfig.from_pretrained(path).to_dict(),
				"lm": LanguageModelConfig.from_pretrained_sae(path).to_dict(),
				"openai_api_key": os.environ.get("OPENAI_API_KEY"),
				"openai_base_url": os.environ.get("OPENAI_BASE_URL"),
			}
		)
		feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
		result = generate_description(model, feature["analysis"][0], cfg)
		interpretation = {
			"text": result["response"],
			"validation": [],
			"detail": result,
		}
	elif type == "validate":
		cfg = AutoInterpConfig(
			**{
				"sae": SAEConfig.from_pretrained(path).to_dict(),
				"lm": LanguageModelConfig.from_pretrained_sae(path).to_dict(),
				"openai_api_key": os.environ.get("OPENAI_API_KEY"),
				"openai_base_url": os.environ.get("OPENAI_BASE_URL"),
			}
		)
		feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
		interpretation = feature["interpretation"] if "interpretation" in feature else None
		if interpretation is None:
			return Response(content="Feature interpretation not found", status_code=404)
		validation = cast(Any, interpretation["validation"])
		if not any(v["method"] == "activation" for v in validation):
			validation_result = check_description(
				model,
				cfg,
				feature_index,
				cast(str, interpretation["text"]),
				False,
				feature_activation=feature["analysis"][0],
			)
			validation.append(
				{
					"method": "activation",
					"passed": validation_result["passed"],
					"detail": validation_result,
				}
			)
		if not any(v["method"] == "generative" for v in validation):
			validation_result = check_description(
				model,
				cfg,
				feature_index,
				cast(str, interpretation["text"]),
				True,
				sae=get_sae(dictionary_name),
			)
			validation.append(
				{
					"method": "generative",
					"passed": validation_result["passed"],
					"detail": validation_result,
				}
			)

	try:
		client.update_feature(dictionary_name, feature_index,
							  {"interpretation": interpretation}, dictionary_series=dictionary_series)
	except ValueError as e:
		return Response(content=str(e), status_code=400)
	return interpretation


app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
