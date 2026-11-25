"""Pluggable LLM inference backends for the self-rewarding factory.

Backend specs (used by --backend):
    groq:openai/gpt-oss-120b                 -> Groq Cloud (matches the repo's
                                                text_generation_models.py usage;
                                                needs GROQ_CLOUD_API_KEY or
                                                GROQ_API_KEY in the environment)
    openai-compat:<model>@<base_url>         -> any OpenAI-compatible server,
                                                e.g. vLLM on HiPerGator:
                                                openai-compat:openai/gpt-oss-120b@http://127.0.0.1:8000/v1
    hf:<hf-model-id>                         -> local transformers model
    mock                                     -> deterministic fake, for testing
                                                the pipeline end-to-end
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, *,
                 temperature: float = 0.7, max_tokens: int = 4096,
                 seed: int | None = None) -> str:
        """Return the raw text completion."""


class GroqBackend(LLMBackend):
    def __init__(self, model_name: str):
        from groq import Groq  # noqa: import deferred until used
        api_key = os.getenv("GROQ_CLOUD_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GroqBackend needs GROQ_CLOUD_API_KEY (or GROQ_API_KEY) in the "
                "environment.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.name = f"groq:{model_name}"

    def generate(self, system_prompt, user_prompt, *, temperature=0.7,
                 max_tokens=4096, seed=None):
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            # Groq supports a seed parameter on chat completions.
            kwargs["seed"] = seed
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content


class OpenAICompatibleBackend(LLMBackend):
    """Works with vLLM `serve`, TGI, LM Studio, etc."""

    def __init__(self, model_name: str, base_url: str):
        import openai  # noqa: deferred
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY"),
        )
        self.model_name = model_name
        self.name = f"openai-compat:{model_name}@{base_url}"

    def generate(self, system_prompt, user_prompt, *, temperature=0.7,
                 max_tokens=4096, seed=None):
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content


class TransformersBackend(LLMBackend):
    """Local Hugging Face model (for machines with the weights cached)."""

    def __init__(self, model_id: str):
        import torch  # noqa: deferred
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        self.model_id = model_id
        self.name = f"hf:{model_id}"

    def generate(self, system_prompt, user_prompt, *, temperature=0.7,
                 max_tokens=4096, seed=None):
        import torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=temperature or None)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class MockBackend(LLMBackend):
    """Deterministic fake backend for end-to-end pipeline testing.

    Writer behaviour: emits `per_call` candidate JSON objects drawn from
    varied sentence templates; every 4th candidate is intentionally broken
    (a non-verbatim span) so rejection paths are exercised.
    Judge behaviour: scores 5 when the sentence contains "will", else 2.
    """

    _SUBJECTS = ["widget sales", "solar panel shipments", "cocoa futures",
                 "apartment rents", "chip exports", "airline bookings",
                 "wheat harvests", "cloud spending"]
    _REGIONS = ["the Nordic region", "Southeast Asia", "the Midwest",
                "coastal provinces", "the eurozone", "Latin America"]
    _VERBS = ["will rise by", "will fall by", "are expected to reach",
              "will climb past", "are projected to double to"]

    def __init__(self, per_call: int = 5):
        self.name = "mock"
        self.per_call = per_call
        self._counter = 0

    def generate(self, system_prompt, user_prompt, *, temperature=0.7,
                 max_tokens=4096, seed=None):
        import json
        if "Score:" in user_prompt or "additive 5-point" in user_prompt:
            # Judge call: derive a deterministic score from the sentence.
            good = " will " in f" {user_prompt} "
            score = 5 if good else 2
            return (f"1. Sentence quality: ok.\n2. Target: ok.\n"
                    f"3. Outcome: ok.\n4. Source & Date: ok.\n"
                    f"5. Schema: ok.\nScore: {score}")
        # Writer call.
        items = []
        for _ in range(self.per_call):
            self._counter += 1
            n = self._counter
            if n % 4 == 0:
                # Broken candidate: span not verbatim in the sentence.
                items.append({
                    "sentence": f"Analysts expect mock index {n} to climb "
                                f"next year.",
                    "spans": {"1": ["Analysts"], "2": ["MOCK INDEX"],
                              "3": ["next year"], "4": ["to climb"]},
                    "is_tolsa": True,
                })
            else:
                subj = self._SUBJECTS[n % len(self._SUBJECTS)]
                region = self._REGIONS[n % len(self._REGIONS)]
                verb = self._VERBS[n % len(self._VERBS)]
                value = 3 + (n * 7) % 40
                sentence = (f"Mock Research Group {n} expects {subj} in "
                            f"{region} {verb} {value}% next year.")
                items.append({
                    "sentence": sentence,
                    "spans": {"1": [f"Mock Research Group {n}"],
                              "2": [f"{subj} in {region}"],
                              "3": ["next year"],
                              "4": [f"{verb} {value}%"]},
                    "is_tolsa": True,
                })
        return json.dumps(items)


def make_backend(spec: str, **kwargs) -> LLMBackend:
    if spec == "mock":
        return MockBackend(**kwargs)
    kind, _, rest = spec.partition(":")
    if kind == "groq":
        return GroqBackend(rest)
    if kind == "openai-compat":
        model, _, base_url = rest.partition("@")
        if not base_url:
            raise ValueError(
                "openai-compat spec must be '<model>@<base_url>', e.g. "
                "openai-compat:openai/gpt-oss-120b@http://127.0.0.1:8000/v1")
        return OpenAICompatibleBackend(model, base_url)
    if kind == "hf":
        return TransformersBackend(rest)
    raise ValueError(f"Unknown backend spec: {spec!r}")
