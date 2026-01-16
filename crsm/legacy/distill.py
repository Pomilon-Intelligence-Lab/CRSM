"""
Distillation dataset generation and teacher interface.
Supports multiple providers (gemini, openai, local transformers) via a common interface.

This module provides:
- a Provider abstraction
- a GeminiProvider implementation (uses google.generativeai if installed)
- an OpenAIProvider implementation (uses openai package if installed)
- a LocalProvider implementation (uses transformers pipeline if installed)
- generate_traces() to produce JSONL traces using the selected teacher

Note: network calls are not executed in tests here. The code is written to be
safe when the provider libraries are not installed; in that case the provider
will raise a helpful error at runtime.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Optional


class Provider:
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError()


class GeminiProvider(Provider):
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-2.5-flash'):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.model = model
        try:
            import google.generativeai as genai
            self.genai = genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
        except Exception as e:
            raise ImportError('google.generativeai package required for GeminiProvider') from e

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.genai.generate(model=self.model, prompt=prompt, max_output_tokens=max_tokens)
        # The response structure depends on the package version; try common fields
        if hasattr(resp, 'text'):
            return resp.text
        if isinstance(resp, dict) and 'candidates' in resp:
            return resp['candidates'][0]['content'][0]['text']
        # fallback
        return str(resp)


class OpenAIProvider(Provider):
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o-mini'):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        try:
            import openai
            self.openai = openai
            if self.api_key:
                openai.api_key = self.api_key
        except Exception as e:
            raise ImportError('openai package required for OpenAIProvider') from e

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        res = self.openai.Completion.create(model=self.model, prompt=prompt, max_tokens=max_tokens)
        if 'choices' in res and len(res['choices']) > 0:
            return res['choices'][0]['text']
        return str(res)


class LocalProvider(Provider):
    def __init__(self, model_name: str = 'gpt2'):
        try:
            from transformers import pipeline
            self.pipe = pipeline('text-generation', model=model_name, device_map='auto')
        except Exception as e:
            raise ImportError('transformers pipeline required for LocalProvider') from e

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        out = self.pipe(prompt, max_length=max_tokens, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get('generated_text', '')
        return str(out)


def get_provider(kind: str = 'gemini', **kwargs) -> Provider:
    kind = kind.lower()
    if kind == 'gemini':
        return GeminiProvider(**kwargs)
    if kind == 'openai':
        return OpenAIProvider(**kwargs)
    if kind == 'local':
        return LocalProvider(**kwargs)
    raise ValueError(f'Unknown provider: {kind}')


def generate_traces(prompts: List[str], out_path: str, provider: Provider, prompt_erasure: bool = False):
    """Generate reasoning traces from teacher provider and save as JSONL.

    Each line contains {"prompt":..., "trace":..., "erased": boolean}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            trace = provider.generate(p)
            if prompt_erasure:
                # remove explicit prompt text from the trace (best effort)
                erased_trace = trace.replace(p, '')
                entry = {'prompt': p, 'trace': erased_trace, 'erased': True}
            else:
                entry = {'prompt': p, 'trace': trace, 'erased': False}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # simple local demo runner
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', type=str, default='gemini')
    parser.add_argument('--out', type=str, default='data/distill_traces.jsonl')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--prompt-erasure', action='store_true')
    args = parser.parse_args()

    prov = get_provider(args.provider)
    generate_traces([args.prompt], args.out, prov, prompt_erasure=args.prompt_erasure)
