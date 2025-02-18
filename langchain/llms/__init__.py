"""Wrappers on top of large language models APIs."""
from typing import Dict, Type

from langchain.llms.ai21 import AI21
from langchain.llms.base import BaseLLM
from langchain.llms.cohere import Cohere
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.nlpcloud import NLPCloud
from langchain.llms.openai import AzureOpenAI, OpenAI, OpenAIPooled

__all__ = [
    "Cohere",
    "NLPCloud",
    "OpenAI",
    "OpenAIPooled",
    "HuggingFaceHub",
    "HuggingFacePipeline",
    "AI21",
    "AzureOpenAI",
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "ai21": AI21,
    "cohere": Cohere,
    "huggingface_hub": HuggingFaceHub,
    "nlpcloud": NLPCloud,
    "openai": OpenAI,
    "pooled_openai": OpenAIPooled,
    "huggingface_pipeline": HuggingFacePipeline,
    "azure": AzureOpenAI,
}
