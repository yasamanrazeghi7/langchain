"""Prompt template classes."""
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotPromptTemplate2
from langchain.prompts.last_letter_concat_prompt import LastLetterConcat, LastLetterOutputParser, LastLetterConcatCoT, LastLetterOutputParserCoT, NoPrefix, NoPrefixOutputParser
from langchain.prompts.loading import load_from_hub, load_prompt
from langchain.prompts.prompt import Prompt, PromptTemplate
#TODO: ask about this

__all__ = [
    "BasePromptTemplate",
    "load_prompt",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptTemplate2",
    "Prompt",
    "load_from_hub",
    "LastLetterConcat",
    "LastLetterOutputParser",
    "LastLetterConcatCoT",
    "LastLetterOutputParserCoT",
    "NoPrefix",
    "NoPrefixOutputParser",
]
