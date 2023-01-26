from pydantic import BaseModel
from langchain.prompts import BasePromptTemplate, FewShotPromptTemplate2
from langchain.prompts.base import BaseOutputParser, DEFAULT_FORMATTER_MAPPING
# from tabmwp.utilities import extract_prediction, normalize_answer
# from tabmwp.base_prompt import get_table_text, get_question_text, create_one_example

# prefix_template ="""
# Answer the Question based on the Table.
# """.strip()

class LastLetterConcat(BasePromptTemplate):
    prompt_format: str = ''

    input_variables: list[str] = ['question', 'answer']
    """A list of the names of the variables the prompt template expects."""

    def format(self, prompt_format=None, test=False, append_output=True,  **kwargs):
        question = kwargs.pop('question')
        answer = kwargs.pop('answer')
        prompt_format = prompt_format or self.prompt_format
        # solution = kwargs.pop('answer') 
        if test:
            example = f'Q: {question}\nA:'
        else:
            example = f'Q: {question}\nA: {answer}'
        return example

class LastLetterConcatCoT(BasePromptTemplate): #todo ask Navid about cleaning this up #rename this to LetterConcat
    prompt_format: str = ''

    input_variables: list[str] = ['question', 'explanation']
    """A list of the names of the variables the prompt template expects."""

    def format(self, prompt_format=None, test=False, append_output=True,  **kwargs):
        question = kwargs.pop('question')
        explaination = kwargs.pop('explanation')
        prompt_format = prompt_format or self.prompt_format
        # solution = kwargs.pop('answer') 
        if test:
            example = f'Q: {question}\nA:'
        else:
            example = f'Q: {question}\nA: {explaination}'
        return example


class LastLetterOutputParserCoT(BaseOutputParser, BaseModel):
    #see if this is too hacky
    def parse(self, text, **kwargs):
        predictions = text.split('\n')
        if len(predictions) > 1:
            if "The answer is" in predictions[0]:
                prediction = predictions[0].split("The answer is")[-1][:-1].strip().lower()
                return prediction    
        return ""

class LastLetterOutputParser(BaseOutputParser, BaseModel):
    def parse(self, text, **kwargs):
        predictions = text.split('\n')
        if len(predictions) > 1:
            return predictions[0].strip()
        else:
            return ""
        