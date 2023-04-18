import json
import torch
import random
import numpy as np
import xmltodict

from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.prompts import BasePromptTemplate, FewShotPromptTemplate2, PromptTemplate, LastLetterConcat, LastLetterOutputParser, LastLetterConcatCoT, LastLetterOutputParserCoT, NoPrefix, NoPrefixOutputParser


def make_batch(question_list, batch_size):
    for i in range(0, len(question_list), batch_size):
        yield question_list[i:i+batch_size]


def read_jsonl(filename):
    with open(filename, 'r') as filename:
        all_lines = filename.readlines()
    all_data = [json.loads(x) for x in all_lines] 
    return all_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reform_gsm8k_list(data_list):
    reform_list = []
    for data in data_list:
        new_data = {}
        new_data['question'] = data['question']
        new_data['answer'] = data['answer'].split('####')[1].strip()
        new_data['explanation'] = f"{data['answer'].split('####')[0].strip()} The answer is {new_data['answer']}."
        new_data['explanation'] = new_data['explanation'].replace('\n', ' ')
        reform_list.append(new_data)
    return reform_list

def set_up_letter_concat_data_set(data_set_name: str, shots: int, learning_mode: str) -> FewShotPromptTemplate2:
    # reads the data from and returns a FewShotPromptTemplate2 it should have exmpla_templat and ouput_parser
    train_data = read_jsonl(f"./datasets/{data_set_name}_1000.0_2_train.jsonl")
    test_data = read_jsonl(f"./datasets/{data_set_name}_1000.0_2_test.jsonl")
    train_shots = random.sample(train_data, k=shots)
    if learning_mode == 'icl':
        example_template = LastLetterConcat()
        input_variables=['question', 'answer']
        output_parser = LastLetterOutputParser()
    else:  
        example_template = LastLetterConcatCoT()
        input_variables=['question', 'explanation']
        output_parser = LastLetterOutputParserCoT()
    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)

def set_up_arithmetics(data_set_name: str, shots: int, learning_mode: str) -> FewShotPromptTemplate2:
    # reads the data from and returns a FewShotPromptTemplate2 it should have exmpla_templat and ouput_parser
    train_data = read_jsonl(f"./datasets/simple_arithmetics/arithmetic-train-{data_set_name}.jsonl")
    test_data = read_jsonl(f"./datasets/simple_arithmetics/arithmetic-dev-{data_set_name}.jsonl")
    train_shots = random.sample(train_data, k=shots)
    if learning_mode == 'icl':
        example_template = LastLetterConcat()
        input_variables=['question', 'answer']
        output_parser = LastLetterOutputParser()
    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)




def set_up_gsm8k(data_set_name: str, shots: int, learning_mode: str):
    test_data = read_jsonl("./datasets/gsm8k/GSM8k-dev-seperated.jsonl")
    if learning_mode == 'icl':
        train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        example_template = LastLetterConcat()
        input_variables=['question', 'answer']
        output_parser = LastLetterOutputParser()
    else:
        if learning_mode == 'standard_cot':
            train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        else:
            train_shots = read_jsonl(f"./datasets/gsm8k/GSM8k-dev-seperated.jsonl")
            train_shots = random.sample(train_shots, k=shots)
        example_template = LastLetterConcatCoT()
        input_variables=['question', 'explanation']
        output_parser = LastLetterOutputParserCoT()
    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)

def set_up_dm_math(data_set_name: str, shots: int, learning_mode: str):
    train_shots = read_jsonl(f"./datasets/dm_math/{data_set_name}_train.jsonl")
    train_shots = random.sample(train_shots, k=shots)
    test_data = read_jsonl(f"./datasets/dm_math/{data_set_name}_dev.jsonl")
    example_template = NoPrefix()
    input_variables=['question', 'answer']
    output_parser = NoPrefixOutputParser()
    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)


def set_up_svamp(data_set_name: str, shots: int, learning_mode: str):
    with open('./datasets/svamp.json', 'r') as f:
        test_data = json.loads(f.read())
            
    for i, question in enumerate(test_data):
        golden_answer = question['Answer']
        delimiter = ' ' if question['Body'][-1]=='.' else ', '
        question['question'] = question['Body'] + delimiter + question['Question'] 
        question['answer'] = golden_answer
        question['explanation'] = None
    
    if learning_mode == 'icl':
        # raise ValueError('SVAMP does not have a train set for standard learning as of now')
        train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        train_shots = random.sample(train_shots, k=shots)
        example_template = LastLetterConcat()
        input_variables=['question', 'answer']
        output_parser = LastLetterOutputParser()
    else:
        train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        train_shots = random.sample(train_shots, k=shots)
        example_template = LastLetterConcatCoT()
        input_variables=['question', 'explanation']
        output_parser = LastLetterOutputParserCoT()

    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)

def set_up_asdiv(data_set_name: str, shots: int, learning_mode: str):
    with open('./datasets/asdiv.xml', 'r') as f:
        dict = xmltodict.parse(f.read())
    test_data = dict['Machine-Reading-Corpus-File']['ProblemSet']['Problem']
    
    for i, question in enumerate(test_data):
        golden_answer = question['Answer'].split(" ")[0]
        question['question'] = question['Body'] + ' ' + question['Question'] 
        question['answer'] = golden_answer.strip()
        question['explanation'] = None
    
    if learning_mode == 'icl':
        # raise ValueError('asdiv does not have a train set for standard learning as of now')
        train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        train_shots = random.sample(train_shots, k=shots)
        example_template = NoPrefix()
        input_variables=['question', 'answer']
        output_parser = NoPrefixOutputParser()
    else:
        train_shots = read_jsonl(f"./datasets/cot_prompt_math_8shot.jsonl")
        train_shots = random.sample(train_shots, k=shots)
        example_template = LastLetterConcatCoT()
        input_variables=['question', 'explanation']
        output_parser = LastLetterOutputParserCoT()
    return test_data, FewShotPromptTemplate2(examples = train_shots, example_template=example_template, input_variables=input_variables, output_parser=output_parser)




def set_up_data_set(data_set_name: str, shots: int, learning_mode: str) -> FewShotPromptTemplate2:
    # this is where you can add more datasets
    if data_set_name == 'last_letter' or data_set_name == 'first_letter':
        return set_up_letter_concat_data_set(data_set_name, shots, learning_mode)
    elif data_set_name == 'gsm8k':
        if 'cot' in learning_mode and shots > 8:
            raise ValueError("GSM8K dataset only supports less than 8 shots in CoT learning mode because we are following the original paper wei 2022")
        return set_up_gsm8k(data_set_name, shots, learning_mode)
    elif data_set_name == 'svamp':
        if 'cot' in learning_mode and shots != 8:
            raise ValueError("svamp dataset only supports 8 shots in CoT learning mode because we are following the original paper wei 2022")
        return set_up_svamp(data_set_name, shots, learning_mode)
    elif data_set_name == 'asdiv':
        if 'cot' in learning_mode and shots != 8:
            raise ValueError("asdiv dataset only supports 8 shots in CoT learning mode because we are following the original paper wei 2022")
        return set_up_asdiv(data_set_name, shots, learning_mode)
    elif "dmmath" in data_set_name:
        if 'cot' in learning_mode:
            raise ValueError("dm_math do not have explanations")
        return set_up_dm_math(data_set_name, shots, learning_mode)
    elif "digits" in data_set_name:
        if 'cot' in learning_mode:
            raise ValueError("arithmetics do not have explanations")
        return set_up_arithmetics(data_set_name, shots, learning_mode)
    else:
        raise ValueError("Dataset not supported")

    

    