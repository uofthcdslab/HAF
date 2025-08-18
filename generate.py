import argparse
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from data_loader import DataLoader
import torch
import numpy as np
import random
import pickle
from pathlib import Path
from tqdm import tqdm
import json
from utils.logger_setup import setup_logger
from utils.data_path_prefixes import GEN_OUTPUT_PATH

class Generator:
    """A class to generate LLM responses"""
    
    def __init__(self, args, logger):
        self.logger = logger
        self.logger.info(f"Initializing Generator with model: {args.model_name}")        
        self.initiate_class_variables(args)
        self.set_required_seeds()
        
        # initialize tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token_id is None:
            # tokenizer.pad_token = tokenizer.eos_token  # use EOS token as PAD token
            self.logger.info("Adding pad token to tokenizer")
            self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.padding_side = "left"  # for decoder-type mdoels
        
        # Modify chat template to support disable_system_prompt
        self.logger.info("Modifying chat template to support custom system prompts")
        self.tokenizer.chat_template = "{% if not disable_system_prompt %}{{'<|im_start|>system<|im_sep|>You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:<|im_end|>'}}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}{% generation %}{{message['content'] + '<|im_end|>'}}{% endgeneration %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
        
        self.logger.info("Tokenizer initialized successfully")

    def initiate_class_variables(self, args):
        self.logger.debug("Setting class variables from arguments")
        self.data_name = args.data_name
        self.data_size = args.data_size
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.do_sample = True if args.do_sample == 'True' else False
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.write_frequency = args.write_frequency
        self.seed_value = args.seed_value
        self.cache_dir = args.cache_dir if args.cache_dir != '' else None
        self.generation_stage = args.generation_stage
        self.select_new_dataset_samples = True if args.select_new_dataset_samples == 'True' else False
        self.explicit_prompting = '_explicit' if args.explicit_prompting == 'True' and self.generation_stage != 'individual' else ''
        
    def set_required_seeds(self):
        self.logger.info(f"Setting random seeds to {self.seed_value} for reproducibility")
        # Set the seeds for reproducibility
        os.environ["PYTHONHASHSEED"] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        torch.cuda.manual_seed_all(self.seed_value)
        # the below may affect performance
        # torch.backends.cudnn.deterministic = True  # Ensures deterministic algorithms
        # torch.backends.cudnn.benchmark = False    # Ensures determinism

    def format_inputs_as_chat(self, input_text, reasons, second_text=''):
        first_text = '\nTEXT: \n' + input_text.lstrip()
        second_text = '\n\nREASON(S): \n' if second_text == '' else second_text
        for ix, reason in enumerate(reasons):
            second_text += str(ix+1)+'. ' + reason + '\n'
        return first_text + second_text

    def create_input_list(self, **data_args):
        self.logger.info(f"Creating input list for {self.generation_stage} generation stage")
        # load instructions
        with open("utils/prompt_instructions.json", "r") as file:
            instructions = json.load(file)
            self.logger.debug("Loaded prompt instructions")

        data_loader = DataLoader(self.data_name, total_samples=self.data_size, batch_size=self.batch_size, random_state=self.seed_value, select_new_dataset_samples=self.select_new_dataset_samples)
        if self.generation_stage == 'initial': # initial generation
            self.logger.info("Loading data for initial generation")
            raw_data = data_loader.load_for_initial_generation(**data_args)
            self.logger.info(f"DATASET SIZE: {len(raw_data)}")
            input_data = raw_data.groupby('batch')['input_text'].apply(list).tolist()
            input_data = [[self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": instructions["for_"+self.generation_stage+"_generation"+self.explicit_prompting]},
                        {"role": "user", "content": "\nTEXT: \n" + text.lstrip()}],
                        tokenize=False,
                        add_generation_prompt=True,
                        disable_system_prompt=True
                    ) for text in sublist] 
                for sublist in input_data]
        
        elif self.generation_stage == 'individual':
            self.logger.info(f"Loading data for {self.generation_stage} reliance generation")   
            self.logger.info(f"Loading reasons generated from explicit prompting")   
            extracted_info = data_loader.load_for_conditional_generation(model_name=self.model_name, explcit_prompting='_explicit', **data_args)
            input_data = []
            for ix in range(len(extracted_info['input_texts'])):
                one_sample_batch = []
                system_prompt = instructions["for_individual_reliance"]
                if extracted_info['decisions'][ix] == 'non-toxic':
                    for rix in range(len(extracted_info['reasons'][ix])):
                        system_prompt = system_prompt.replace("Given below is a toxic TEXT and a REASON explaining why it is toxic. Is any additional reason required to justify why the TEXT is toxic?",
                                                            "Given below is a non-toxic TEXT and a list of REASON(S) explaining why it is not toxic. Is any additional reason required to justify why the TEXT is not toxic?")            
                        modified_reasons = extracted_info['reasons'][ix][:rix] + extracted_info['reasons'][ix][rix+1:]
                        one_sample_batch.append(self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": self.format_inputs_as_chat(extracted_info['input_texts'][ix], modified_reasons)}],
                            tokenize=False,
                            add_generation_prompt=True,
                            disable_system_prompt=True
                        ))
                else:
                    for rix in range(len(extracted_info['reasons'][ix])):
                        one_sample_batch.append(self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": self.format_inputs_as_chat(extracted_info['input_texts'][ix], [extracted_info['reasons'][ix][rix]], second_text='\n\nREASON: \n')}],
                            tokenize=False,
                            add_generation_prompt=True,
                            disable_system_prompt=True
                        ))    
                input_data.append(one_sample_batch)
         
        else: # conditional generation - for internal/external reliances
            self.logger.info(f"Loading data for {self.generation_stage} reliance generation")
            extracted_info = data_loader.load_for_conditional_generation(model_name=self.model_name, explcit_prompting=self.explicit_prompting, **data_args)
            input_data = []
            for ix in range(len(extracted_info['input_texts'])):
                if extracted_info['decisions'][ix] == 'non-toxic':
                    system_prompt = instructions["for_"+self.generation_stage+"_reliance"+self.explicit_prompting].replace("Given below is a toxic TEXT and a list of REASON(S) explaining why it is toxic",
                                                                                   "Given below is a non-toxic TEXT and a list of REASON(S) explaining why it is not toxic")
                    if self.explicit_prompting:
                        system_prompt = system_prompt.replace("required to justify why the TEXT is toxic", "required to justify why the TEXT is not toxic")
                else:
                    system_prompt = instructions["for_"+self.generation_stage+"_reliance"+self.explicit_prompting]
                
                input_data.append(self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.format_inputs_as_chat(extracted_info['input_texts'][ix], extracted_info['reasons'][ix])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    disable_system_prompt=True
                ))
            input_data = [input_data[i:i + self.batch_size] for i in range(0, len(input_data), self.batch_size)]    
    
        self.logger.info(f"Created {len(input_data)} batches of input data")
        return input_data

    def run_model(self, input_data):
        # Setup
        self.logger.info("Starting model inference")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {device} device")
        with open("utils/model_size_map.json", "r") as file:
            model_size = json.load(file)

        # Load the model
        self.logger.info(f"Loading model: {self.model_name}")
        if model_size[self.model_name] >= 13:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, 
                                                        cache_dir=self.cache_dir, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, 
                                                        cache_dir=self.cache_dir).cuda()
            
        # Set the model to eval mode
        self.model.eval()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.logger.info("Model loaded and prepared for inference")
        # model.generation_config.cache_implementation = "static"
        
        # create directory for results
        directory_path = Path(GEN_OUTPUT_PATH+"/"+self.model_name.split('/')[1]+'/'+self.data_name+'/'+self.generation_stage+self.explicit_prompting)
        directory_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Results will be saved to {directory_path}")
        
        # Generations
        input_tokens = []
        output_tokens = []
        logits_entropies = []
        scores_entropies = []
        generated_text = [] 
        start_ix = 0
        end_ix = 0            
        self.logger.info('Generating LLM responses...')
        with torch.no_grad():
            for batch_idx, batch_input in enumerate(tqdm(input_data)):
                self.logger.debug(f"Processing batch {batch_idx+1}/{len(input_data)}")
                if len(batch_input) == 0: # happens in individual runs
                    self.logger.debug("Empty batch encountered, saving empty lists")
                    file_path = directory_path / ("samples_"+str(batch_idx+1)+"-"+str(len(batch_input))+".pkl")
                    self.save_results(file_path, input_tokens, output_tokens, logits_entropies, scores_entropies, generated_text)
                    continue
                inputs = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(device)
                try:
                    self.logger.debug(f"Generating responses for batch of size {len(batch_input)}")
                    generations = self.model.generate(
                        **inputs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=self.do_sample,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                    )
                    input_tokens.append(inputs["input_ids"].to('cpu')) # batch x len_seq
                    output_tokens.append(generations["sequences"].to('cpu'))
                    self.logger.debug("Computing entropies")
                    processed_logits, processed_scores = self.get_entropies(inputs["input_ids"].to('cpu'), generations["sequences"].to('cpu'),
                                                                            torch.stack(generations["logits"], dim=1).to('cpu'),
                                                                            torch.stack(generations["scores"], dim=1).to('cpu'))
                    logits_entropies.append(processed_logits)
                    scores_entropies.append(processed_scores)
                    generated_text.append(self.tokenizer.batch_decode(generations.sequences.to("cpu"), skip_special_tokens=True))
                    end_ix += len(batch_input)
                        
                    if (self.generation_stage == 'individual') or ((self.generation_stage != 'individual') and (end_ix - start_ix) >= self.write_frequency):
                        self.logger.info(f"Writing results for samples {start_ix+1}-{end_ix}")
                        directory_path = Path(GEN_OUTPUT_PATH+"/"+self.model_name.split('/')[1]+'/'+self.data_name+'/'+self.generation_stage+self.explicit_prompting)
                        directory_path.mkdir(parents=True, exist_ok=True)
                        if self.generation_stage == 'individual':
                            file_path = directory_path / ("samples_"+str(batch_idx+1)+"-"+str(len(batch_input))+".pkl")
                        else:
                            file_path = directory_path / ("samples_"+str(start_ix+1)+"-"+str(end_ix)+".pkl")
                        self.save_results(file_path, input_tokens, output_tokens, logits_entropies, scores_entropies, generated_text)
                        input_tokens = []
                        output_tokens = []
                        logits_entropies = []
                        scores_entropies = []
                        generated_text = [] 
                        start_ix = end_ix
                
                except Exception as e:
                    self.logger.error(f"Error during generation: {str(e)}", exc_info=True)
                    self.logger.info(f"Saving partial results for samples {start_ix+1}-{end_ix}")
                    directory_path = Path(GEN_OUTPUT_PATH+"/"+self.model_name.split('/')[1]+'/'+self.data_name+'/'+self.generation_stage+self.explicit_prompting)
                    directory_path.mkdir(parents=True, exist_ok=True)
                    if self.generation_stage == 'individual':
                        file_path = directory_path / ("samples_"+str(batch_idx+1)+"-"+str(len(batch_input))+".pkl")
                    else:
                        file_path = directory_path / ("samples_"+str(start_ix+1)+"-"+str(end_ix)+".pkl")
                    self.save_results(file_path, input_tokens, output_tokens, logits_entropies, scores_entropies, generated_text)
                    break
            
            if len(input_tokens) > 0: # store remaining data
                self.logger.info(f"Saving final results for samples {start_ix+1}-{end_ix}")
                directory_path = Path(GEN_OUTPUT_PATH+"/"+self.model_name.split('/')[1]+'/'+self.data_name+'/'+self.generation_stage+self.explicit_prompting)
                directory_path.mkdir(parents=True, exist_ok=True)
                if self.generation_stage == 'individual':
                    file_path = directory_path / ("samples_"+str(batch_idx+1)+"-"+str(len(batch_input))+".pkl")
                else:
                    file_path = directory_path / ("samples_"+str(start_ix+1)+"-"+str(end_ix)+".pkl")
                self.save_results(file_path, input_tokens, output_tokens, logits_entropies, scores_entropies, generated_text)

    def get_entropies(self, input_tokens, output_tokens, logits, scores):
        # token-wise predictive entropies
        processed_logits = []
        processed_scores = []
        for sample_ix in range(len(input_tokens)):
            this_sample_input_len = len(input_tokens[sample_ix])
            target_ids = output_tokens[sample_ix].clone()[this_sample_input_len:]
            token_wise_entropy_logits = torch.nn.CrossEntropyLoss(reduction='none')(logits[sample_ix], target_ids)
            token_wise_entropy_scores = torch.nn.CrossEntropyLoss(reduction='none')(scores[sample_ix], target_ids)
            processed_logits.append(token_wise_entropy_logits)
            processed_scores.append(token_wise_entropy_scores)
        return processed_logits, processed_scores
        
    def save_results(self, file_path, input_tokens, output_tokens, logits_entropies, scores_entropies, generated_texts):
        self.logger.info(f"Saving results to {file_path}")
        results = {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                'logits': logits_entropies, 'scores': scores_entropies, 'generated_texts': generated_texts}
        with file_path.open("wb") as f:
            pickle.dump(results, f)
        self.logger.debug(f"Results saved successfully to {file_path}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name", type=str, required=True, help="name of the input data file"
    )
    parser.add_argument(
        "--data_size", type=int, required=False, default=1024, help="size of the input data file"
    )
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument(
        "--batch_size", type=int, required=False, default=16, help="batch size for inference"
    )
    parser.add_argument(
        "--do_sample", type=str, required=False, default='True', help="do sampling for decoding or not"
    )
    parser.add_argument(
        "--temperature", type=float, required=False, default=0.6, help="temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, required=False, default=0.8, help="top_p for sampling")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        required=False,
        default=1024,
        help="max number of tokens to generate",
    )
    parser.add_argument(
        "--write_frequency", type=int, required=False, default=256, help="frequency of writing to disk"
    )
    parser.add_argument(
        "--seed_value", type=int, required=False, default=17, help="random seed"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=False, default='', help="HF cache dir to store model data"
    )
    parser.add_argument(
        "--generation_stage", type=str, required=True, help="initial, internal, external, or individual"
    )
    parser.add_argument(
        "--select_new_dataset_samples", type=str, required=False, default='False', help="select new samples or not"
    )
    parser.add_argument(
        "--explicit_prompting", type=str, required=False, default='True', help="prompt with explicit instructions"
    )
    parser.add_argument(
        "--log_level", type=str, required=False, default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Logging level"
    )
  
    
    # Parse known and dataset-specific arguments
    args, extra_args = parser.parse_known_args()
    
    # Set up logger
    logger_name = f"generator_{args.model_name.split('/')[1].replace('-', '_')}_{args.data_name}_{args.generation_stage}"
    logger = setup_logger(logger_name, args.log_level, "generation_logs")
    logger.info(f"Starting generator with arguments: {args}")
    
    data_args = {}
    for i in range(0, len(extra_args), 2):
        if i + 1 < len(extra_args):
            key = extra_args[i].lstrip("-")  # Remove leading '--'
            value = extra_args[i + 1]
            data_args[key] = value
        else:
            logger.warning(f"Invalid argument pair: {extra_args[i]}")
            
    # run generator
    try:
        generator = Generator(args, logger)
        input_data = generator.create_input_list(**data_args)
        generator.run_model(input_data=input_data)
        logger.info("Generation completed successfully")
    except Exception as e:
        logger.critical(f"Fatal error during execution: {str(e)}", exc_info=True)
        raise
