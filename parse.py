import argparse
import os

from transformers import (
    AutoTokenizer,
)
from utils import helpers as hp
from utils.helpers import SentenceSimilarity
from utils.logger_setup import setup_logger

import torch
import numpy as np
import pandas as pd
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from utils.data_path_prefixes import GEN_OUTPUT_PATH, PARSE_OUTPUT_PATH

class HAFParser:
    """A class to extract reasons and other required information for computing HAF"""
    
    def __init__(self, args, logger):
        self.logger = logger
        self.logger.info(f"Initializing HAF parser with model: {args.model_name}, data: {args.data_name}")
           
        # initiate class variables and others to store results
        self.initiate_class_variables(args)
        self.set_required_seeds(seed_value=self.seed_value)

        # initialize tokenizer
        self.logger.info(f"Initializing tokenizer for model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token_id is None:
            # tokenizer.pad_token = tokenizer.eos_token  # use EOS token as PAD token
            self.logger.info("Adding pad token to tokenizer")
            self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.padding_side = "left"  # for decoder-type mdoels
            
        # initiate similarity computing class
        self.sims_hp = SentenceSimilarity(self.similarity_model, self.logger)
        self.logger.info("HAF parser initialization complete")
    
    def set_required_seeds(self, seed_value=17):
        self.logger.info(f"Setting random seeds to {seed_value} for reproducibility")
        # Set the seeds for reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # the below may affect performance
        # torch.backends.cudnn.deterministic = True  # Ensures deterministic algorithms
        # torch.backends.cudnn.benchmark = False    # Ensures determinism

    def initiate_class_variables(self, args):
        self.logger.debug("Setting class variables from arguments")
        # init variables
        self.data_name = args.data_name
        self.model_name = args.model_name
        self.seed_value = args.seed_value
        self.cache_dir = args.cache_dir if args.cache_dir != '' else None
        self.similarity_model = args.similarity_model
        self.stage = args.stage
        self.explicit_prompting = '_explicit' if args.explicit_prompting == 'True' and self.stage != 'individual' else ''
        
        # output variables
        self.total_samples = 0
        self.input_texts = []
        self.decisions = []
        self.decision_sentences = []
        self.reasons = []
        self.sims_input = []
        self.sims_reasons = []
        self.decision_indices = []
        self.reasons_indices = []
        self.entropies_logits = []
        self.entropies_scores = []
        self.decision_relevances = []
        self.reasons_relevances = []
        
    def create_batch_lists(self):
        self.input_texts_batch = []
        self.decisions_batch = []
        self.decision_sentences_batch = []
        self.reasons_batch = []
        self.sims_input_batch = []
        self.sims_reasons_batch = []
        self.decision_indices_batch = []
        self.reasons_indices_batch = []
        self.entropies_logits_batch = []
        self.entropies_scores_batch = []
        self.decision_relevances_batch = []
        self.reasons_relevances_batch = []
    
    def add_batch(self):
        self.input_texts.append(self.input_texts_batch)
        self.decisions.append(self.decisions_batch)
        self.decision_sentences.append(self.decision_sentences_batch)
        self.reasons.append(self.reasons_batch)
        self.sims_input.append(self.sims_input_batch)
        self.sims_reasons.append(self.sims_reasons_batch)
        self.decision_indices.append(self.decision_indices_batch)
        self.reasons_indices.append(self.reasons_indices_batch)
        self.entropies_logits.append(self.entropies_logits_batch)
        self.entropies_scores.append(self.entropies_scores_batch)
        self.decision_relevances.append(self.decision_relevances_batch)
        self.reasons_relevances.append(self.reasons_relevances_batch)
    
    def add_empty_values(self):
         self.input_texts.append("")
         self.decisions.append("")
         self.decision_sentences.append("")
         self.reasons.append([])
         self.sims_input.append([])
         self.sims_reasons.append([])
         self.entropies_logits.append([])
         self.entropies_scores.append([])
         self.decision_relevances.append([])
         self.reasons_relevances.append([])
         self.decision_indices.append([])
         self.reasons_indices.append([])
    
    def parse_llm_generation(self):
        self.logger.info("Starting parse_llm_generation")           
        self.logger.info(f"Using stage type: {self.stage}")
         
        directory_path = Path(GEN_OUTPUT_PATH + "/" + self.model_name.split('/')[1]+'/'+ self.data_name+'/'+ self.stage + self.explicit_prompting)
        self.logger.info(f"Looking for data files in: {directory_path}")
        
        file_count = 0 
        pickle_files = sorted(directory_path.glob("*.pkl"), key=hp.extract_first_number)
    
        for file in tqdm(pickle_files):  
        # for file in tqdm(directory_path.glob("*.pkl")): # list of batches
            file_count += 1
            file = str(file) 
            self.logger.info(f"Processing file: {file}")
            with open(file, "rb") as f:
                llm_generation = pickle.load(f) 
                    
            if self.stage == 'individual' and len(llm_generation['generated_texts']) == 0:
                self.add_empty_values()
                continue
            
            # looping through each batch
            total_batches = len(llm_generation['generated_texts'])
            self.logger.info(f"Found {total_batches} batches in file")
            if self.stage == 'individual': self.create_batch_lists()
            
            for batch_ix in range(total_batches): # batch_ix is the equivalent of sample_ix for individual
                total_samples_this_batch = len(llm_generation['generated_texts'][batch_ix])
                self.total_samples += total_samples_this_batch
                self.logger.debug(f"Processing batch {batch_ix} with {total_samples_this_batch} samples")
                
                # input texts
                this_batch_input_texts = self.tokenizer.batch_decode(llm_generation['input_tokens'][batch_ix], skip_special_tokens=True)
                this_batch_input_texts, this_batch_llm_texts = hp.get_cleaned_inputs_outputs(this_batch_input_texts, llm_generation['generated_texts'][batch_ix], self.stage, self.logger)
                self.input_texts_batch.extend(this_batch_input_texts) if self.stage == 'individual' else self.input_texts.extend(this_batch_input_texts)
                    
                # decisions and reasons
                decisions, decision_sentences = hp.extract_decisions(this_batch_llm_texts, self.logger)
                self.decisions_batch.extend(decisions) if self.stage == 'individual' else self.decisions.extend(decisions)
                self.decision_sentences_batch.extend(decision_sentences) if self.stage == 'individual' else self.decision_sentences.extend(decision_sentences)
                decisions_tokens = self.tokenizer(decision_sentences, add_special_tokens=False)['input_ids']
                reasons = hp.extract_reasons(this_batch_llm_texts, decision_sentences, self.stage, self.logger)
                self.reasons_batch.extend(reasons) if self.stage == 'individual' else self.reasons.extend(reasons)
                
                # similarity scores with input and between reasons
                self.logger.debug("Computing similarity scores")
                with_input, between_reasons = self.sims_hp.get_input_reasons_similarities(this_batch_input_texts, reasons)
                self.sims_input_batch.extend(with_input) if self.stage == 'individual' else self.sims_input.extend(with_input)
                self.sims_reasons_batch.extend(between_reasons) if self.stage == 'individual' else self.sims_reasons.extend(between_reasons)
                
                # token-wise predictive entropies
                self.logger.debug("Processing entropy values")
                self.entropies_logits_batch.extend([entropy.clone() for entropy in llm_generation['logits'][batch_ix]]) if self.stage == 'individual' else self.entropies_logits.extend([entropy.clone() for entropy in llm_generation['logits'][batch_ix]])
                self.entropies_scores_batch.extend([entropy.clone() for entropy in llm_generation['scores'][batch_ix]]) if self.stage == 'individual' else self.entropies_scores.extend([entropy.clone() for entropy in llm_generation['scores'][batch_ix]])
                
                # extract toxicity decision and reasons list for each data point - TODO: modify the below code for batch processing here
                for sample_ix in range(total_samples_this_batch):
                    self.logger.debug(f"Processing sample {sample_ix} in batch {batch_ix} in batch {batch_ix}")
                    
                    # extract (start, end) reason and decision indices - to get relevant entropy values
                    if not reasons[sample_ix]:
                        reasons_tokens = []
                    else:
                        reasons_tokens = self.tokenizer(reasons[sample_ix], add_special_tokens=False)['input_ids']
                    this_sample_input_len = len(llm_generation['input_tokens'][batch_ix][sample_ix])
                    target_ids = llm_generation['output_tokens'][batch_ix][sample_ix].clone()[this_sample_input_len:]
                    reasons_indices, decision_indices = hp.extract_indices_for_one_sample(reasons_tokens, decisions_tokens[sample_ix], target_ids.to('cpu'), self.logger)
                    self.decision_indices_batch.append(decision_indices) if self.stage == 'individual' else self.decision_indices.extend(decision_indices)
                    self.reasons_indices_batch.append(reasons_indices) if self.stage == 'individual' else self.reasons_indices.extend(reasons_indices)
                    
                    # similarity-based relevance for decision and reasons
                    self.logger.debug(f"Computing relevance scores for sample {sample_ix}")
                    self.decision_relevances_batch.append(self.get_relevance_scores_for_sentence(torch.tensor(decisions_tokens[sample_ix]), decisions[sample_ix])) if self.stage == 'individual' else self.decision_relevances.append(self.get_relevance_scores_for_sentence(torch.tensor(decisions_tokens[sample_ix]), decisions[sample_ix]))
                    one_reason_relevance = []
                    for reason_ix in range(len(reasons_tokens)):
                        rel = self.get_relevance_scores_for_sentence(torch.tensor(reasons_tokens[reason_ix]), reasons[sample_ix][reason_ix])
                        one_reason_relevance.append(rel)
                    self.reasons_relevances_batch.append(one_reason_relevance) if self.stage == 'individual' else self.reasons_relevances.extend(one_reason_relevance)
                     
            self.add_batch() if self.stage == 'individual' else None # add rsults of each batch
                                                       
        self.logger.info(f"Processed {file_count} files with a total of {self.total_samples} samples")
        if len(self.input_texts) > 0:
            self.logger.info("Writing results to disk")
            self.write_results_to_disk()
        else:
            self.logger.warning("No input texts found, skipping write to disk")
    
    def get_relevance_scores_for_sentence(self, sentence_tokens, sentence_target_str):
        self.logger.debug(f"Computing relevance scores for sentence: {sentence_target_str[:30]}...")
        sentence_tokens_masked = [sentence_tokens[torch.arange(len(sentence_tokens)) != i] for i in range(len(sentence_tokens))]
        sentence_str_masked = self.tokenizer.batch_decode(sentence_tokens_masked)
        sentence_pairs = [(sentence_target_str, sentence_m) for sentence_m in sentence_str_masked]
        scores = self.sims_hp.predict(sentence_pairs)
        return [float(1-s) for s in scores]
    
    def write_results_to_disk(self):
        results_dict = {"input_texts": self.input_texts,
                        "decisions": self.decisions,
                        "decision_sentences": self.decision_sentences,
                        "reasons": self.reasons,
                        "sims_input": self.sims_input,
                        "sims_reasons": self.sims_reasons,
                        "entropies_logits": self.entropies_logits,
                        "entropies_scores": self.entropies_scores,
                        "decision_relevances": self.decision_relevances,
                        "reasons_relevances": self.reasons_relevances,
                        "decision_indices": self.decision_indices,
                        "reasons_indices": self.reasons_indices}
        directory_path = Path(PARSE_OUTPUT_PATH + "/" + self.model_name.split('/')[1]+'/'+self.data_name+'/'+self.stage+self.explicit_prompting)
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / ("extracted_info.pkl") # TODO: `extracted_path.pkl` stores the results of all datapoints of a model-data combo in a single file - how to do this batches? or should we even do this differently?
        self.logger.info(f"💾 Saving results to {file_path}")
        with file_path.open("wb") as f:
            pickle.dump(results_dict, f)
        torch.cuda.empty_cache()

def do_sanity_checks(model_name, data_name, decisions, decision_sentences, reasons, stage, explicit_prompting, logger):
    if stage == 'individual':
        decisions = [item for sublist in decisions for item in sublist]
        decision_sentences = [item for sublist in decision_sentences for item in sublist]
        reasons = [item for sublist in reasons for item in sublist]
    
    results = []
    results.append(model_name)
    results.append(data_name)
    
    ser = pd.Series(decisions)
    results.append((ser == 'toxic').sum())
    results.append((ser == 'maybe').sum())
    results.append((ser == 'non-toxic').sum())
    results.append((ser == 'NO OR UNCLEAR DECISION').sum())
    results.append(ser[ser == 'NO OR UNCLEAR DECISION'].index.tolist())
    
    ser = pd.Series(decision_sentences)        
    results.append((ser == 'NO OR UNCLEAR DECISION').sum())
    results.append(ser[ser == 'NO OR UNCLEAR DECISION'].index.tolist())
    
    incompl_reasons = 0
    samples_incompl_reasons = 0
    samples_incompl_reasons_ixes = []
    no_reasons = 0
    no_reasons_ixes = []
    for ix in range(len(reasons)):
        if len(reasons[ix]) == 0:
            no_reasons += 1
            no_reasons_ixes.append(ix)
            continue
            
        prev_incompl_reasons = incompl_reasons
        for reason in reasons[ix]:
            if not reason.strip().endswith((".", "?", "!", "\"", "'")):
                incompl_reasons += 1
                
        if incompl_reasons > prev_incompl_reasons:
            samples_incompl_reasons += 1
            samples_incompl_reasons_ixes.append(ix)
    
    results.append(no_reasons)
    results.append(no_reasons_ixes)
    results.append(incompl_reasons)
    results.append(samples_incompl_reasons)
    results.append(samples_incompl_reasons_ixes)

    directory_path = Path(PARSE_OUTPUT_PATH + "/" + model_name.split('/')[1]+'/'+data_name+'/'+stage + explicit_prompting)
    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / ("sanity_checks.pkl") 
    logger.info(f"💾 Saving results of sanity checks to {file_path}")
    with file_path.open("wb") as f:
        pickle.dump(results, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name", type=str, required=True, help="name of the input data file"
    )
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument(
        "--seed_value", type=int, required=False, default=17, help="random seed"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=False, default='', help="HF cache dir to store model data"
    )
    parser.add_argument(
        "--similarity_model", type=str, required=True, default='', help="sentence similarity model"
    )
    parser.add_argument(
        "--stage", type=str, required=True, help="initial, internal, or external"
    )
    parser.add_argument(
        "--explicit_prompting", type=str, required=False, default='False', help="prompt with explicit instructions"
    )
    parser.add_argument(
        "--log_level", type=str, required=False, default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    parser.add_argument(
        "--log_dir", type=str, required=False, default="logs/parser", 
        help="Directory to store log files"
    )
    
    args = parser.parse_args()
    
    # Set up logger using the centralized logger setup
    logger_name = f"haf_parser_{args.model_name.split('/')[1].replace('-', '_')}_{args.data_name}_{args.stage}"
    logger = setup_logger(logger_name, args.log_level, "haf_parser_logs")
    logger.info(f"Starting HAF parsing with model: {args.model_name}, data: {args.data_name}")
    
    try:
        haf_parser = HAFParser(args, logger)
        logger.info("Extracting required info...")
        haf_parser.parse_llm_generation()
        logger.info("HAF parsing complete")
        
        logger.info("Performing sanity checks...")
        do_sanity_checks(haf_parser.model_name, haf_parser.data_name, haf_parser.decisions, haf_parser.decision_sentences, 
                         haf_parser.reasons, haf_parser.stage, haf_parser.explicit_prompting, haf_parser.logger)
        logger.info("Sanity checks complete")
                        
    except Exception as e:
        logger.critical(f"Fatal error during HAF parsing: {str(e)}", exc_info=True)
        raise
