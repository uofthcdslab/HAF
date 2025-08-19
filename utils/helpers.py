import numpy as np
import re
import itertools
import torch
import pickle
import os
from pathlib import Path
from sentence_transformers import CrossEncoder
from utils.data_path_prefixes import GEN_OUTPUT_PATH, PARSE_OUTPUT_PATH

DECISION_KEY_WORDS = ["determine", "decision", "decide", "classify", "classified", "classification", "categorize", "consider", "identify", "identified"]
MAYBE_TOXIC_KEY_WORDS = ["but", "mild", "partial", "may", "might", "could", "possibly", "some", "likely", "seem", "appear", "suggest", "probab", "potential", "neutral"]
ADD_REASONS_TEMPLATES = ["there are no additional reasons", "i agree with the provided reasons", "there are additional reasons"]

def extract_first_number(filename):
    match = re.search(r"samples_(\d+)-\d+", filename.stem)  
    return int(match.group(1)) if match else float('inf')

def get_cleaned_inputs_outputs(this_batch_input_texts, this_batch_llm_texts, stage, logger):
    logger.info("Starting cleaning of inputs and outputs")
    
    cleaned_inputs = []
    cleaned_outputs = []
    
    if not this_batch_input_texts or not this_batch_llm_texts:
        logger.warning("Empty input or LLM texts received")
        return ["NO INPUT"], ["NO OUTPUT"]
        
    for sample_ix in range(len(this_batch_input_texts)):
        if sample_ix >= len(this_batch_llm_texts):
            logger.warning(f"Index {sample_ix} out of range for LLM texts")
            cleaned_outputs.append("NO OUTPUT")
            cleaned_inputs.append("NO INPUT")
            continue
            
        output_text = this_batch_llm_texts[sample_ix] or ""
        input_text = this_batch_input_texts[sample_ix] or ""
        
        logger.debug(f"Processing sample {sample_ix}")
        
        cleaned_outputs.append(output_text[len(input_text):] if len(output_text) > len(input_text) else "")
        
        try:
            if stage == 'initial':
                cleaned_input = input_text.split("TEXT: \n")[1].split("assistant")[0]
            else:
                cleaned_input = input_text.split("TEXT: \n")[1].split("\n\nREASON")[0]
        except Exception as e:
            logger.error(f"Error cleaning input text: {str(e)}")
            cleaned_input = input_text
            
        cleaned_inputs.append(cleaned_input)
    
    logger.info(f"Completed cleaning {len(cleaned_inputs)} samples")
    return cleaned_inputs, cleaned_outputs 

def extract_decisions(gen_texts_list, logger):
    logger.info("Starting decision extraction")
    
    decisions = []
    decision_sentences = []
    
    if not gen_texts_list:
        logger.warning("Empty generation texts list received")
        return ["NO DECISION"], ["NO DECISION SENTENCE"]
    
    for sample_ix in range(len(gen_texts_list)):
        logger.debug(f"Processing sample {sample_ix}")
            
        sample = gen_texts_list[sample_ix] or ""
        sentences = re.split(r'(?<=[.!?])\s+|\n+', sample.strip()) or [""]
        
        decision_found = False
        best_sentence = None
        best_confidence = 0
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
                
            confidence = 0
            
            # Skip task-related sentences (not actual decisions)
            if re.search(r'\b(need to check|need to answer|need to say|need to provide|we need|instructions|request|question)', clean_sentence, re.IGNORECASE):
                continue
            
            # High confidence patterns (actual decision statements)
            if re.search(r'^(Yes|No|The text is)\b.*\btoxic\b', clean_sentence, re.IGNORECASE):
                confidence = 3
            elif re.search(r'^(The text is)\s+(toxic|not toxic|non-toxic)\b', clean_sentence, re.IGNORECASE):
                confidence = 3
            elif re.search(r'\b(is|isn\'t|not)\s+(toxic|non-toxic)\b', clean_sentence, re.IGNORECASE):
                confidence = 3
            # Declarative statements about toxicity (but exclude task-related ones)
            elif re.search(r'^(It is|This is|Text is)\s.*\btoxic\b', clean_sentence, re.IGNORECASE):
                confidence = 2
            elif re.search(r'\btoxic\b\.$', clean_sentence, re.IGNORECASE) and not re.search(r'\?|check|answer|provide', clean_sentence, re.IGNORECASE):
                confidence = 2
            # Medium confidence (contains decision words and toxic)
            elif re.search(r'\btoxic\b', clean_sentence, re.IGNORECASE) and re.search(r"(" + "|".join(DECISION_KEY_WORDS) + ")", clean_sentence, re.IGNORECASE):
                confidence = 1
            # Low confidence (just contains keywords)
            elif re.search(r"(" + "|".join(DECISION_KEY_WORDS + MAYBE_TOXIC_KEY_WORDS + ['toxic']) + ")", clean_sentence, re.IGNORECASE):
                # Skip questions and prompts
                if re.search(r'\?|request:|instructions:', clean_sentence, re.IGNORECASE):
                    continue
                confidence = 0.5
                
            if confidence > best_confidence:
                best_confidence = confidence
                best_sentence = clean_sentence
                decision_found = True
                
        if decision_found and best_sentence:
            decision_sentences.append(best_sentence)
            decision = get_only_decisions(best_sentence, logger)
            logger.debug(f"Found decision in sample {sample_ix}: {decision} (confidence: {best_confidence})")
            decisions.append(decision)
        else:
            logger.warning(f"No decision found in sample {sample_ix}")
            decision_sentences.append("NO OR UNCLEAR DECISION")
            decisions.append("NO OR UNCLEAR DECISION")
    
    logger.info(f"Completed decision extraction for {len(decisions)} samples")
    return decisions, decision_sentences

def get_only_decisions(clean_sentence, logger):
    if not clean_sentence:
        logger.warning("Empty sentence received for decision extraction")
        return "NO OR UNCLEAR DECISION"
        
    logger.debug(f"Processing decision from sentence: {clean_sentence}")
        
    if re.search(r"(" + "|".join(MAYBE_TOXIC_KEY_WORDS) + ")", clean_sentence, re.IGNORECASE):
        decision = "maybe"
    elif re.search(r"(\b(not|no|non|never|from|without|lack)\b|n't)(?:\s+\w+)*\s+toxic|-toxic|nontoxic", clean_sentence, re.IGNORECASE): 
        decision = "non-toxic"
    # elif re.search(r"(\b(is|as|be|was|were|being|been)\b|'s)(?:\s+\w+)*\s+toxic", clean_sentence, re.IGNORECASE):
    elif re.search(r"(?:\b(?:is|as|be|was|were|being|been)\b|'s)?\s*toxic", clean_sentence, re.IGNORECASE): #perhaps can simply use "in"
        decision = "toxic"
    else:
        logger.warning(f"No clear decision pattern found in: {clean_sentence}")
        decision = "NO OR UNCLEAR DECISION"
        
    logger.debug(f"Extracted decision: {decision}")
    return decision

def extract_reasons(gen_texts_list, decision_sentences, stage, logger):
    logger.info("Starting reasons extraction")
        
    reasons = []
    
    # Pattern for numbered items (1. or 1) format) - works for embedded and standalone
    numbered_pattern = r'\d+[.)]\s*([^0-9][^.]*?)(?=\s*\d+[.)]|\.|\s*$)'
    
    # Pattern for bullet points (- or * at start of line)
    bullet_pattern = r'^[-*]\s*([\s\S]+?)(?=\n^[-*]\s*|\Z)'
    
    for i, sample in enumerate(gen_texts_list):
        logger.debug(f"Processing sample {i}")
        
        sample = sample.replace(decision_sentences[i], '')
        
        # Extract numbered reasons
        numbered_reasons = re.findall(numbered_pattern, sample, re.MULTILINE | re.DOTALL)
        numbered_reasons = [s.strip() for s in numbered_reasons if s.strip()]
        
        # Extract bullet point reasons
        bullet_reasons = re.findall(bullet_pattern, sample, re.MULTILINE)
        bullet_reasons = [s.strip().split('\n\n', 1)[0] for s in bullet_reasons if s.strip()]
        
        # Combine all reasons
        reasons_in_this_sample = numbered_reasons + bullet_reasons
        
        # Filter out short or invalid reasons
        reasons_in_this_sample = [s for s in reasons_in_this_sample if s not in ['', '*'] and len(s) > 20]
        
        logger.debug(f"Removing incorrect reasons in sample {i}")
        del_ix = []
        for jx, item in enumerate(reasons_in_this_sample):
            if re.search(r'\b(reason|reasons)\b', item, re.IGNORECASE) and len(item) < 20:
                del_ix.append(jx)
                break
        if len(del_ix)>0:
            del reasons_in_this_sample[del_ix[0]]
        
        if stage != 'initial':
            reasons_in_this_sample = [reason for reason in reasons_in_this_sample if 'additional reason' not in reason.lower()]
        
        if not reasons_in_this_sample:
            logger.warning(f"No reasons found in sample {i}, using placeholder")
            reasons_in_this_sample = []
            
        logger.debug(f"Found {len(reasons_in_this_sample)} reasons in sample {i}")
        reasons.append(reasons_in_this_sample)
    
    logger.info(f"Completed reasons extraction for {len(reasons)} samples")
    return reasons

def extract_indices_for_one_sample(reasons_tokens, decision_tokens, output_tokens, logger):
    logger.info("Starting index extraction")
    
    # helper
    def get_indices(target_tokens):
        matching_indices = torch.nonzero(torch.isin(output_tokens, target_tokens), as_tuple=True)[0]
        
        # Handle case where no matches are found
        if len(matching_indices) == 0:
            if logger:
                logger.warning(f"No matches found for target tokens: {target_tokens}")
            return (0, 0)  # or return None, depending on how you want to handle this case
        
        matching_indices_diff = torch.cat([torch.tensor([0]), torch.diff(matching_indices)]) 
        cont_matches = (matching_indices_diff == 1).int()
        cont_matches = torch.diff(torch.cat([torch.tensor([0]), cont_matches, torch.tensor([0])]))
        starts = (cont_matches == 1).nonzero(as_tuple=True)[0]
        ends = (cont_matches == -1).nonzero(as_tuple=True)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            if logger:
                logger.warning(f"No continuous sequences found for target tokens: {target_tokens}")
            return (matching_indices[0].item(), matching_indices[-1].item() + 1)
        
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        
        if logger:
            logger.info(f"Found continuous match for target tokens: {target_tokens}")
            
        return ((matching_indices[starts[max_idx]]-1).item(), (matching_indices[ends[max_idx]-1]+1).item())
 
    # for reasons
    if not reasons_tokens or not isinstance(reasons_tokens, list):
        if logger:
            logger.warning("No valid reasons tokens provided")
        reasons_indices = [(0, 0)]
    else:
        reasons_indices = []
        for one_reason_tokens in reasons_tokens:
            reasons_indices.append(get_indices(torch.tensor(one_reason_tokens)))
        
    # for decision
    if not decision_tokens or not isinstance(decision_tokens, list):
        if logger:
            logger.warning("No valid decision tokens provided")
        decision_indices = (0, 0)
    else:
        decision_indices = get_indices(torch.tensor(decision_tokens))
    
    return reasons_indices, decision_indices

def get_additional_decisions(sims_hp, decision_sentences):
    scores = []
    for dix, decision in enumerate(decision_sentences):
        sim = []
        for template in ADD_REASONS_TEMPLATES:
            pred = round(float(sims_hp.predict([decision, template])), 2)
            sim.append(pred)
        scores.append(sim)
        if sim[0] > 0.4 and sim[2] > 0.4:
            print(f"Contradictory similarity scores found for sample index: {dix}")
    return scores    

def get_output_tokens(model_name, data_name, explicit_prompting):
    output_tokens = {}
    stage_list = ['initial', 'internal', 'external', 'individual']
    if explicit_prompting == '': stage_list = stage_list[:-1]
    
    for stage in stage_list:            
        output_tokens[stage] = []
        if stage == 'individual':
            explicit_prompting = ''
            
        directory_path = Path(GEN_OUTPUT_PATH + "/" + model_name.split('/')[1]+'/'+ data_name+'/'+ stage + explicit_prompting)
        pickle_files = sorted(directory_path.glob("*.pkl"), key=extract_first_number)
        for file in pickle_files:
            file = str(file)
            if os.path.basename(file) == 'samples_1-0.pkl':
                continue
            with open(file, "rb") as f:
                llm_generation = pickle.load(f) 
                
            if len(llm_generation['generated_texts']) == 0:
                output_tokens[stage].append([])
                continue
            
            if stage == 'individual':
                for sample_ix in range(len(llm_generation['generated_texts'])):
                    one_sample_outputs = []
                    for ind_ix in range(len(llm_generation['generated_texts'][sample_ix])):
                        inpt = llm_generation['input_tokens'][sample_ix][ind_ix]
                        outt = llm_generation['output_tokens'][sample_ix][ind_ix]
                        one_sample_outputs.append(outt[len(inpt):])
                    output_tokens[stage].append(one_sample_outputs)   
            else:                
                for batch_ix in range(len(llm_generation['generated_texts'])):
                    for sample_ix in range(len(llm_generation['generated_texts'][batch_ix])):
                        inpt = llm_generation['input_tokens'][batch_ix][sample_ix]
                        outt = llm_generation['output_tokens'][batch_ix][sample_ix]
                        output_tokens[stage].append(outt[len(inpt):])
                        
    return output_tokens

def get_parsed_outputs(model_name, data_name, explicit_prompting):
    parsed_outputs = {}
    stage_list = ['initial', 'internal', 'external', 'individual']
    if explicit_prompting == '': stage_list = stage_list[:-1]
    
    for stage in stage_list:
        if stage == 'individual':
            explicit_prompting = ''
        file_path = Path(PARSE_OUTPUT_PATH + "/" + model_name.split('/')[1]+'/'+ data_name+'/'+ stage + explicit_prompting + '/extracted_info.pkl')
        with file_path.open("rb") as f:
            parsed_outputs[stage]  = pickle.load(f)
    return parsed_outputs
    
def get_common_sublists(A, B):
    max_len = 0
    a_idx = b_idx = -1
    dp = {}
    for i in range(len(A)):
        for j in range(len(B)):
            if A[i] == B[j]:
                dp[(i, j)] = dp.get((i-1, j-1), 0) + 1
                if dp[(i, j)] > max_len:
                    max_len = dp[(i, j)]
                    a_idx = i
                    b_idx = j

    if max_len == 0:
        return -1, -1, 0  # need to throw an error here

    return  a_idx - max_len + 1, b_idx - max_len + 1, max_len

def get_mean_std(this_data):
    clean = [x for x in this_data if x is not None and not np.isnan(x)]
    if not clean:
        return np.nan, np.nan
    if len(clean) == 1:
        return clean[0], np.nan

    mean = round(np.mean(clean), 3)
    std = round(np.std(clean, ddof=1), 3)  # sample standard deviation
    return (mean, std)
    
def get_probs_from_entropies(entropies):
    return torch.exp(-entropies)
    
def get_reasons_similarity_matrix(reasons, sims_reasons):
    N = len(reasons)
    similarity_matrix = np.eye(N)
    triu_indices = np.triu_indices(N, k=1)  # Get indices of the upper triangle (excluding diagonal)
    similarity_matrix[triu_indices] = sims_reasons
    similarity_matrix += similarity_matrix.T - np.eye(N)
    return similarity_matrix

def convert_list_to_col_matrix(input_list):
    N = len(input_list) 
    return np.tile(input_list, (N, 1)) # Repeat the list N times   

def get_average_from_matrix(similarity_matrix, tot_nas=0):
    n = similarity_matrix.shape[0] - tot_nas
    if n == 1 or n == 0:
        return np.nan
    count = n * (n - 1)
    return np.nansum(similarity_matrix) / count
    # n = similarity_matrix.shape[0]
    # upper = np.triu(similarity_matrix, k=1) 
    # count = n * (n - 1) / 2 
    # return upper.sum() / count


class SentenceSimilarity:
  
    """A class to compute similarities between texts."""

    def __init__(self, model_name="cross-encoder/stsb-distilroberta-base", logger=None):
        self.logger = logger
        self.logger.info(f"Initializing SentenceSimilarity with model: {model_name}")
        self.similarity_model = CrossEncoder(model_name)
    
    def get_input_reasons_similarities(self, input_texts, reasons):
        self.logger.info("Starting similarity computation")
            
        with_input = []
        between_reasons = []
        
        if not input_texts or not reasons:
            self.logger.warning("Empty input texts or reasons received")
            return [[]], [[]]
            
        for sample_ix in range(len(input_texts)):
            self.logger.debug(f"Processing sample {sample_ix}")
                
            if sample_ix >= len(reasons):
                self.logger.warning(f"Index {sample_ix} out of range for reasons")
                with_input.append([])
                between_reasons.append([])
                continue
                
            # Handle input similarities
            try:
                sentence_pairs = [(input_texts[sample_ix] or "", reason or "") 
                                for reason in reasons[sample_ix]]
                if sentence_pairs:
                    self.logger.debug(f"Computing {len(sentence_pairs)} input-reason similarities")
                    scores = self.predict(sentence_pairs)
                    with_input.append([float(s) for s in scores])
                else:
                    self.logger.warning(f"No valid sentence pairs for sample {sample_ix}")
                    with_input.append([])
            except Exception as e:
                self.logger.error(f"Error computing input similarities: {str(e)}")
                with_input.append([])
            
            # Handle between reasons similarities
            try:
                valid_reasons = [r for r in reasons[sample_ix] if r]
                sentence_pairs = list(itertools.combinations(valid_reasons, 2))
                if sentence_pairs:
                    self.logger.debug(f"Computing {len(sentence_pairs)} between-reason similarities")
                    scores = self.predict(sentence_pairs)
                    between_reasons.append([float(s) for s in scores])
                else:
                    self.logger.warning(f"No valid reason pairs for sample {sample_ix}")
                    between_reasons.append([])
            except Exception as e:
                self.logger.error(f"Error computing between-reason similarities: {str(e)}")
                between_reasons.append([])
        
        self.logger.info(f"Completed similarity computation for {len(with_input)} samples")
        return with_input, between_reasons
    
    def predict(self, sentence_pairs):
        return self.similarity_model.predict(sentence_pairs)        

    