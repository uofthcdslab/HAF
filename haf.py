import argparse
import numpy as np
import pandas as pd
import json
import json
import torch
import re
import pickle
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoTokenizer,
)
from utils import helpers as hp
from utils.data_path_prefixes import HAF_RESULTS_PATH
from utils.logger_setup import setup_logger
from utils import haf_hyperparams as haf_hp
TARGET_SENTS = {'YES': ['yes additional reasons are required', 'there are additional reasons', 'provided reasons are insufficient'],
 'NO': ['no additional reasons are required','additional reasons are not required', 'there are no additional reasons', 'provided reasons are sufficient'],}


class Haf:
    
    def __init__(self, explicit_prompting='_explicit', use_scores=False, similarity_model="cross-encoder/stsb-distilroberta-base"):
        self.explicit_prompting = explicit_prompting
        self.entropy_mode = 'scores' if use_scores else 'logits'
        self.similarity_model = similarity_model

        # setup logger
        self.logger = setup_logger("haf_computations", "ERROR", "haf_compute_logs")

        # Load model and data details
        with open("utils/model_size_map.json", "r") as file:
            model_size = json.load(file)
        with open("utils/data_path_map.json", "r") as file:
            data_path = json.load(file)
        self.data_names = list(data_path.keys())
        self.model_names = list(model_size.keys())
        self.tokenizers_dict = {}        
        self.sims_hp = hp.SentenceSimilarity(self.similarity_model, self.logger)
        self.individual_decision_imp = {'RS':{'NO': 1.0, 'MAYBE': 0.5, 'YES': 0.1, 'NO OR UNCLEAR DECISION': 0.1},
                                        'RN':{'YES': 1.0, 'MAYBE': 0.5, 'NO': 0.1, 'NO OR UNCLEAR DECISION': 0.1}} 
        
    def compute_samplewise(self):        
        for data_name in self.data_names:
            for model_name in self.model_names:
                print(f"Processing {model_name} on {data_name} data")
                self.logger.info(f"Processing {model_name} on {data_name} data")
                # initializers
                if model_name in self.tokenizers_dict:
                    self.tokenizers_dict[model_name] = self.tokenizers_dict[model_name]
                else:
                    self.tokenizers_dict[model_name] = AutoTokenizer.from_pretrained(model_name)
                
                output_tokens_dict = hp.get_output_tokens(model_name, data_name, self.explicit_prompting)  
                parsed_output_dict = hp.get_parsed_outputs(model_name, data_name, self.explicit_prompting)
                   
                for sample_ix in tqdm(range(len(parsed_output_dict['initial']['input_texts']))):
                    this_sample_result = {}
                                        
                    # relevance dimension metrics
                    ## decision sentence confidence
                    decision_sent = parsed_output_dict['initial']['decision_sentences'][sample_ix]
                    decision_sent_tokens = self.tokenizers_dict[model_name](decision_sent, add_special_tokens=False)['input_ids']
                    # start_ix, end_ix = parsed_output_dict['initial']['decision_indices'][sample_ix]
                    start_ix, end_ix = self.get_indices(torch.tensor(decision_sent_tokens), output_tokens_dict['initial'][sample_ix])
                    out_tokens = output_tokens_dict['initial'][sample_ix][start_ix:end_ix].tolist()
                    confidence, _ = self.compute_confidence(start_ix, out_tokens, 
                                                            decision_sent_tokens,
                                                            parsed_output_dict['initial']['entropies_'+self.entropy_mode][sample_ix],
                                                            parsed_output_dict['initial']['decision_relevances'][sample_ix])
                    this_sample_result['initial_decision_confidence'] = confidence
                    # unclear if we have to check encoding issue here as well?

                    initial_reasons = parsed_output_dict['initial']['reasons'][sample_ix]
                    if len(initial_reasons) == 0:
                        self.logger.warning(f"No reasons found for sample {sample_ix} in {model_name} on {data_name} for initial")
                        self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                        continue
                    
                    this_sample_result['SoS'] = {} 
                    this_sample_result['initial_token_mismatch'] = []
                    this_sample_result['initial_reasons_confidences'] = []
                    reasons_tokens = self.tokenizers_dict[model_name](initial_reasons, add_special_tokens=False)['input_ids']
                    initial_reasons_sims_input = parsed_output_dict['initial']['sims_input'][sample_ix]
                    initial_reasons_sims_reasons = parsed_output_dict['initial']['sims_reasons'][sample_ix]
                    
                    ## computing SoS
                    for reason_ix in range(len(initial_reasons)):
                        start_ix, end_ix = parsed_output_dict['initial']['reasons_indices'][sample_ix][reason_ix]
                        out_tokens = output_tokens_dict['initial'][sample_ix][start_ix:end_ix].tolist()
                        confidence, encoding_issue = self.compute_confidence(start_ix, out_tokens, 
                                                            reasons_tokens[reason_ix],
                                                            parsed_output_dict['initial']['entropies_'+self.entropy_mode][sample_ix],
                                                            parsed_output_dict['initial']['reasons_relevances'][sample_ix][reason_ix])
                        this_sample_result['initial_reasons_confidences'].append(confidence)
                        if encoding_issue: #np.isnan(confidence):
                            self.logger.warning("Issues with decoding: ", model_name, data_name, 'initial', self.explicit_prompting, 
                                    sample_ix, reason_ix, len(reasons_tokens[reason_ix]) - len(out_tokens))
                            this_sample_result['initial_token_mismatch'].append(reason_ix)
                            #this_sample_result['SoS']['reason_'+str(reason_ix)] = np.nan
                            #continue
                        this_sample_result['SoS']['reason_'+str(reason_ix)] = (haf_hp.SoS_Prediction_Weight * confidence) + (haf_hp.SoS_Similarity_Weight * initial_reasons_sims_input[reason_ix])
                       
                    ## computing DiS   
                    if len(initial_reasons) == 1:
                        this_sample_result['DiS_dpp'] = np.nan
                        this_sample_result['DiS_avg'] = np.nan
                    else:
                        tot_nas = 0 #len([conf for conf in initial_reasons_confidences if np.isnan(conf)])
                        prob_weights = hp.convert_list_to_col_matrix(this_sample_result['initial_reasons_confidences'])
                        similarity_matrix = hp.get_reasons_similarity_matrix(initial_reasons, initial_reasons_sims_reasons)
                        assert similarity_matrix.shape == prob_weights.shape, f"Shape mismatch: similarity_matrix {similarity_matrix.shape} vs prob_weights {prob_weights.shape}"
                        this_sample_result['DiS_dpp'] = np.linalg.det(similarity_matrix * prob_weights)
                        this_sample_result['DiS_avg'] = hp.get_average_from_matrix((1-similarity_matrix) * prob_weights, tot_nas=tot_nas)
                     
                    ##--------------------------------------------------------------------
                    
                    # internal and external reliance dimension metrics
                    for reliance_type, metric_name in zip(['internal', 'external'], ['UII', 'UEI']):
                        reliance_reasons = parsed_output_dict[reliance_type]['reasons'][sample_ix]
                        
                        ## decision sentence confidence
                        decision_sent = parsed_output_dict[reliance_type]['decision_sentences'][sample_ix]
                        decision_sent_tokens = self.tokenizers_dict[model_name](decision_sent, add_special_tokens=False)['input_ids']
                        # start_ix, end_ix = parsed_output_dict[reliance_type]['decision_indices'][sample_ix]
                        start_ix, end_ix = self.get_indices(torch.tensor(decision_sent_tokens), output_tokens_dict[reliance_type][sample_ix])
                        out_tokens = output_tokens_dict[reliance_type][sample_ix][start_ix:end_ix].tolist()
                        confidence, _ = self.compute_confidence(start_ix, out_tokens, 
                                                                decision_sent_tokens,
                                                                parsed_output_dict[reliance_type]['entropies_'+self.entropy_mode][sample_ix],
                                                                parsed_output_dict[reliance_type]['decision_relevances'][sample_ix])
                        this_sample_result[reliance_type+'_decision_confidence'] = confidence
                        
                        if len(reliance_reasons) == 0:
                            self.logger.warning(f"No reasons found for sample {sample_ix} in {model_name} on {data_name} for internal") 
                        else: 
                            this_sample_result[metric_name] = {}
                            this_sample_result[reliance_type+'_token_mismatch'] = []
                            this_sample_result[reliance_type+'_reasons_confidences'] = []                   
                            reasons_tokens = self.tokenizers_dict[model_name](reliance_reasons, add_special_tokens=False)['input_ids']
        
                            ## computing UII/UEI
                            for reason_ix in range(len(reliance_reasons)):
                                start_ix, end_ix = parsed_output_dict[reliance_type]['reasons_indices'][sample_ix][reason_ix]
                                out_tokens = output_tokens_dict[reliance_type][sample_ix][start_ix:end_ix].tolist()
                                confidence, encoding_issue = self.compute_confidence(start_ix, out_tokens, 
                                                                    reasons_tokens[reason_ix],
                                                                    parsed_output_dict[reliance_type]['entropies_'+self.entropy_mode][sample_ix],
                                                                    parsed_output_dict[reliance_type]['reasons_relevances'][sample_ix][reason_ix])
                                this_sample_result[reliance_type+'_reasons_confidences'].append(confidence)
                                if encoding_issue:
                                    self.logger.warning("Issues with decoding: ", model_name, data_name, reliance_type, self.explicit_prompting, 
                                            sample_ix, reason_ix, len(reasons_tokens[reason_ix]) - len(out_tokens))
                                    this_sample_result[reliance_type+'_token_mismatch'].append(reason_ix)

                                between_runs_diversity = self.compute_between_runs_similarity(reliance_reasons[reason_ix], initial_reasons, this_sample_result['initial_reasons_confidences'], diversity=True)
                                this_sample_result[metric_name]['reason_'+str(reason_ix)] = (haf_hp.UII_Prediction_Weight * confidence) + (haf_hp.UII_Diversity_Weight * between_runs_diversity)

                            ## computing del-pe
                            reliance_pe = torch.mean(-1.0 * torch.log(torch.tensor(this_sample_result[reliance_type+'_reasons_confidences']))).item()
                            initial_pe = torch.mean(-1.0 * torch.log(torch.tensor(this_sample_result['initial_reasons_confidences']))).item()
                            this_sample_result[reliance_type+'_del_pe'] = reliance_pe - initial_pe


                    if self.explicit_prompting == '':
                        self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                        continue
                    
                    ##--------------------------------------------------------------------
                    # individual reliance dimension metrics      
                                                
                    if sample_ix >= len(output_tokens_dict['individual']):
                        self.logger.warning(f"No data found for sample {sample_ix} in {model_name} on {data_name} for individual")
                        self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                        continue
                                  
                    if len(output_tokens_dict['individual'][sample_ix]) == 0:
                        self.logger.warning(f"No data found for sample {sample_ix} in {model_name} on {data_name} for individual")
                        self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                        continue
                    
                    if parsed_output_dict['initial']['decisions'][sample_ix] == 'NO OR UNCLEAR DECISION':
                        self.logger.warning(f"No decision found for sample {sample_ix} in {model_name} on {data_name} for individual")
                        self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                        continue

                    # find individual decisions  - the below adjustment is required only for a few samples where conflicting decisions are generated - for others this is skewing the interpretation
                    # ind_decision, ind_decision_sent, ind_decision_tokens, ind_decision_indices, ind_decision_relevances = self.find_individual_decision(model_name, output_tokens_dict['individual'][sample_ix])
                    # self.store_individual_decisions_info(sample_ix, model_name, data_name, ind_decision, ind_decision_sent, ind_decision_tokens, ind_decision_indices, ind_decision_relevances)

                    if parsed_output_dict['initial']['decisions'][sample_ix] == 'non-toxic':
                        metric_name = 'RN'
                    else:
                        metric_name = 'RS'
                        
                    this_sample_result[metric_name] = {}
                    this_sample_result['individual_token_mismatch'] = {}
                    this_sample_result['individual_reasons_confidences'] = {}
                    reliance_reasons = parsed_output_dict['individual']['reasons'][sample_ix]
                    this_sample_result['individual_decision_confidence'] = {}
                    
                    for subsample_ix in range(len(output_tokens_dict['individual'][sample_ix])):
                        
                        this_sample_result[metric_name][subsample_ix] = {}
                        
                        ## part-1: computing S/N - decision importance
                        # decision_imp = self.individual_decision_imp[metric_name][ind_decision[subsample_ix]]
                        new_decision = self.get_new_decision(parsed_output_dict['individual']['decision_sentences'][sample_ix][subsample_ix])
                        decision_imp = self.individual_decision_imp[metric_name][new_decision]
                        
                        ## part-2: decision sentence confidence - this is a hack
                        ### original decision confidence - 
                        decision_sent = parsed_output_dict['individual']['decision_sentences'][sample_ix][subsample_ix]
                        decision_sent_tokens = self.tokenizers_dict[model_name](decision_sent, add_special_tokens=False)['input_ids']
                        # start_ix, end_ix = parsed_output_dict['individual']['decision_indices'][sample_ix][subsample_ix]
                        start_ix, end_ix = self.get_indices(torch.tensor(decision_sent_tokens), output_tokens_dict['individual'][sample_ix][subsample_ix])
                        out_tokens = output_tokens_dict['individual'][sample_ix][subsample_ix][start_ix:end_ix].tolist()
                        confidence_orig, _ = self.compute_confidence(start_ix, out_tokens, 
                                                                decision_sent_tokens,
                                                                parsed_output_dict['individual']['entropies_'+self.entropy_mode][sample_ix][subsample_ix],
                                                                parsed_output_dict['individual']['decision_relevances'][sample_ix][subsample_ix])
                        
                        ### new decision confidence - the below adjustment is required only for a few samples where conflicting decisions are generated - for others this is skewing the interpretation
                        # out_tokens = output_tokens_dict['individual'][sample_ix][subsample_ix][ind_decision_indices[subsample_ix][0]:ind_decision_indices[subsample_ix][1]].tolist()
                        # confidence_new, _ = self.compute_confidence(start_ix, out_tokens, 
                        #                                         ind_decision_tokens[subsample_ix],
                        #                                         parsed_output_dict['individual']['entropies_'+self.entropy_mode][sample_ix][subsample_ix],
                        #                                         ind_decision_relevances[subsample_ix])
                        this_sample_result['individual_decision_confidence'][subsample_ix] = confidence_orig # np.nanmean([confidence_orig, confidence_new])

                        ## part-3: computing IS/IN                        
                        if len(reliance_reasons[subsample_ix]) == 0:
                            additional_informativeness = 0 if metric_name == 'RS' else 0.01 # is it too penalizing?
                        else:            
                            additional_informativeness = 0 
                            this_sample_result['individual_token_mismatch'][subsample_ix] = []
                            this_sample_result['individual_reasons_confidences'][subsample_ix] = []       
                            reasons_tokens = self.tokenizers_dict[model_name](reliance_reasons[subsample_ix], add_special_tokens=False)['input_ids']
                            for reason_ix in range(len(reliance_reasons[subsample_ix])):
                                start_ix, end_ix = parsed_output_dict['individual']['reasons_indices'][sample_ix][subsample_ix][reason_ix]
                                out_tokens = output_tokens_dict['individual'][sample_ix][subsample_ix][start_ix:end_ix].tolist()
                                confidence, encoding_issue = self.compute_confidence(start_ix, out_tokens, 
                                                                    reasons_tokens[reason_ix],
                                                                    parsed_output_dict['individual']['entropies_'+self.entropy_mode][sample_ix][subsample_ix],
                                                                    parsed_output_dict['individual']['reasons_relevances'][sample_ix][subsample_ix][reason_ix])
                                this_sample_result['individual_reasons_confidences'][subsample_ix].append(confidence)
                                if encoding_issue: #np.isnan(confidence):
                                    self.logger.warning("Issues with decoding: ", model_name, data_name, 'individual', self.explicit_prompting, 
                                            sample_ix, reason_ix, len(reasons_tokens[reason_ix]) - len(out_tokens))
                                    this_sample_result['individual_token_mismatch'][subsample_ix].append(reason_ix)

                                if metric_name == 'RS':
                                    target_reasons = initial_reasons[:subsample_ix] + initial_reasons[subsample_ix+1:]
                                    target_reasons_confidences = this_sample_result['initial_reasons_confidences'][:subsample_ix] + this_sample_result['initial_reasons_confidences'][subsample_ix+1:]
                                    between_runs_diversity = self.compute_between_runs_similarity(reliance_reasons[subsample_ix][reason_ix], target_reasons, target_reasons_confidences, diversity=True)
                                    additional_informativeness += ((0.5 * confidence) + (0.5 * between_runs_diversity))
                                else:
                                    target_similarity = float(self.sims_hp.predict((reliance_reasons[subsample_ix][reason_ix], initial_reasons[subsample_ix])))
                                    target_similarity = target_similarity * this_sample_result['initial_reasons_confidences'][subsample_ix]
                                    additional_informativeness += ((0.5 * confidence) + (0.5 * target_similarity))
                                    
                            additional_informativeness /= len(reliance_reasons[subsample_ix])
                            
                        if metric_name == 'RS': additional_informativeness = 1 - additional_informativeness 
                        final_rs = decision_imp * this_sample_result['individual_decision_confidence'][subsample_ix] * additional_informativeness
                        this_sample_result[metric_name][subsample_ix] = final_rs
                                                
                    self.save_sample_results(this_sample_result, sample_ix, model_name, data_name)
                 
    def get_new_decision(self, decision_sent):
        # prob_yes = float(self.sims_hp.predict([decision_sent, hp.ADD_REASONS_TEMPLATES[2]]))
        # prob_no = float(max(self.sims_hp.predict([decision_sent, hp.ADD_REASONS_TEMPLATES[0]]),
        #                     self.sims_hp.predict([decision_sent, hp.ADD_REASONS_TEMPLATES[1]])))
        
        # for sufficiency and necessity metrics, the following target sentences reflect the true semantics better
        prob_yes = max([float(self.sims_hp.predict([decision_sent, TARGET_SENTS['YES'][i]])) for i in range(len(TARGET_SENTS['YES']))])
        prob_no = max([float(self.sims_hp.predict([decision_sent, TARGET_SENTS['NO'][i]])) for i in range(len(TARGET_SENTS['NO']))])   
            
        if prob_yes < 0.15 and prob_no < 0.15:
            return 'NO OR UNCLEAR DECISION'
        else:           
            if prob_yes >= prob_no:
                return 'YES'
            else:
                return 'NO'
            
    def compute_confidence(self, start_ix, out_tokens, reason_tokens,
                                 entropies, relevances):
        if out_tokens == [] or reason_tokens == []:
            return np.nan, False
    
        reason_adj, out_adj, max_len = hp.get_common_sublists(reason_tokens, out_tokens)
        
        # some issues with decoding/encoding special characters - "", ', etc.
        encoding_issue = False
        if abs(len(reason_tokens) - max_len) > 4 or abs(len(out_tokens) - max_len) > 4:
            #return np.nan
            encoding_issue = True
        
        # compute token-wise predictive entropies
        pe = entropies[(start_ix+out_adj):(start_ix+out_adj+max_len)].to('cpu')

        # compute token-wise relevances
        rel = relevances[reason_adj:(reason_adj+max_len)]
        rel = [r/sum(rel) for r in rel] # length normalization
                
        # token sar, generative prob
        token_sar = sum([p*r for p, r in zip(pe, rel)])
        return torch.exp(-torch.tensor(token_sar)).item(), encoding_issue
    
    def get_indices(self, target_tokens, output_tokens):
        matching_indices = torch.nonzero(torch.isin(output_tokens, target_tokens), as_tuple=True)[0]
        
        # Handle case where no matches are found
        if len(matching_indices) == 0:
            return (0, 0)  # or return None, depending on how you want to handle this case
        
        matching_indices_diff = torch.cat([torch.tensor([0]), torch.diff(matching_indices)]) 
        cont_matches = (matching_indices_diff == 1).int()
        cont_matches = torch.diff(torch.cat([torch.tensor([0]), cont_matches, torch.tensor([0])]))
        starts = (cont_matches == 1).nonzero(as_tuple=True)[0]
        ends = (cont_matches == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        
        return ((matching_indices[starts[max_idx]]-1).item(), (matching_indices[ends[max_idx]-1]+1).item())        
    
    def compute_between_runs_similarity(self, one_reason, target_reasons, target_reasons_confidences, diversity=True):
        num = 0
        den = 0
        for target_reason, target_confidence in zip(target_reasons, target_reasons_confidences):
            sim = float(self.sims_hp.predict((one_reason, target_reason)))
            if diversity: sim = 1.0 - sim
            num += (sim * target_confidence)
            den += target_confidence
        return num/den if den > 0 else 0.0
    
    def get_indices(self, target_tokens, output_tokens):
        matching_indices = torch.nonzero(torch.isin(output_tokens, target_tokens), as_tuple=True)[0]
        
        # Handle case where no matches are found
        if len(matching_indices) == 0:
            return (0, 0)  # or return None, depending on how you want to handle this case
        
        matching_indices_diff = torch.cat([torch.tensor([0]), torch.diff(matching_indices)]) 
        cont_matches = (matching_indices_diff == 1).int()
        cont_matches = torch.diff(torch.cat([torch.tensor([0]), cont_matches, torch.tensor([0])]))
        starts = (cont_matches == 1).nonzero(as_tuple=True)[0]
        ends = (cont_matches == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        
        return ((matching_indices[starts[max_idx]]-1).item(), (matching_indices[ends[max_idx]-1]+1).item())

    def find_individual_decision(self, model_name, output_tokens):
        
        decision = [] # "NO OR UNCLEAR DECISION"
        decision_sentence = [] # "NO OR UNCLEAR DECISION"
        decision_indices = [] # (0,0)
        decision_tokens = [] # []
        decision_relevances = [] # []
        
        match_words = ['may', 'might', 'could', 'but', 'however', 'though', 'although']
        for batch_ix in range(len(output_tokens)): # for each batch of a sample
            output_text = self.tokenizers_dict[model_name].decode(output_tokens[batch_ix])
            sentences = re.split(r'(?<=[.!?])\s+|\n+', output_text.strip()) or [""]
            decision_found = False
            for sent in sentences[0:2]:
                # prob_yes = float(self.sims_hp.predict([sent, hp.ADD_REASONS_TEMPLATES[2]]))
                # prob_no = float(max(self.sims_hp.predict([sent, hp.ADD_REASONS_TEMPLATES[0]]),
                #                     self.sims_hp.predict([sent, hp.ADD_REASONS_TEMPLATES[1]])))
                prob_yes = max([float(self.sims_hp.predict([sent, TARGET_SENTS['YES'][i]])) for i in range(len(TARGET_SENTS['YES']))])
                prob_no = max([float(self.sims_hp.predict([sent, TARGET_SENTS['NO'][i]])) for i in range(len(TARGET_SENTS['NO']))])   
                
                if prob_yes < 0.15 and prob_no < 0.15:
                    continue # check the next sentence
                
                decision_found = True
                decision_sentence.append(sent) # if at least one prob is > 0.33, then it has alternative decision
                if re.search(r"(" + "|".join(match_words) + ")", sent, re.IGNORECASE):
                    decision.append('MAYBE')
                elif prob_yes >= prob_no:
                    decision.append('YES')
                else:
                    decision.append('NO')            
                break
            
            if not decision_found:
                decision.append('NO OR UNCLEAR DECISION')
                decision_sentence.append('NO OR UNCLEAR DECISION')
                decision_tokens.append([])
                decision_indices.append((0,0))
                decision_relevances.append([])
                continue  
            
            decision_sent_tokens = self.tokenizers_dict[model_name](decision_sentence[batch_ix], add_special_tokens=False)['input_ids']   
            decision_tokens.append(decision_sent_tokens)
            start_ix, end_ix = self.get_indices(torch.tensor(decision_sent_tokens), output_tokens[batch_ix])
            decision_indices.append((start_ix, end_ix))
            rels = self.get_relevance_scores_for_sentence(model_name, torch.tensor(decision_sent_tokens), decision_sentence[batch_ix])
            decision_relevances.append(rels)
        
        return decision, decision_sentence, decision_tokens, decision_indices, decision_relevances 
            
    def get_relevance_scores_for_sentence(self, model_name, sentence_tokens, sentence_target_str):
        sentence_tokens_masked = [sentence_tokens[torch.arange(len(sentence_tokens)) != i] for i in range(len(sentence_tokens))]
        sentence_str_masked = self.tokenizers_dict[model_name].batch_decode(sentence_tokens_masked)
        sentence_pairs = [(sentence_target_str, sentence_m) for sentence_m in sentence_str_masked]
        scores = self.sims_hp.predict(sentence_pairs)
        return [float(1-s) for s in scores]
        
    def store_individual_decisions_info(self, sample_ix, model_name, data_name, ind_decision, ind_decision_sent, ind_decision_tokens, ind_decision_indices, ind_decision_relevances):
        directory_path = Path(HAF_RESULTS_PATH + "/" + model_name.split('/')[1]+'/' + data_name+'/'+'individual_decisions/')
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / (str(sample_ix) + '.pkl')
        self.logger.info(f"💾 Saving results to {file_path}")
        results = {'ind_decision': ind_decision,
                  'ind_decision_sent': ind_decision_sent,
                  'ind_decision_tokens': ind_decision_tokens,
                  'ind_decision_indices': ind_decision_indices,
                  'ind_decision_relevances': ind_decision_relevances}
        with file_path.open("wb") as f:
            pickle.dump(results, f)   

    def save_sample_results(self, results, sample_ix, model_name, data_name):
        if self.explicit_prompting == '':
            directory_path = Path(HAF_RESULTS_PATH + "_naive" + "/" + model_name.split('/')[1]+'/' + data_name+'/')
        else:
            directory_path = Path(HAF_RESULTS_PATH + "/" + model_name.split('/')[1]+'/' + data_name+'/')
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / (str(sample_ix) + '.pkl')
        self.logger.info(f"💾 Saving results to {file_path}")
        with file_path.open("wb") as f:
            pickle.dump(results, f)   
        
    def load_computed_results(self, data_name):
        file_path = HAF_RESULTS_PATH / (data_name + '_' + self.explicit_prompting + '.csv')
        return pd.read_csv(file_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explicit_prompting", type=str, required=False, default='True', help="prompt with explicit instructions"
    )
    parser.add_argument(
        "--use_scores", type=str, required=True, default='False', help="use entropy of logits or scores")
    parser.add_argument(
        "--similarity_model", type=str, required=True, default='cross-encoder/stsb-distilroberta-base', help="semantic similarity model name")
    
    args = parser.parse_args()
    explicit_prompting = '_explicit' if args.explicit_prompting == 'True' else ''
    use_scores = True if args.use_scores == 'True' else False

    haf = Haf(explicit_prompting=explicit_prompting, scores=use_scores, similarity_model=args.similarity_model)
    haf.compute_samplewise()
