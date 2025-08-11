'''
Consolidates sanity checks results of HAF parser
'''
import pandas as pd
import argparse
import pickle
from pathlib import Path
from utils.data_path_prefixes import PARSE_OUTPUT_PATH

def main(args_main):

    model_names = ['meta-llama/Llama-3.2-3B-Instruct', 
               'meta-llama/Llama-3.1-8B-Instruct', 
               'mistralai/Ministral-8B-Instruct-2410',
               'meta-llama/Llama-3.3-70B-Instruct',
               'microsoft/Phi-4-reasoning']
    data_names = ['civil_comments', 'hate_explain', 'implicit_toxicity', 'real_toxicity_prompts', 'toxigen']
    sanity_checks_colnames = ['model', 'data', 'toxic', 'maybe', 'nontoxic', 'unclear', 
                              'unclear_ixes', 'unclear_sents', 'unclear_sents_ixes', 
                              'no_reasons', 'no_reasons_ixes',
                              'incompl_reasons', 'samples_incompl_reasons', 'samples_incompl_reasons_ixes']
       
    rows = []
    for model_name in model_names:
        for data_name in data_names:    
            print(model_name, ' | ', data_name, ' | ')    
            directory_path = Path(PARSE_OUTPUT_PATH + "/" + model_name.split('/')[1]+'/'+data_name+'/'+args_main.stage)
            directory_path.mkdir(parents=True, exist_ok=True)
            file_path = directory_path / ("sanity_checks.pkl") 
            with file_path.open("rb") as f:
                row = pickle.load(f)
            rows.append(row)
            
    sanity_checks_df = pd.DataFrame(rows, columns=sanity_checks_colnames)
    file_path = Path(PARSE_OUTPUT_PATH) / ('sanity_checks_summary_'+args_main.stage+'.csv')
    sanity_checks_df.to_csv(file_path, index=False)

if __name__ == '__main__':
    # set up argument parser
    parser = argparse.ArgumentParser(description='Do sanity checks')
    parser.add_argument('--stage', type=str, default='initial',
                      help='Stage: initial, internal, or external')
    
    args_main = parser.parse_args()
    main(args_main)