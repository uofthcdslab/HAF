import datasets
import requests
import json
import pandas as pd
import numpy as np
import sklearn
from datasets import load_dataset
from pathlib import Path
from utils.logger_setup import setup_logger
from utils.data_path_prefixes import INPUT_DATA_PATH, PROCESSED_DATA_PATH

# PROCESS:
# - check if the processed data exists. if it does (and the user doesnt want new samples), return that. 
# - check if the full dataset exists. if it does, select a sample, record which indices are used in the sample, and return that. 
# - if the full dataset does not exist, load raw data and preprocess data (remove short texts, sample mild and toxic examples, change labels to text and given_label)
# - save the full dataset with indices (so the indices can be referenced and consistent across runs)
# - select a sample of the full dataset (according to the total_samples parameter) as processed data, record which indices are used in the sample, and return that. 


# Setup logger for this module
logger = setup_logger('data_processor', log_dir='data_processor_logs')

def load_data_from_file(data_path):
    """Load data from a file into a pandas DataFrame.
    
    Args:
        data_path (str): Path to the data file (CSV, XLSX, or JSON)
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    logger.info(f"Loading data from file: {data_path}")
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
        logger.debug(f"Loaded CSV file with {len(data)} rows")
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path)
        logger.debug(f"Loaded Excel file with {len(data)} rows")
    elif data_path.endswith('.json'):
        data = pd.read_json(data_path)
        logger.debug(f"Loaded JSON file with {len(data)} rows")
    else:
        logger.error(f"Unsupported file format: {data_path}")
        raise ValueError("Unsupported file format. Please provide a CSV, XLSX, or JSON file.")
    return data    

def extract_mild_and_toxic_samples(data):    
    """Extract balanced samples of mild and toxic content from the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing text and labels
        total_samples (int): Total number of samples to extract
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Balanced dataset with mild (0.5-0.6 toxicity) and toxic (>0.75 toxicity) samples
    """
    logger.info(f"Extracting samples (mild and toxic)")
    # Adjust sample size to not exceed available data
    available_mild = data[(data["given_label"] > 0.5) & (data["given_label"] < 0.6)]
    available_toxic = data[(data["given_label"] > 0.75)]
    
    logger.debug(f"Available mild samples: {len(available_mild)}")
    logger.debug(f"Available toxic samples: {len(available_toxic)}")

    return pd.concat([available_mild, available_toxic], ignore_index=True, sort=False)


def remove_shorter_texts(data, short_text_length, max_text_length):
    """Filter out texts that are too short or too long.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing text data
        short_text_length (int): Minimum text length to keep
        max_text_length (int): Maximum text length to keep
        
    Returns:
        pd.DataFrame: Filtered DataFrame with texts of appropriate length
    """
    logger.info(f"Removing texts shorter than {short_text_length} characters")
    filtered_data = data[(data["input_text"].str.len() > int(short_text_length)) & (data["input_text"].str.len() < int(max_text_length))]
    logger.debug(f"Removed {len(data) - len(filtered_data)} rows, {len(filtered_data)} remaining")
    return filtered_data


def save_processed_sampled_data(data, processed_sampled_data_path):
    """Save processed and sampled data to file, creating backups if necessary.
    
    Args:
        data (pd.DataFrame): Processed data to save
        processed_sampled_data_path (Path): Path where the data should be saved
    """
    directory_path = Path(PROCESSED_DATA_PATH)
    directory_path.mkdir(parents=True, exist_ok=True)
    if processed_sampled_data_path.exists():
        index = 0
        backup_path = Path(str(processed_sampled_data_path) + f"_{index}")
        while backup_path.exists():
            index += 1
            backup_path = Path(str(processed_sampled_data_path) + f"_{index}")

        existing_data = pd.read_csv(processed_sampled_data_path)
        existing_data.to_csv(backup_path, index=False)
        logger.debug(f"Backed up existing data to {backup_path}")
        
    logger.info(f"Saving processed data to {processed_sampled_data_path}")
    data.to_csv(processed_sampled_data_path, index=False)
    logger.debug(f"Saved {len(data)} rows to {processed_sampled_data_path}")


def save_processed_full_data(data, data_name):
    """Save the full processed dataset.
    
    Args:
        data (pd.DataFrame): Full processed dataset
        data_name (str): Name of the dataset for file naming
    """
    processed_full_data_path = Path(INPUT_DATA_PATH) / f"full_{data_name}.csv"
    processed_full_data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_full_data_path, index=False)
    logger.info(f"Saved full dataset to {processed_full_data_path}")


def record_processed_indices(data, data_name):
    """Record which indices from the full dataset have been processed.
    
    Args:
        data (pd.DataFrame): Processed data containing indices
        data_name (str): Name of the dataset
    """
    processed_indices_path = Path(PROCESSED_DATA_PATH) / "processed_indices.json"
    processed_indices = data["index"].tolist()
    
    # Load existing indices if file exists
    if processed_indices_path.exists():
        with open(processed_indices_path, "r") as f:
            existing_indices = json.load(f)
    else:
        existing_indices = {}
        
    # Update with new indices
    if data_name not in existing_indices:
        existing_indices[data_name] = []
    existing_indices[data_name].extend(processed_indices)
    
    # Save updated indices
    with open(processed_indices_path, "w") as f:
        json.dump(existing_indices, f)

def sample_subset(data, total_samples, random_state, data_name):
    """Sample a subset of data while avoiding previously processed indices.
    
    Args:
        data (pd.DataFrame): Full dataset to sample from
        total_samples (int): Number of samples to extract
        random_state (int): Random seed for reproducibility
        data_name (str): Name of the dataset
        
    Returns:
        pd.DataFrame: Sampled subset of the data
    """
    # Load processed indices if they exist
    processed_indices_path = Path(PROCESSED_DATA_PATH) / "processed_indices.json"
    if processed_indices_path.exists():
        with open(processed_indices_path, "r") as f:
            existing_indices = json.load(f)
            if data_name in existing_indices:
                # Filter out already processed indices
                data = data[~data["index"].isin(existing_indices[data_name])]
                logger.debug(f"Filtered out {len(existing_indices[data_name])} already processed indices")
    
    if len(data) < total_samples:
        logger.warning(f"Not enough unprocessed samples available. Requested {total_samples} but only {len(data)} remain")
        total_samples = len(data)

    return data.sample(n=total_samples, random_state=random_state)

def civil_comments(
    self,
    data_path = "civil_comments",
    remove_shorter_text=True,
    short_text_length=64,
    max_text_length=1024,
):
    """Process the civil comments dataset.
    
    Args:
        data_path: Path to raw data
        remove_shorter_text: Whether to filter out short texts
        short_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep
        
    Returns:
        Processed DataFrame containing civil comments samples
    """

    # Check if the processed data exists
    processed_sampled_data_path = Path(PROCESSED_DATA_PATH) / f"processed_{self.data_name}.csv"
    if processed_sampled_data_path.exists() and not self.select_new_dataset_samples:
        logger.info(f"Found existing processed data at {processed_sampled_data_path}")
        return load_data_from_file(processed_sampled_data_path.as_posix())

    # Check if the full dataset exists
    processed_full_data_path = Path(INPUT_DATA_PATH) / f"full_{self.data_name}.csv"
    if processed_full_data_path.exists():
        logger.info(f"Found existing full dataset at {processed_full_data_path}")
        processed_full_data = load_data_from_file(processed_full_data_path.as_posix())
    else:
        logger.info("Processing data from scratch")
        dataset = load_dataset(data_path, split="train")
        keep_columns = ['text', 'toxicity']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in keep_columns])

        # Convert to pandas and rename columns
        processed_full_data = dataset.to_pandas()
        processed_full_data = processed_full_data[keep_columns].rename(
            columns={"text": "input_text", "toxicity": "given_label"}
        )
        logger.debug("Renamed columns from text and toxicity to input_text and given_label")

        # Apply text length filtering if requested
        if remove_shorter_text:
            processed_full_data = remove_shorter_texts(processed_full_data, short_text_length, max_text_length)

        # Save full dataset
        processed_full_data = extract_mild_and_toxic_samples(processed_full_data)
        processed_full_data.insert(0, "index", range(len(processed_full_data)))
        save_processed_full_data(processed_full_data, self.data_name)

    processed_sampled_data = sample_subset(processed_full_data, self.total_samples, self.random_state, self.data_name)
    
    # Save processed data if configured
    if self.save_processed_sampled_data:
        # Save processed data
        save_processed_sampled_data(processed_sampled_data, processed_sampled_data_path)
        record_processed_indices(processed_sampled_data, self.data_name)

    logger.info(f"Completed processing civil_comments dataset, returning {len(processed_sampled_data)} samples")
    return processed_sampled_data
        

def implicit_toxicity(
    self,
    data_path = "jiaxin-wen/Implicit-Toxicity",
    remove_shorter_text=True,
    short_text_length=64,
    max_text_length=1024
):
    """Process the implicit toxicity dataset.
    
    Args:
        data_path: Path to raw data
        remove_shorter_text: Whether to filter out short texts
        short_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep
    Returns:
        Processed DataFrame containing implicit toxic samples
    """

    # check if the processed data exists
    processed_sampled_data_path = Path(PROCESSED_DATA_PATH) / f"processed_{self.data_name}.csv"
    if processed_sampled_data_path.exists() and not self.select_new_dataset_samples:
        logger.info(f"Found existing processed data at {processed_sampled_data_path}")
        return load_data_from_file(processed_sampled_data_path.as_posix())
    
    processed_full_input_data_path = Path(INPUT_DATA_PATH) / f"full_{self.data_name}.csv"
    if processed_full_input_data_path.exists():
        logger.info(f"Found existing full dataset at {processed_full_input_data_path}")
        processed_full_data = load_data_from_file(processed_full_input_data_path.as_posix())
    else:
        logger.info("Processing data from scratch")
        dataset = load_dataset(
            data_path,
            data_files="train/aug-train.json"
        )
        processed_full_data = pd.DataFrame(dataset["train"])
        logger.debug(f"Loaded {len(processed_full_data)} rows")
        
        # Format text by combining context and response
        logger.debug("Formatting text with context and response") 
        processed_full_data = processed_full_data.rename(columns={'label': 'given_label'})
        processed_full_data = pd.DataFrame({
            "input_text": processed_full_data.apply(
                lambda row: f"CONTEXT:\n{row['context']}\n\nRESPONSE:\n{row['response']}", axis=1
            ),
            "given_label": processed_full_data['given_label']
        })

        logger.debug(f"Created DataFrame with {len(processed_full_data)} rows")

        # Filter for toxic samples and apply length constraints
        processed_full_data = processed_full_data[processed_full_data["given_label"] == 1]
        if remove_shorter_text:
            processed_full_data = remove_shorter_texts(processed_full_data, short_text_length, max_text_length)
        
        # Save full dataset
        processed_full_data.insert(0, "index", range(len(processed_full_data)))
        save_processed_full_data(processed_full_data, self.data_name)

    # Sample subset of data
    processed_sampled_data = sample_subset(processed_full_data, self.total_samples, self.random_state, self.data_name)
    
    # Save processed data if configured
    if self.save_processed_sampled_data:
        # Save processed data
        save_processed_sampled_data(processed_sampled_data, processed_sampled_data_path)
        record_processed_indices(processed_sampled_data, self.data_name)

    logger.info(f"Completed processing implicit_toxicity dataset, returning {len(processed_sampled_data)} samples")
    return processed_sampled_data

def hate_explain(
    self,
    data_path: str = "https://raw.githubusercontent.com/hate-alert/HateXplain/refs/heads/master/Data/dataset.json",
    remove_shorter_text: bool = True,
    short_text_length: int = 64,
    max_text_length: int = 1024
) -> pd.DataFrame:
    """Process HateXplain dataset.
    
    Args:
        data_path: URL to raw HateXplain dataset JSON
        remove_shorter_text: Whether to filter out short texts
        short_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep
        
    Returns:
        Processed DataFrame containing hate speech samples with scores
    """
    processed_sampled_data_path = Path(PROCESSED_DATA_PATH) / f"processed_{self.data_name}.csv"
    if processed_sampled_data_path.exists() and not self.select_new_dataset_samples:
        logger.info(f"Found existing processed data at {processed_sampled_data_path}")
        return load_data_from_file(processed_sampled_data_path.as_posix())

    processed_full_input_data_path = Path(INPUT_DATA_PATH) / f"full_{self.data_name}.csv"
    if processed_full_input_data_path.exists():
        logger.info(f"Found existing full dataset at {processed_full_input_data_path}")
        processed_full_data = load_data_from_file(processed_full_input_data_path.as_posix())
    else:
        train_ids = requests.get("https://raw.githubusercontent.com/hate-alert/HateXplain/refs/heads/master/Data/post_id_divisions.json")
        train_ids.raise_for_status()  # Raise exception for bad status codes
        train_ids = train_ids.json()['train']

        logger.info(f"Processing data from {data_path}")
        response = requests.get(data_path)
        response.raise_for_status()  # Raise exception for bad status codes
        response_data = response.json()
        
        # preprocess all data
        texts = []
        scores = []
        
        for id_ in train_ids:
            annotators = response_data[id_]['annotators']
            non_normal_count = sum(a['label'] != 'normal' for a in annotators)
            if non_normal_count > 1:  # Only keep samples with non_normal_count > 1
                scores.append(non_normal_count / len(annotators))
                texts.append(' '.join(response_data[id_]['post_tokens']))
        
        # Create DataFrame with only valid texts and scores
        processed_full_data = pd.DataFrame({
            'input_text': texts,
            'given_label': scores
        })

        if remove_shorter_text:
            processed_full_data = remove_shorter_texts(processed_full_data, short_text_length, max_text_length)

        # Save full dataset (so the indices can be referenced and consistent across runs)
        processed_full_data.insert(0, "index", range(len(processed_full_data)))
        save_processed_full_data(processed_full_data, self.data_name)
    
    # Sample subset of data
    processed_sampled_data = sample_subset(processed_full_data, self.total_samples, self.random_state, self.data_name)
    
    # Save processed data if configured
    if self.save_processed_sampled_data:
        # Save processed data
        save_processed_sampled_data(processed_sampled_data, processed_sampled_data_path)
        record_processed_indices(processed_sampled_data, self.data_name)

    logger.info(f"Completed processing hate_explain dataset, returning {len(processed_sampled_data)} samples")
    return processed_sampled_data

def real_toxicity_prompts(
    self,
    data_path: str = "allenai/real-toxicity-prompts",
    remove_shorter_text: bool = True,
    short_text_length: int = 64,
    max_text_length: int = 1024
) -> pd.DataFrame:
    """Process Real Toxicity Prompts dataset.
    
    Args:
        data_path: Path to dataset on HuggingFace
        remove_shorter_text: Whether to filter out short texts
        short_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep
        
    Returns:
        Processed DataFrame containing toxicity samples with scores
    """
    logger.info(f"Processing real_toxicity_prompts dataset from {data_path}")
    
    processed_sampled_data_path = Path(PROCESSED_DATA_PATH) / f"processed_{self.data_name}.csv"
    
    if processed_sampled_data_path.exists() and not self.select_new_dataset_samples:
        logger.info(f"Found existing processed data at {processed_sampled_data_path}")
        return load_data_from_file(processed_sampled_data_path.as_posix())

    processed_full_input_data_path = Path(INPUT_DATA_PATH) / f"full_{self.data_name}.csv"
    if processed_full_input_data_path.exists():
        logger.info(f"Found existing full dataset at {processed_full_input_data_path}")
        processed_full_data = load_data_from_file(processed_full_input_data_path.as_posix())
    else:
        logger.info("Processing data from scratch")       
        dataset = datasets.load_dataset(data_path, split="train")
        processed_full_data = pd.DataFrame(dataset)
        
        # Extract toxicity scores and combine prompt + continuation texts
        toxicity_scores = processed_full_data.apply(
            lambda row: row['continuation']['toxicity'], 
            axis=1
        ).tolist()
        
        texts = processed_full_data.apply(
            lambda row: f"{row['prompt']['text']}{row['continuation']['text']}", 
            axis=1
        ).tolist()
        
        # Create DataFrame with texts and scores
        processed_full_data = pd.DataFrame({
            'input_text': texts,
            'given_label': toxicity_scores
        })

        if remove_shorter_text:
            processed_full_data = remove_shorter_texts(processed_full_data, short_text_length, max_text_length)
        
        processed_full_data = extract_mild_and_toxic_samples(processed_full_data)
        processed_full_data.insert(0, "index", range(len(processed_full_data)))
        save_processed_full_data(processed_full_data, self.data_name)

    # Sample subset of data
    processed_sampled_data = sample_subset(processed_full_data, self.total_samples, self.random_state, self.data_name)
    
    # Save processed data if configured
    if self.save_processed_sampled_data:
        # Save processed data
        save_processed_sampled_data(processed_sampled_data, processed_sampled_data_path)
        record_processed_indices(processed_sampled_data, self.data_name)

    logger.info(f"Completed processing real_toxicity_prompts dataset, returning {len(processed_sampled_data)} samples")
    return processed_sampled_data

def toxigen(
    self, 
    data_path: str = "toxigen/toxigen-data",
    remove_shorter_text: bool = True,
    short_text_length: int = 64,
    max_text_length: int = 1024
) -> pd.DataFrame:
    """Process Toxigen dataset.
    
    Args:
        data_path: Path to dataset on HuggingFace
        remove_shorter_text: Whether to filter out short texts
        short_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep
        
    Returns:
        Processed DataFrame containing toxicity samples with scores
    """
    logger.info(f"Processing toxigen dataset from {data_path}")
    
    processed_sampled_data_path = Path(PROCESSED_DATA_PATH) / f"processed_{self.data_name}.csv"
    
    if processed_sampled_data_path.exists() and not self.select_new_dataset_samples:
        logger.info(f"Found existing processed data at {processed_sampled_data_path}")
        return load_data_from_file(processed_sampled_data_path.as_posix())

    processed_full_input_data_path = Path(INPUT_DATA_PATH) / f"full_{self.data_name}.csv"
    if processed_full_input_data_path.exists():
        logger.info(f"Found existing full dataset at {processed_full_input_data_path}")
        processed_full_data = load_data_from_file(processed_full_input_data_path.as_posix())
    else:
        logger.info("Processing data from scratch")       
        dataset = datasets.load_dataset(data_path, split="train")
        processed_full_data = pd.DataFrame(dataset)
    
        # Scale human toxicity scores to [0,1] range
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        toxicity_scores = scaler.fit_transform(
            np.array(processed_full_data['toxicity_human']).reshape(-1, 1)
        ).flatten().tolist()
        
        processed_full_data = pd.DataFrame({
            'input_text': processed_full_data['text'],
            'given_label': toxicity_scores
        })
        
        # Filter by text length before sampling if specified
        if remove_shorter_text:
            processed_full_data = remove_shorter_texts(processed_full_data, short_text_length, max_text_length)
        
        processed_full_data = extract_mild_and_toxic_samples(processed_full_data)
        processed_full_data.insert(0, "index", range(len(processed_full_data)))
        save_processed_full_data(processed_full_data, self.data_name)

    # Sample subset of data
    processed_sampled_data = sample_subset(processed_full_data, self.total_samples, self.random_state, self.data_name)
    
    # Save processed data if configured
    if self.save_processed_sampled_data:
        # Save processed data
        save_processed_sampled_data(processed_sampled_data, processed_sampled_data_path)
        record_processed_indices(processed_sampled_data, self.data_name)

    logger.info(f"Completed processing toxigen dataset, returning {len(processed_sampled_data)} samples")
    return processed_sampled_data
    
    