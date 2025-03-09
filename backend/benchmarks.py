import json
from datasets import load_dataset
from bs4 import BeautifulSoup
from random import choice
import random
import os

def clean_html(html_content: str) -> str:
    """Clean HTML content by removing HTML tags."""
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(strip=True)
    return ""

def clean_description(description: list) -> str:
    """Clean description list into a single string."""
    if description:
        return " ".join(description)
    return ""


def stream_clean_rag_samples(dataset_name: str, dataset_config: str, num_samples: int):
    """Stream and clean RAG samples from a dataset."""
    data = []
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return data

    count = 0
    for sample in dataset:
        if count >= num_samples:
            break

        # Extract question
        question = sample.get("question", None)
        if isinstance(question, dict):
            question = question.get("text", None)

        context = sample.get("document_title", None) or sample.get("context", None) or sample.get("document", None) or sample.get("search_results", None)

        # Clean HTML
        if context and isinstance(context, dict):
            if "html" in context:
                context = clean_html(context["html"])
            elif "description" in context:
                context = clean_description(context["description"])

        try:
            response = choice(sample.get("answer", {}).get("aliases", [None])) or sample.get("long_answer_candidates", None) or sample.get("answers", {}).get("text", [None])[0]
        except Exception as e:
            response = sample.get("answer", None) or sample.get("long_answer_candidates", None) or sample.get("answers", {}).get("text", [None])[0]

        missing_fields = []
        if question is None:
            missing_fields.append("question")
        if context is None:
            missing_fields.append("context/document/search_results")
        if response is None:
            missing_fields.append("response/long_answer_candidates/answer")

        if missing_fields:
            print(f"Sample {count + 1} - Missing fields: {', '.join(missing_fields)}")

        data.append({"Question": question, "Context": context, "Response": response})
        count += 1

    return data

def get_random_rows(data: list, num_samples: int) -> list:
    """Randomly select num_samples from the data list."""
    if num_samples > len(data):
        print(f"Warning: Requested {num_samples} samples but only {len(data)} available. Returning all.")
        return data
    return random.sample(data, num_samples)

def load_datasets_from_json(datasets: list, num_samples: int):
    """Load datasets from JSON files in the data folder and return random samples."""
    if not isinstance(datasets, list):
        print("Error: Expected a list of datasets.")
        return {}

    full_data = {}
    for dataset in datasets:
        dataset_name = dataset.get("name")
        if not dataset_name:
            print("Dataset missing 'name' key. Skipping.")
            continue

        file_path = os.path.join("data", f"{dataset_name}.json")
        print(f"Loading samples from: {file_path}")

        # Load data from JSON file
        try:
            with open(file_path, 'r') as f:
                dataset_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found. Skipping.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {file_path}. Skipping.")
            continue

        # Get random samples from the loaded data
        full_data[dataset_name] = get_random_rows(dataset_data, num_samples)
        print(f"Successfully sampled {len(full_data[dataset_name])} entries")
        print("=" * 100)

    return full_data

def main():
    """Main function to run the script independently."""
    with open('config.json','r') as file:
        config = json.load(file)
        
    json_file = config['BENCHMARKS'] # Check formatting of this in config.json
    num_samples = 10  # Number of samples to fetch per dataset
    full_data = load_datasets_from_json(json_file, num_samples)
    print(full_data)

if __name__ == "__main__":
    main()




