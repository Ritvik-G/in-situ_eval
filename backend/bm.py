import json
import re
from datasets import load_dataset
from bs4 import BeautifulSoup
from random import choice

def sanitize_filename(name: str) -> str:
    """Sanitize the dataset name to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

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

def download_clean_rag_samples(dataset_name: str, dataset_config: str):
    """Download and clean the entire RAG dataset."""
    data = []
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return data

    for sample in dataset:
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

        data.append({
            "Question": question, 
            "Context": context, 
            "Response": response
        })

    return data

def process_and_save_datasets(datasets: list):
    """Process datasets and save the entire cleaned dataset to JSON files."""
    if not isinstance(datasets, list):
        print("Error: Expected a list of datasets.")
        return

    for dataset in datasets:
        dataset_name = dataset.get("name")
        dataset_config = dataset.get("config")
        print(f"Processing dataset: {dataset_name}")

        cleaned_samples = download_clean_rag_samples(dataset_name, dataset_config)
        
        if not cleaned_samples:
            print(f"No samples processed for {dataset_name}.")
            continue

        filename = f"{sanitize_filename(dataset_name)}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(cleaned_samples, f, indent=2)
            print(f"Saved {len(cleaned_samples)} samples to {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
        
        print("=" * 100)

def main():
    """Main function to execute the script."""
    with open('config.json', 'r') as file:
        config = json.load(file)
        
    datasets_config = config['BENCHMARKS']
    process_and_save_datasets(datasets_config)

if __name__ == "__main__":
    main()