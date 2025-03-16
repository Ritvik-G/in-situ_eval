# In-Situ Evaluator 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Accelerating domain-specific LLM evaluation through strategic subsampling and real-time analysis**

---

## 📖 Overview

Traditional LLM leaderboards often fail to predict performance in specialized domains, while conventional adaptation methods like fine-tuning demand excessive computational resources. To solve this issue, we present you a proof of concept for running Real-time evaluations on your dataset with configurable interface for model, hyperparameters and RAG technique selection -  **In-Situ Evaluator**. Specifically, we employ:
- **Dataset subsampling** for rapid domain-specific benchmarking.
- **API Interface** for choosing between various LLM providers and models.
- **Custom RAG pipelines** for 3 most popular RAG architectures.


By following the examples in this repository, you can:
- **Load your custom dataset**
- **Choose LLM provider and Model for evals** (e.g. Groq (Llama-2, Mixtral), OpenAI(GPT-3.5, GPT-4))
- **Customize model based hyperparameters** to your tailored liking (e.g. Temperature, Top P)
- **Choose between RAG techniques** (e.g. Vanilla RAG, Graph RAG, RAPTOR)
- **Configure hyperparameters for RAG** (e.g. Chunk size, Chunk Overlap)
- **Run Real-time evaluation** of LLMs, hyperparameters, and RAG configurations
- **Compare with metrics** (BLEU, ROUGE, RoBERTa-NLI, etc.)


_Paper preprint coming soon. This repository contains a production-ready proof-of-concept._

---

## ✨ Features

| **Component**         | **Supported Options**                                                                 |
|------------------------|---------------------------------------------------------------------------------------|
| **LLM Providers**           | Groq , OpenAI                                       |
| **RAG Techniques**     | [Vanilla RAG](https://arxiv.org/pdf/2005.11401), [Graph RAG](https://arxiv.org/pdf/2404.16130), [RAPTOR](https://arxiv.org/pdf/2401.18059)|
| **Model Hyperparameters**    | Temperature, Top_P, Stop Sequence, Stream RAG chunk size/overlap             |
| **RAG Hyperparameters** | Chunk Size, Chunk Overlap, Top K | 
| **Proxy Datasets**           | SQuAD (easy), TriviaQA (medium), WikiQA (hard)                                  |
| **Metrics**            | BLEU, ROUGE-L, METEOR, RoBERTa-NLI, Cosine Similarity                                 |

---


---

# ⚡ Running the Codebase

## Prerequisites
- Python 3.8+ (for backend)
- ReactJS (for frontend)
- GROQ/OpenAI API keys (for LLM calls). You can obtain these API keys by following the steps of the respective providers - 
   - [GROQ](https://console.groq.com/keys)
   - [OpenAI](https://platform.openai.com/api-keys)
- (Optional) Before uploading the custom dataset, please ensure it is of `json` file type and is of the following format - 
   ```bash
      [
         {
            "Question": "This is a sample",
            "Context": "This is the context related to the question.",
            "Response": "This is the ground truth answer"
         },
         {
            "Question": "What is the hottest planet in our solar system?",
            "Context": "The planets in our solar system vary in temperature due to their distance from the Sun, atmospheric composition, and other factors.",
            "Response": "Venus is the hottest planet in our solar system, with surface temperatures reaching up to 462°C (864°F), due to its thick atmosphere and runaway greenhouse effect."
         }
      ]
   ```

## Initial Installation
Clone the git repository - 
```bash
git clone https://github.com/Ritvik-G/in-situ_eval.git
cd in-situ_eval
```

## Frontend Setup

To set up the frontend, follow these steps:

1. **Navigate to the Frontend Directory**  
   First, change the directory to the frontend folder:
   ```bash
   cd frontend
2. **Install Dependencies**  
   Use npm to install all the required dependencies:
   ```bash
   npm install
3. **Start the Frontend**  
   Finally, start the frontend server:
   ```bash
   npm start

## Backend Setup

The structure of backend is as follows - 

```bash
   Backend/
    ├── data/ # proxy datasets
    │   ├── squad.json
    │   ├── trivia_qa.json
    │   └── wiki_qa.json
    ├── RAG/
    │   ├── rag.py
    │   ├── raptor.py
    │   ├── graphrag.py
    │   └── model_config.py # LLM caller function 
    ├── Benchmarks/ # Benchmarker that calls data
    │   └── benchmarks.py
    ├── Evaluations/
    │   ├── evaluations.py
    │   └── consolidate_metrics.py
    ├── app.py
    └── requirements.txt

```

1. **Navigate to the Frontend Directory**  
   First, change the directory to the frontend folder:
   ```bash
   cd backend
2. **Install Dependencies**  
   Use pip to install all the required dependencies:
   ```bash
   pip install -r requirements.txt
3. **Run the Backend**  
   Run the backend server. By default, it would be running on `http://localhost:5000/`:
   ```bash
   python app.py
   ```
## Combined Setup
Once both the frontend and backend servers are running, you can access the application via the frontend URL `http://localhost:3000/api` and interact with the application.
