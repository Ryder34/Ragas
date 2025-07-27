# RAGAs Metrics Integration for LLM Logs

This project fulfills the assignment to integrate RAGAs evaluation metrics (`faithfulness`, `answer_relevancy`, `context_precision`) into a given LLM log file (`logs.json`).

## Project Structure

Your repository should be structured as follows:

```
.
├── logs.json           # The input LLM log data provided for the assignment.
├── local.py/keys.py   # The main Python script to run the evaluation.
├── requirements.txt    # All the dependencies are in this
├── output_local/output.json         # The final JSON output with computed RAGAs scores.
└── README.md           # This file.
```

## Setup and Execution

### 1. Prerequisites
- Python 3.10 or higher
- An active OpenAI API key/local LLM setup

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Add Your API Key
**This is a critical step.** Open the `keys.py` script and replace the API key with your own.

```python
# Replace it with your actual key:
api_key = "sk-proj-YourActualKeyGoesHere..."
```

### 5. Run the Evaluation
Execute the script from your terminal. Make sure `logs.json` is in the same directory.

```bash
python keys.py
```

The script will print its progress and, upon completion, will generate the `output.json` file containing the final scores.

## My Approach

The primary goal was to parse the provided `logs.json` file and structure its contents into a format suitable for the RAGAs `evaluate` function.

1.  **Data Mapping:** Based on the assignment's notes, I mapped the JSON fields to the columns required by RAGAs:
    *   **question:** The content of the `user` role from the `input` array.
    *   **contexts:** A list containing the content of the `system` role.
    *   **answer:** The `content` from the `expectedOutput` object. This is the LLM-generated response that we are evaluating.
    *   **reference:** This is the ground truth or "ideal" answer. As per the final implementation, the `content` from the `expectedOutput` object was used for this field, treating the provided output as the gold standard for evaluation.

2.  **Evaluation Strategy:** My initial approach was to use a free, locally-run open-source model (`microsoft/Phi-3-mini`) to avoid API costs. However, this method proved to be unreliable, resulting in `NaN` scores due to the model's difficulty in consistently following the complex evaluation prompts on CPU-based hardware.

    To ensure robust and repeatable results, I pivoted to the recommended approach of using the **OpenAI API** with the `gpt-3.5-turbo` model. This required setting up both the `ChatOpenAI` and `OpenAIEmbeddings` components with an API key, as both are needed by RAGAs for a full evaluation. This strategy proved to be fast, reliable, and produced the expected numerical scores.

3.  **Output Generation:** After the evaluation, the results from the RAGAs `DataFrame` were processed and mapped back to their original log item `id`. The final scores were rounded to two decimal places and saved to `output.json` as specified.

## Libraries Used

*   **`ragas`**: The core library for running the RAG evaluation.
*   **`datasets`**: A dependency of RAGAs, used to create the evaluation dataset object.
*   **`langchain-openai`**: Provides the necessary wrappers (`ChatOpenAI`, `OpenAIEmbeddings`) to connect RAGAs with the OpenAI API.
*   **`openai`**: The underlying client library for interacting with OpenAI.

## Assumptions and Simplifications

*   **`reference` Source:** The most significant assumption is that the `expectedOutput` provided in `logs.json` is a suitable ground truth (`reference`) for the evaluation. In a production system, this reference might be a separate, human-verified answer.
*   **API Key Handling:** For the purpose of this self-contained assignment, the script requires the API key to be pasted directly into the file. In a real-world application, a more secure method like environment variables should be used.
*   **Data Structure Consistency:** The script assumes that each log item contains a `system` input, a `user` input, and an `expectedOutput` to be valid for evaluation. Items not conforming to this structure are skipped.