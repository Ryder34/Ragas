import os
import json
import torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# --- Local Model Setup ---
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def setup_local_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("...loading local LLM: microsoft/Phi-3-mini-4k-instruct")
    llm_model_id = "microsoft/Phi-3-mini-4k-instruct"
    llm_model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if device == "cuda":
        llm_model_kwargs["load_in_8bit"] = True

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, **llm_model_kwargs)
        pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=512)
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        ragas_llm = LangchainLLMWrapper(hf_llm)
        print(" LLM setup complete.")
    except Exception as e:
        print(f"error loading LLM: {e}")
        return None, None
    print("...Loading local Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
    embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model_kwargs = {'device': device}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs=embed_model_kwargs
        )
        print("model setup complete.")
    except Exception as e:
        print(f"error loading embedding model: {e}")
        return ragas_llm, None

    return ragas_llm, embeddings

def main():
    """ Main function to run the RAGAs evaluation using local models. """
    
    # Setup both the LLM and the embedding model
    llm, embeddings = setup_local_models()
    if not llm or not embeddings:
        print("🛑 Exiting due to model setup failure.")
        return

    # Load and prepare data (this part is already correct)
    print("\nLoading logs.json...")
    try:
        with open("logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        print("🛑 Error: logs.json not found.")
        return

    print("Preparing records for RAGAs...")
    records = []
    for log_entry in logs:
        for item in log_entry.get('items', []):
            system_contexts = [p['context'] for p in item.get('input', []) if p.get('role') == 'system']
            user_queries = [p['context'] for p in item.get('input', []) if p.get('role') == 'user']
            output_list = item.get('expectedOutput', [])
            answer = output_list[0].get('content') if output_list and output_list[0] else None

            if system_contexts and user_queries and answer:
                records.append({
                    "id": item.get("id"),
                    "question": user_queries[0],
                    "contexts": system_contexts,
                    "answer": answer,
                    "reference": answer,
                })

    if not records:
        print("🛑 Error: No valid records found to evaluate.")
        return

    print(f"Created {len(records)} records. Creating HuggingFace dataset...")
    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "answer": [r["answer"] for r in records],
        "reference": [r["reference"] for r in records],
    })

    # =================================================================== #
    # ✨ AND THE FINAL FIX IS HERE ✨
    # We now pass BOTH the llm AND the embeddings to the evaluate function.
    # =================================================================== #
    print("\nStarting RAGAs evaluation with local models... (This will be slow)")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings, # <-- Pass the local embedding model here
        raise_exceptions=False
    )
    print("\n✅ Evaluation complete.")

    print("📊 Formatting and saving results...")
    results_df = result.to_pandas()
    final_output = []
    for i, row in results_df.iterrows():
        final_output.append({
            "id": records[i]["id"],
            "faithfulness": round(float(row.get("faithfulness", 0.0)), 2),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0)), 2),
            "context_precision": round(float(row.get("context_precision", 0.0)), 2),
        })

    output_filename = "output_local.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Done! Results saved to '{output_filename}'.")
    print("\nFinal Output:")
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()