import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    api_key = "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"

    print("api key found.")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.0)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    print("\nloading logs.json...")
    try:
        with open("logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        print("error: logs.json not found.")
        return

    print("Preparing records for RAGAs...")
    records = []
    for log_entry in logs:
        for item in log_entry.get('items', []):
            system_contexts = [p['context'] for p in item.get('input', []) if p.get('role') == 'system']
            user_queries = [p['context'] for p in item.get('input', []) if p.get('role') == 'user']
            
            output_list = item.get('expectedOutput', [])
            ground_truth_answer = output_list[0].get('content') if output_list and output_list[0] else None

            if system_contexts and user_queries and ground_truth_answer:
                records.append({
                    "id": item.get("id"),
                    "question": user_queries[0],
                    "contexts": system_contexts,
                    "answer": ground_truth_answer,
                    "reference": ground_truth_answer, 
                })

    if not records:
        print("error: No valid records found.")
        return

    print(f"created {len(records)} records. Creating HuggingFace dataset...")
    
    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "answer": [r["answer"] for r in records],
        "reference": [r["reference"] for r in records],
    })

    print("\n evaluating with OpenAI API...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    print("\n evaluated successfully.")

    print("saving results...")
    results_df = result.to_pandas()
    final_output = []
    for i, row in results_df.iterrows():
        final_output.append({
            "id": records[i]["id"],
            "faithfulness": round(float(row.get("faithfulness", 0.0)), 2),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0)), 2),
            "context_precision": round(float(row.get("context_precision", 0.0)), 2),
        })

    output_filename = "output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"results saved to '{output_filename}'.")
    print("\nFinal Output:")
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()