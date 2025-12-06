import json
import sys
import os
import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
    answer_relevancy
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

load_dotenv(os.path.join(parent_dir, '.env'))

try:
    from agent.supervisor import app
    print("Successfully imported 'app' from agent.supervisor")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def run_genbi_evaluation():
    """
    Run evaluation on clinical structured data using RAGAS and Gemini Judge.
    """
    print("Starting GenBI (SQL) Evaluation...")

    json_path = os.path.join(current_dir, 'genbi_test.json')
    
    try:
        with open(json_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases from {json_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'genbi_test.json' in {current_dir}")
        return

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("Running Agent on Test Cases...")
    
    for i, item in enumerate(test_data):
        q_text = item['question']
        g_truth = item['ground_truth']
        
        print(f"   Processing {i+1}/{len(test_data)}: {item['id']}...", end="\r")

        unique_config = {"configurable": {"thread_id": f"eval_genbi_{item['id']}"}}
        
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=q_text)]}, 
                config=unique_config
            )

            generated_answer = result['messages'][-1].content
            retrieved_context_list = result.get('ragas_context', [])
            
            if not retrieved_context_list:
                retrieved_context_list = ["No SQL data retrieved."]

            questions.append(q_text)
            answers.append(generated_answer)
            contexts.append(retrieved_context_list)
            ground_truths.append(g_truth)

        except Exception as e:
            print(f"\nError on {item['id']}: {e}")
            questions.append(q_text)
            answers.append("Error during generation.")
            contexts.append(["Error"])
            ground_truths.append(g_truth)

    print("\nInference Complete. Preparing Ragas...")

    data_dict = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths 
    }
    
    ragas_dataset = Dataset.from_dict(data_dict)

    metrics_to_run = [
        faithfulness,
        answer_correctness,
        context_recall,
        context_precision,
        answer_relevancy
    ]

    print("Initializing Gemini 2.5 Flash Judge...")
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    

    print("Calculating Metrics...")
    
    results = evaluate(
        ragas_dataset,
        metrics=metrics_to_run,
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    print("\nEvaluation Results:")
    print(results)
    
    output_file = os.path.join(current_dir, "genbi_eval_results.csv")
    df = results.to_pandas()
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    run_genbi_evaluation()