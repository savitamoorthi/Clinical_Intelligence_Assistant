import json
import sys
import os
import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
    answer_relevancy
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,  
    HarmCategory         
)

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

def run_clinical_evaluation():
    """
    Run evaluation on clinical unstructured report QA using RAGAS and Gemini Judge.
    """
    print("Starting Clinical Reports (Unstructured) Evaluation...")

    json_path = os.path.join(current_dir, 'clinical_reports_test.json')
    
    try:
        with open(json_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases from {json_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'clinical_reports_test.json' in {current_dir}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print("   Check for trailing commas in your JSON file!")
        return

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("\nRunning Agent on Clinical Questions...")
    
    for i, item in enumerate(test_data):
        # Handle variations in JSON keys
        q_text = item.get('question', item.get('query'))
        g_truth = item.get('ground_truth', item.get('answer'))
        q_id = item.get('id', f"Q-{i}")
        
        print(f"   Processing {i+1}/{len(test_data)}: {q_id}...", end="\r")
        unique_config = {"configurable": {"thread_id": f"eval_clinical_{q_id}"}}
        
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=q_text)]}, 
                config=unique_config
            )

            generated_answer = result['messages'][-1].content
            retrieved_context_list = result.get('ragas_context', [])
            
            if not retrieved_context_list:
                retrieved_context_list = ["No clinical documents found."]

            questions.append(q_text)
            answers.append(generated_answer)
            contexts.append(retrieved_context_list)
            ground_truths.append(g_truth)

        except Exception as e:
            print(f"\nError on {q_id}: {e}")
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
    
    judge_llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    judge_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    my_run_config = RunConfig(
        timeout=180,
        max_retries=10,
        max_wait=60
    )

    results = evaluate(
        ragas_dataset,
        metrics=metrics_to_run,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=my_run_config,
        batch_size=1, 
        raise_exceptions=False 
    )

    print("Evaluation Results:")
    print(results)
    
    output_file = os.path.join(current_dir, "clinical_eval_results.csv")
    df = results.to_pandas()
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    run_clinical_evaluation()