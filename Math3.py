
import re
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from llama import Llama, Dialog
import fire

def clean_response(response):
    # Strip leading and trailing whitespace
    response = response.strip()
    
    # Define a pattern to capture the answer choice
    match = re.search(r'\b([abcde])\b', response)
    
    if match:
        return match.group(1)
    else:
        return None

def run_experiment(generator, prompt_design, test_df, train_df, experiment_name, temperature, top_p, max_gen_len):
    test_results = []
    for index, row in test_df.iterrows():
        if prompt_design == "zeroU":
            dialog = [{"role": "user", "content": f"""Solve the following math question and determine the correct answer.
                 Provide the answer as a single samall letter (without any additional marks or words like 'answer:'), choosing from: "a", "b", "c", "d", "e".
                 After providing the letter answer, explain your rationale and how you arrived at the solution.
```{row['question']}```""" }]
        elif prompt_design == "zeroSU":
            dialog = [{"role": "system", "content": f"""Solve the following math question and determine the correct answer.
                 Provide the answer as a single samall letter (without any additional marks or words like 'answer:'), choosing from: "a", "b", "c", "d", "e".
                 After providing the letter answer, explain your rationale and how you arrived at the solution."""},
                        {"role": "user", "content": f"""```{row['question']}```"""}]
        elif prompt_design == "fewU":
            dialog = [{"role": "user", "content": f"""Solve the following math question and determine the correct answer.
                 Provide the answer as a single samall letter (without any additional marks or words like 'answer:'), choosing from: "a", "b", "c", "d", "e".
                 After providing the letter answer, explain your rationale and how you arrived at the solution.
Examples:
```{train_df.iloc[0]['question']}``` - {train_df.iloc[0]['answer']} Explanation : the percent of alcohol in the solution is ( 0.05 ( 40 ) + 2.5 ) / 50 = 4.5 / 50 = 9 % the answer is a
```{train_df.iloc[2]['question']}``` - {train_df.iloc[2]['answer']} Explanation : total = 100 t = 80 nt = 20 80 * ( 20 / 100 ) = 24 80 * ( 20 / 100 ) = 24 16 + 16 = 32 = > 100 - 32 = 68 % answer : b
```{train_df.iloc[3]['question']}``` - {train_df.iloc[3]['answer']} Explanation : let the price be = rs . 100 , and number of units sold = 100 then , sale value = rs . ( 100 \u00d7 100 ) = rs . 10000 new sale value = rs . ( 170 \u00d7 80 ) = rs . 13600 increase % = 3600 / 10000 \u00d7 100 = 36 % answer : c
...
```{row['question']}```""" }]
        elif prompt_design == "fewSU":
            dialog = [{"role": "system", "content": f"""Solve the following math question and determine the correct answer.
                 Provide the answer as a single samall letter (without any additional marks or words like 'answer:'), choosing from: "a", "b", "c", "d", "e".
                 After providing the letter answer, explain your rationale and how you arrived at the solution."""},
                        {"role": "user", "content": f"""Examples:
```{train_df.iloc[0]['question']}``` - {train_df.iloc[0]['answer']} Explanation : the percent of alcohol in the solution is ( 0.05 ( 40 ) + 2.5 ) / 50 = 4.5 / 50 = 9 % the answer is a
```{train_df.iloc[2]['question']}``` - {train_df.iloc[2]['answer']} Explanation : total = 100 t = 80 nt = 20 80 * ( 20 / 100 ) = 24 80 * ( 20 / 100 ) = 24 16 + 16 = 32 = > 100 - 32 = 68 % answer : b
```{train_df.iloc[3]['question']}``` - {train_df.iloc[3]['answer']} Explanation : let the price be = rs . 100 , and number of units sold = 100 then , sale value = rs . ( 100 \u00d7 100 ) = rs . 10000 new sale value = rs . ( 170 \u00d7 80 ) = rs . 13600 increase % = 3600 / 10000 \u00d7 100 = 36 % answer : c
...
```{row['question']}```"""}]
        elif prompt_design == "fewSUA":
            dialog = [{"role": "system", "content": f"""Solve the following math question and determine the correct answer.
                 Provide the answer as a single samall letter (without any additional marks or words like 'answer:'), choosing from: "a", "b", "c", "d", "e".
                 After providing the letter answer, explain your rationale and how you arrived at the solution."""},
                        {"role": "user", "content": train_df.iloc[0]['question']}, {"role": "assistant", "content": f""" {train_df.iloc[0]['answer']} Explanation : the percent of alcohol in the solution is ( 0.05 ( 40 ) + 2.5 ) / 50 = 4.5 / 50 = 9 % the answer is a"""},
                        {"role": "user", "content": train_df.iloc[2]['question']}, {"role": "assistant", "content": f""" {train_df.iloc[2]['answer']} Explanation : total = 100 t = 80 nt = 20 80 * ( 20 / 100 ) = 24 80 * ( 20 / 100 ) = 24 16 + 16 = 32 = > 100 - 32 = 68 % answer : b"""},
                        {"role": "user", "content": train_df.iloc[3]['question']}, {"role": "assistant", "content": f""" {train_df.iloc[3]['answer']} Explanation : let the price be = rs . 100 , and number of units sold = 100 then , sale value = rs . ( 100 \u00d7 100 ) = rs . 10000 new sale value = rs . ( 170 \u00d7 80 ) = rs . 13600 increase % = 3600 / 10000 \u00d7 100 = 36 % answer : c"""},
                        {"role": "user", "content": f"""```{row['question']}```"""}]

        response = generator.chat_completion(
            [dialog],  # Wrapping dialog in a list to match expected format
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        predicted_answer = response['generation']['content']
        print(index, ' ', predicted_answer)

        test_results.append({'Index': index, 'question': row['question'], 'Predicted_answer': predicted_answer, 'Actual_answer': row['answer']})

    results_df = pd.DataFrame(test_results)

    def check_structural_validity(response):
        # Check if the response starts with a single lowercase letter among a, b, c, d, e
        pattern = r'^[abcde](?!\w)'
        match = re.match(pattern, response.strip().lower())
        return bool(match)

    valid_answer = ['a', 'b', 'c', 'd', 'e']
    results_df['Is_Structurally_Valid'] = results_df['Predicted_answer'].apply(check_structural_validity)
    structural_accuracy = results_df['Is_Structurally_Valid'].mean()

    # Calculate F1 Score
    results_df['Predicted_answer'] = results_df['Predicted_answer'].apply(clean_response)
    results_df = results_df.dropna(subset=['Predicted_answer', 'Actual_answer'])
    f1 = f1_score(results_df['Actual_answer'], results_df['Predicted_answer'], average='micro')

    print(f"Experiment: {experiment_name}")
    print(f"The F1 Score for the predicted answer is: {f1}")
    print(f"The Structural Accuracy is: {structural_accuracy}")

    return f1, structural_accuracy

def main(ckpt_dir: str, tokenizer_path: str, answer_data_path: str, temperature: float = 0.0, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 8, max_gen_len: int = 256, output_file: str = "f1_scores.csv"):
    # Load the dataset
    DS = pd.read_pickle(answer_data_path)
    answer_data = pd.DataFrame(DS, columns=['question', 'answer'])

    # Initialize LLAMA model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Split data into training and testing datasets
    test_samples_per_answer = 49
    train_samples_per_answer = 1
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for answer in answer_data['answer'].unique():
        answer_df = answer_data[answer_data['answer'] == answer]
        test_samples = answer_df.sample(test_samples_per_answer, random_state=1)
        train_samples = answer_df.drop(test_samples.index).sample(train_samples_per_answer, random_state=1)

        test_df = pd.concat([test_df, test_samples])
        train_df = pd.concat([train_df, train_samples])

    # Run experiments
    results = []
    for prompt_design, experiment_name in [("zeroU", "Zero-shot User Prompt"),
                                           ("zeroSU", "Zero-shot System and User Prompt"),
                                           ("fewU", "Few-shot User Prompt"),
                                           ("fewSU", "Few-shot System and User Prompt"),
                                           ("fewSUA", "Few-shot System, User, and Assistant Prompt")]:
        f1, structural_accuracy = run_experiment(generator, prompt_design, test_df, train_df, experiment_name, temperature, top_p, max_gen_len)
        results.append({'Experiment': experiment_name, 'F1 Score': f1, 'Structural Accuracy': structural_accuracy})

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
