

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from llama import Llama, Dialog
import fire

def clean_response(response):
    response = response.strip().lower()
    if 'positive' in response:
        return 'positive'
    elif 'negative' in response:
        return 'negative'
    else:
        return None

def run_experiment(generator, prompt_design, test_df, train_df, experiment_name, temperature, top_p, max_gen_len):
    test_results = []
    for index, row in test_df.iterrows():
        if prompt_design == "zeroU":
            dialog = [{"role": "user", "content": f"""Determine the sentiment of the movie based on the provided review:
             For the review provided, classify its sentiment as a single word (without other marks or words like 'sentiment:'), either "positive", or "negative".
```{row['review']}```""" }]
        elif prompt_design == "zeroSU":
            dialog = [{"role": "system", "content": f"""Determine the sentiment of the movie based on the provided review.
             For the review provided, classify its sentiment as a single word (without other marks or words like 'sentiment:'), either "positive", or "negative"."""},
                        {"role": "user", "content": f"""```{row['review']}```"""}]
        elif prompt_design == "fewU":
            dialog = [{"role": "user", "content": f"""Determine the sentiment of the movie based on the provided review:
             For the review provided, classify its sentiment as a single word (without other marks or words like 'sentiment:'), either "positive", or "negative".
Examples:
```{train_df.iloc[0]['review']}``` - {train_df.iloc[0]['sentiment']}
```{train_df.iloc[2]['review']}``` - {train_df.iloc[2]['sentiment']}
```{train_df.iloc[1]['review']}``` - {train_df.iloc[1]['sentiment']}
...
```{row['review']}```""" }]
        elif prompt_design == "fewSU":
            dialog = [{"role": "system", "content": f"""Determine the sentiment of the movie based on the provided review.
             For the review provided, classify its sentiment as a single word (without other marks or words like 'sentiment:'), either either "positive", or "negative"."""},
                        {"role": "user", "content": f"""Examples:
```{train_df.iloc[0]['review']}``` - {train_df.iloc[0]['sentiment']}
```{train_df.iloc[2]['review']}``` - {train_df.iloc[2]['sentiment']}
```{train_df.iloc[1]['review']}``` - {train_df.iloc[1]['sentiment']}
...
```{row['review']}```"""}]
        elif prompt_design == "fewSUA":
            dialog = [{"role": "system", "content": f"""Determine the sentiment of the movie based on the provided review.
            For the review provided, classify its sentiment as a single word (without other marks or words like 'sentiment:'), either "positive", or "negative"."""},
                        {"role": "user", "content": train_df.iloc[0]['review']}, {"role": "assistant", "content": train_df.iloc[0]['sentiment']},
                        {"role": "user", "content": train_df.iloc[2]['review']}, {"role": "assistant", "content": train_df.iloc[2]['sentiment']},
                        {"role": "user", "content": train_df.iloc[1]['review']}, {"role": "assistant", "content": train_df.iloc[1]['sentiment']},
                        {"role": "user", "content": f"""```{row['review']}```"""}]
        response = generator.chat_completion(
            [dialog],  # Wrapping dialog in a list to match expected format
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        predicted_sentiment = response['generation']['content']
        print(index, ' ', predicted_sentiment)

        test_results.append({'Index': index, 'review': row['review'], 'Predicted_sentiment': predicted_sentiment, 'Actual_sentiment': row['sentiment']})

    results_df = pd.DataFrame(test_results)

    # Calculate structural accuracy before cleaning
    valid_sentiment =['positive', 'negative']
    results_df['Is_Structurally_Valid'] = results_df['Predicted_sentiment'].apply(lambda x: x.strip().lower() in valid_sentiment)
    structural_accuracy = results_df['Is_Structurally_Valid'].mean()


    # Calculate F1 Score
    results_df['Predicted_sentiment'] = results_df['Predicted_sentiment'].apply(clean_response)
    results_df = results_df.dropna(subset=['Predicted_sentiment', 'Actual_sentiment'])
    f1 = f1_score(results_df['Actual_sentiment'], results_df['Predicted_sentiment'], average='micro')


    print(f"Experiment: {experiment_name}")
    print(f"The F1 Score for the predicted sentiment is: {f1}")
    print(f"The Structural Accuracy is: {structural_accuracy}")

    return f1, structural_accuracy

def main(ckpt_dir: str, tokenizer_path: str, sentiment_data_path: str, temperature: float = 0.0, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 8, max_gen_len: int = 256, output_file: str = "f1_scores.csv"):
    # Load the dataset
    DS = pd.read_pickle(sentiment_data_path)
    sentiment_data = pd.DataFrame(DS, columns=['review', 'sentiment'])

    # Initialize LLAMA model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Split data into training and testing datasets
    test_samples_per_sentiment = 996
    train_samples_per_sentiment = 4
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for sentiment in sentiment_data['sentiment'].unique():
        sentiment_df = sentiment_data[sentiment_data['sentiment'] == sentiment]
        test_samples = sentiment_df.sample(test_samples_per_sentiment, random_state=1)
        train_samples = sentiment_df.drop(test_samples.index).sample(train_samples_per_sentiment, random_state=1)

        test_df = pd.concat([test_df, test_samples])
        train_df = pd.concat([train_df, train_samples])

    # Run experiments
    results = []
    for prompt_design, experiment_name in [
	("zeroU", "Zero-shot User Prompt"),
                                          ("zeroSU", "Zero-shot System and User Prompt"),
                                          ("fewU", "Few-shot User Prompt"),
                                          ("fewSU", "Few-shot System and User Prompt"),
                                           ("fewSUA", "Few-shot System, User, and Assistant Prompt")
					   ]:
        f1, structural_accuracy = run_experiment(generator, prompt_design, test_df, train_df, experiment_name, temperature, top_p, max_gen_len)
        results.append({'Experiment': experiment_name, 'F1 Score': f1, 'Structural Accuracy': structural_accuracy})

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
