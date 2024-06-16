

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from llama import Llama, Dialog
import fire

def clean_response(response):
    response = response.strip().lower()
    if 'comedy' in response:
        return 'comedy'
    elif 'action' in response:
        return 'action'
    elif 'drama' in response:
        return 'drama'
    elif 'horror' in response:
        return 'horror'
    else:
        return None

def run_experiment(generator, prompt_design, test_df, train_df, experiment_name, temperature, top_p, max_gen_len):
    test_results = []
    for index, row in test_df.iterrows():
        if prompt_design == "zeroU":
            dialog = [{"role": "user", "content": f"""Determine the genre of the movie based on the provided plot:
             For the plot provided, classify its genre as a single word (without other marks or words like 'genre:'), either "comedy", "action", "drama", or "horror".
```{row['Plot']}```"""}]
        elif prompt_design == "zeroSU":
            dialog = [{"role": "system", "content": f"""Determine the genre of the movie based on the provided plot.
             For the plot provided, classify its genre as a single word (without other marks or words like 'genre:'), either "comedy", "action", "drama", or "horror"."""},
                                      {"role": "user", "content": f"""```{row['Plot']}```"""}]
        elif prompt_design == "fewU":
            dialog = [{"role": "user", "content": f"""Determine the genre of the movie based on the provided plot:
             For the plot provided, classify its genre as a single word (without other marks or words like 'genre:'), either "comedy", "action", "drama", or "horror".
Examples:
```{train_df.iloc[0]['Plot']}``` - {train_df.iloc[0]['Genre']}
```{train_df.iloc[14]['Plot']}``` - {train_df.iloc[14]['Genre']}
```{train_df.iloc[17]['Plot']}``` - {train_df.iloc[17]['Genre']}
...
```{row['Plot']}```"""}]
        elif prompt_design == "fewSU":
            dialog = [{"role": "system", "content": f"""Determine the genre of the movie based on the provided plot.
             For the plot provided, classify its genre as a single word (without other marks or words like 'genre:'), either "comedy", "action", "drama", or "horror"."""},
                                      {"role": "user", "content": f"""Examples:
```{train_df.iloc[0]['Plot']}``` - {train_df.iloc[0]['Genre']}
```{train_df.iloc[14]['Plot']}``` - {train_df.iloc[14]['Genre']}
```{train_df.iloc[17]['Plot']}``` - {train_df.iloc[17]['Genre']}
...
```{row['Plot']}```"""}]
        elif prompt_design == "fewSUA":
            dialog = [{"role": "system", "content": f"""Determine the genre of the movie based on the provided plot.
                                    For the plot provided, classify its genre as a single word (without other marks or words like 'genre:'), either "comedy", "action", "drama", or "horror"."""},
                                      {"role": "user", "content": train_df.iloc[0]['Plot']}, {"role": "assistant", "content": train_df.iloc[0]['Genre']},
                                      {"role": "user", "content": train_df.iloc[14]['Plot']}, {"role": "assistant", "content": train_df.iloc[14]['Genre']},
                                      {"role": "user", "content": train_df.iloc[17]['Plot']}, {"role": "assistant", "content": train_df.iloc[17]['Genre']},
                                      {"role": "user", "content": f"""```{row['Plot']}```"""}]

        response = generator.chat_completion(
            [dialog],  # Wrapping dialog in a list to match expected format
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        predicted_genre = response['generation']['content']
        print(index, ' ', predicted_genre)

        test_results.append({'Index': index, 'Plot': row['Plot'], 'Predicted_Genre': predicted_genre, 'Actual_Genre': row['Genre']})

    results_df = pd.DataFrame(test_results)

    valid_genres = ['comedy', 'action', 'drama', 'horror']
    results_df['Is_Structurally_Valid'] = results_df['Predicted_Genre'].apply(lambda x: x.strip().lower() in valid_genres)
    structural_accuracy = results_df['Is_Structurally_Valid'].mean()

    # Calculate F1 Score
    results_df['Predicted_Genre'] = results_df['Predicted_Genre'].apply(clean_response)
    results_df = results_df.dropna(subset=['Predicted_Genre', 'Actual_Genre'])
    f1 = f1_score(results_df['Actual_Genre'], results_df['Predicted_Genre'], average='micro')



    print(f"Experiment: {experiment_name}")
    print(f"The F1 Score for the predicted genre is: {f1}")
    print(f"The Structural Accuracy is: {structural_accuracy}")

    return f1, structural_accuracy

def main(ckpt_dir: str, tokenizer_path: str, genre_data_path: str, temperature: float = 0.0, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 8, max_gen_len: int = 256, output_file: str = "f1_scores.csv"):
    # Load the dataset
    DS = pd.read_pickle(genre_data_path)
    genre_data = pd.DataFrame(DS, columns=['Plot', 'Genre'])

    # Initialize LLAMA model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Split data into training and testing datasets
    test_samples_per_genre = 500
    train_samples_per_genre = 5
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for genre in genre_data['Genre'].unique():
        genre_df = genre_data[genre_data['Genre'] == genre]
        test_samples = genre_df.sample(test_samples_per_genre, random_state=1)
        train_samples = genre_df.drop(test_samples.index).sample(train_samples_per_genre, random_state=1)

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
