# Evaluating Role-Based Prompt Architectures in In-Context Learning
This repository contains the code and data for the paper "Evaluating Role-Based Prompt Architectures in In-Context Learning" 
## Key Contributions
- Systematic evaluation of role designs in zero-shot and few-shot learning scenarios.
- Assessment of model performance across tasks like sentiment analysis, text classification, and question answering.
- Introduction of new metrics for measuring the effectiveness of role designs in prompts.
- Insights into optimizing prompt design strategies for better performance in natural language processing tasks.

## Installation

### Llama 2

Llama 2 is accessible to individuals, creators, and researchers, ranging from 7B to 70B parameters.

1. To download the model weights and tokenizer, visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept their License. Once your request is approved, you will receive a signed URL via email.
2. Run the `download.sh` script, passing the URL provided when prompted to start the download.

For more details, see [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/).

3. In a conda environment with PyTorch / CUDA available, clone and download this repository:
    ```bash
    git clone https://github.com/your-repository.git
    cd your-repository
    pip install -e .
    ```

### GPT Models

1. Open an OpenAI account and obtain your private API key.
2. Insert your API key in the designated place in the code.

## Usage

### Running Llama 2 Models

Once the model(s) you want have been downloaded, you can run the model locally using the following command:

```bash
torchrun --nproc_per_node 1 genre_llama.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 4096 --max_batch_size 6 --genre_data_path balanced_movies_genre.pkl --output_file f1_scores_genre_llama-2-7b-chat.csv
```
# Note:

- Replace llama-2-7b-chat/ with the path to your checkpoint directory and tokenizer.model with the path to your tokenizer model.
- The –nproc_per_node should be set to the appropriate value for the model, with 1 for llama-2-7b-chat and 2 for llama-2-13b-chat.
- All balanced datasets are added in the repository in .pkl format; change it for each dataset.
- For QA datasets, change --genre_data_path to --answer_data_path.
- Modify the Python code to match the specific dataset you are working with.

### Running GPT Models
1. Open the appropriate notebook (IMDB.ipynb, Movies_Genre.ipynb, commonsense_qa.ipynb).
2. Ensure your OpenAI API key is inserted correctly in the code.
3. Run the cells in order to reproduce the results. Note that you should run the cell for either GPT-4o or GPT-3.5 but not both at the same time.
Example Notebooks:

IMDB.ipynb
Movies_Genre.ipynb
commonsense_qa.ipynb

## Datasets
This study utilizes samples from a diverse range of datasets to evaluate the performance of different prompt designs across various natural language tasks:

- commonsense_qa: A dataset designed for common-sense question answering.
- ai2_arc: A benchmark dataset used for evaluating question-answering capabilities.
- wiki_movie_plots: A dataset containing movie plots used for genre classification.
- IMDB_reviews: A sentiment analysis dataset comprising movie reviews with corresponding sentiment labels.
- MATH Dataset: A dataset to test reasoning skills with mathematical word problem questions.


## Original Llama
The repo for the original llama2  is in the [`llama2`](https://github.com/meta-llama/llama).

---

© ... All rights reserved.
