import ast
import sys
import numpy as np
import pandas as pd
from datasets import Dataset
from functools import partial
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from emoji import demojize
from nltk.tokenize import TweetTokenizer

# load NLTK Tweet tokenizer
tokenizer = TweetTokenizer()

def preprocess_function(data, preprocessing_config):
    """
        Applies a sequence of text preprocessing steps to a DataFrame column (`data["text"]`)
        based on the options provided in `preprocessing_config`.

        Parameters:
            data (pd.DataFrame): A DataFrame containing a 'text' column and optionally a 'urls' column.
            preprocessing_config (dict): A dictionary specifying which preprocessing steps to apply.
                Keys:
                    - 'lowercase': bool - Convert text to lowercase.
                    - 'normalize': bool - Apply tweet normalization.
                    - 'emojis': str or False - Replace emojis with a string (e.g., "[EMOJI]"), with emoji shortcodes if "demojize", or skip if False.
                    - 'user_handles': str or False - Replace user handles (e.g., "@user") or skip if False.
                    - 'urls': str, 'original_urls', or False - Replace URLs with a string, or original URLs if specified.

        Returns:
            pd.DataFrame: The input DataFrame with the preprocessed 'text' column.
    """
    if preprocessing_config['lowercase']:
        data["text"] = data["text"].str.lower()

    if preprocessing_config['normalize']:
        data["text"] = data["text"].apply(normalize_tweet)

    if preprocessing_config['emojis'] != False:
        data["text"] = data["text"].apply(partial(replace_emojis, replace=preprocessing_config['emojis']))

    if preprocessing_config['user_handles'] != False:
        data["text"] = data["text"].apply(
            partial(replace_user_handles, replace=preprocessing_config['user_handles']))

    if preprocessing_config['urls'] == 'original_urls':
        data["text"] = data[["text", 'urls']].apply(lambda row: replace_urls(*row), axis=1)

    elif preprocessing_config['urls'] != False:
        data["text"] = data["text"].apply(partial(replace_urls, replace=preprocessing_config['urls']))

    return data


def normalize_tweet(tweet):
    """
        Applies normalization to an input string "tweet".

        Parameters:
            tweet (str): A tweet text.
        Returns:
            str: The normalized Tweet text.
    """
    normTweet = (
        tweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
    )

    return normTweet


def replace_user_handles(tweet, replace='@USER'):
    """
        Replace user mentions (e.g. "@DTrump") in input string "tweet".

        Parameters:
            tweet (str): A tweet text.
            replace (str) - Replace user handles with "replace".
        Returns:
            str: The modified Tweet text.
    """
    tokens = tokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if token.startswith("@"):
            new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def replace_urls(tweet, replace='HTTPURL'):
    """
        Replace URLS in input string "tweet" with "replace".

        Parameters:
            tweet (str): A tweet text.
            replace (str) - Replace URLs with "replace"
        Returns:
            str: The modified Tweet text.
    """
    tokens = tokenizer.tokenize(tweet)

    if type(replace) == str:
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                new_tokens.append(replace)
            else:
                new_tokens.append(token)

    elif type(replace) == list:
        n_replaced_tokens = 0
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                if n_replaced_tokens < len(replace):
                    new_tokens.append(replace[n_replaced_tokens])
                    n_replaced_tokens = n_replaced_tokens + 1
                else:
                    new_tokens.append('')
            else:
                new_tokens.append(token)

    return " ".join(new_tokens)


def replace_emojis(tweet, replace='demojize'):
    """
        Replace emojis in input string "tweet".

        Parameters:
            tweet (str): A tweet text.
            replace (str) - Replace emojis with "replace". If replace == "demojize", replace emojis with emoji shortcodes
        Returns:
            str: The modified Tweet text.
    """
    tokens = tokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if len(token) == 1:
            if replace == 'demojize':
                new_tokens.append(demojize(token))
            else:
                new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def sigmoid(z):
    """
        Apply sigmoid transformation to logits "z".

        Parameters:
            z (np.array[floats]): logits.
        Returns:
            np.array[floats]: The sigmoid transfored logits.
    """
    return 1 / (1 + np.exp(-z))


print('load data')
tweet_data_path = sys.argv[1]
tweet_data = pd.read_csv(tweet_data_path, sep='\t')

print('preprocess tweets')
# store text in a new columnn "original_text"
tweet_data['original_text'] = tweet_data['text']

# if "urls" column available try to read them as a list
has_original_urls = "urls" in tweet_data.columns
if has_original_urls:
    try:
        tweet_data['urls'] = tweet_data['urls'].apply(lambda urls: ast.literal_eval(urls))
    except:
        has_original_urls = False

# define preprocessing config
preprocessing_config = {'lowercase': True,
                        'normalize': True,
                        'urls': 'original_urls' if has_original_urls else False,
                        'user_handles': '@USER',
                        'emojis': 'demojize'}

# preprocess tweets
tweet_data = preprocess_function(tweet_data, preprocessing_config)

print('load model and tokenizer')
tokenizer_config = {'pretrained_model_name_or_path': "sschellhammer/SciTweets_SciBert", 'max_len': 128}
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

model_config = {'pretrained_model_name_or_path': "sschellhammer/SciTweets_SciBert"}
model = AutoModelForSequenceClassification.from_pretrained(**model_config)

print("tokenize data")
def tokenize_transform(examples):
    return tokenizer(examples["text"], max_length=128, truncation=True, padding='max_length', return_tensors="pt")

dataset = Dataset.from_pandas(tweet_data[['text']])
dataset.set_transform(tokenize_transform, output_all_columns=True)

# define training arguments
training_args = TrainingArguments(
    output_dir=".",  # output directory
    per_device_eval_batch_size=128,
    remove_unused_columns=False
)

# define Huggingface Trainer
trainer = Trainer(
    model=model,
    args=training_args
)

print('classify tweets')
pred_output = trainer.predict(dataset)

# calculate Sigmoid transformed scores
tweet_data['cat1_score'] = sigmoid(pred_output.predictions[:, 0])
tweet_data['cat2_score'] = sigmoid(pred_output.predictions[:, 1])
tweet_data['cat3_score'] = sigmoid(pred_output.predictions[:, 2])

# restore the original text and remove the modified text
tweet_data["text"] = tweet_data["original_text"]
del tweet_data["original_text"]

print("save classified tweets")
tweet_data.to_csv(tweet_data_path.replace(".tsv", "_pred.tsv"), sep="\t", index=False)
