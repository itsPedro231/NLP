import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


def filter_df(df, keywords, remove=False, column_to_be_analyzed=""):
    """Filter a DataFrame based on keywords in the 'Consumer complaint narrative'
    column.

        Args:
            df (pandas.DataFrame): The dataframe to filter.
            keywords (list of str): List of keywords to include/exclude.
            remove (bool, optional): If True, exclude rows containing any keyword.
    Defaults to False.

        Returns:
            pandas.DataFrame: Filtered dataframe containing only matching rows.
    """

    pattern = r"\b(?:" + "|".join(keywords) + r")\b"

    if remove:
        # If remove is True, exclude rows that contain any of the keywords
        filtered_df = df[
            ~df[column_to_be_analyzed].str.contains(pattern, case=False, na=False)
        ].copy()
    else:
        # Otherwise, include only rows that contain any of the keywords
        filtered_df = df[
            df[column_to_be_analyzed].str.contains(pattern, case=False, na=False)
        ].copy()

    return filtered_df


def count_words(filePath, topCount, keywords, filter=False, column_to_be_analyzed=""):
    """Count and return the most common ngrams in a text column.

        Args:
            filePath (str): Path to CSV file containing text data.
            topCount (int): Number of top ngrams to return.
            keywords (list of str, optional): Keywords for filtering. If provided, filter
    function will be applied. Defaults to None.
            filter (bool, optional): Whether to apply keyword-based filtering before
    counting. Defaults to False.

        Returns:
            list: List of tuples containing (ngram, count), sorted by count in descending
    order.
    """
    df = pd.read_csv(filePath)

    if filter:
        newDf = filter_df(df, keywords=keywords)
    else:
        newDf = df.copy()

    vectorizer = CountVectorizer(
        ngram_range=(2, 3), stop_words=stopwords.words("english")
    )
    ngrams = vectorizer.fit_transform(newDf[column_to_be_analyzed])

    sum_ngrams = ngrams.sum(axis=0)
    ngram_counts = [
        (word, sum_ngrams[0, idx]) for word, idx in vectorizer.vocabulary_.items()
    ]
    ngram_counts = sorted(ngram_counts, key=lambda x: x[1], reverse=True)

    top_ngrams = ngram_counts[:topCount]

    newDf.to_csv("./test.csv")

    return top_ngrams


def analyze_emotion(
    filePath="", keywords=[], filter=False, remove=False, column_name="", outputPath=""
):
    """Analyze sentiment and emotion in consumer complaints.

        Args:
            filePath (str, optional): Path to CSV file containing text data. Defaults to
    ''.
            keywords (list of str, optional): Keywords for filtering. Defaults to empty
    list.
            filter (bool, optional): Whether to apply keyword-based filtering before
    analysis. Defaults to False.
            remove (bool, optional): Whether to remove matching rows after filtering.
    Defaults to False.
            outputPath (str, optional): Path to save output file. Defaults to ''.

        Returns:
            pandas.DataFrame: DataFrame containing original text and sentiment scores.
    """
    df = pd.read_csv(filePath)

    if filter:
        filtered_df = filter_df(df, keywords=keywords, remove=remove)

        filtered_df[["true_score", "abs_score", "tuned_score"]] = (
            filtered_df[column_name].apply(emotions).apply(pd.Series)
        )

    else:
        filtered_df = df.copy()
        filtered_df[["true_score", "abs_score", "tuned_score"]] = (
            df[column_name].apply(emotions).apply(pd.Series)
        )

    filtered_df.to_csv(outputPath)

    return filtered_df


def emotions(text):
    """Process text for sentiment analysis.

    Args:
        text (str): Text to analyze.

    Returns:
        tuple: Contains:
            - Float: Compound score from SentimentIntensityAnalyzer
            - bool: Absolute positive sentiment score (1 if positive, 0 otherwise)
            - bool: Tuned positive sentiment score (1 if positive > 0.3, 0 otherwise)
    """
    tokens = word_tokenize(text.lower())

    filtered_tokens = [
        token for token in tokens if token not in stopwords.words("english")
    ]

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = " ".join(lemmatized_tokens)

    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(processed_text)

    sentiment_abs = 1 if scores["pos"] > 0 else 0
    sentiment_tuned = 1 if scores["pos"] > 0.3 else 0
    return scores["pos"], sentiment_abs, sentiment_tuned
