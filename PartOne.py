# Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
import spacy
from pathlib import Path
import os
import pandas as pd
import re
import pickle
from tqdm import tqdm
from collections import Counter
from math import log

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3e6

novels_directory = "/Users/marcolecu/Downloads/p1-texts/novels"

nltk.download("punkt_tab")
nltk.download("cmudict")


def fk_level(text: str, d: dict) -> float:
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)

        formula = 0.39 * (total words / total sentence) + 11.8 * (total syllables / total words) - 15.59
    """
    # Tokenise the given text into sentences
    sentences: list[str] = sent_tokenize(text)

    words_token = word_tokenize(text)
    words = []

    for word in words_token:
        if word.isalpha():
            words.append(word)

    if len(sentences) == 0 and len(words) == 0:
        return 0

    total_syllables = sum(count_syl(word, d) for word in words)

    total_words = len(words)
    total_sentences = len(sentences)

    fk_grade_level = (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )
    return fk_grade_level


def count_syl(word: str, d: dict):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()

    if word in d:
        translation = d[word][0]  # extract the first element for simple annotation
        syllables = 0

        for vowels_clusters in translation:
            if vowels_clusters[-1].isdigit():
                syllables += 1
        return syllables

    return len(re.findall(r"[aeiouy]+", word))


def read_novels(path: str = Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year. This function and `read_texts` are interchangeable.  `read_texts` is brought up
    here as it was also mentioned in the brief.
    """
    data = []

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            try:
                title, author, year_txt = filename[:-4].split("-")
                year = int(year_txt)
            except ValueError:
                continue

            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

            data.append({"text": text, "title": title, "author": author, "year": year})

    # Create a dataframe as instructed in 1a.i
    df = pd.DataFrame(data, columns=["text", "title", "author", "year"])
    # Sort values by year as required in 1a.ii
    df.sort_values(by="year", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse(
    df: pd.DataFrame,
    store_path: str = Path.cwd() / "pickles",
    out_name: str = "parsed.pickle",
) -> pd.DataFrame:
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes
    the resulting  DataFrame to a pickle file"""
    # Instantiate a nlp Language instance with disabled pipeline components for efficiency
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    # Increase max window in case limit is met
    nlp.max_length = 3e6

    store_path.mkdir(parents=True, exist_ok=True)

    # Store the parsed docs as new column
    parsed_docs = []

    # Here we print with tqdm to visually inspect list comprehension (loop) progress
    for text in tqdm(df["text"]):
        doc = nlp(text)

        parsed_docs.append(doc)

    # Add a new column called `parsed` to the Dataframe as per 1.e part i)
    df["parsed"] = parsed_docs

    # # Save DataFrame to a pickle file (according to 1.e part ii)
    output_file = store_path / out_name
    with open(output_file, "wb") as f:
        pickle.dump(df, f)

    # # Return DataFrame (according to 1.e part iii)
    return df


def nltk_ttr(text: str) -> float:
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text)

    words = []

    for word in tokens:
        if word.isalpha():
            words.append(word.lower())

    if len(words) == 0:
        return 0

    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    return ttr


def get_ttrs(df: pd.Dataframe):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def most_common_syntatic_objects(doc, n=10):
    """Extracts the most common syntactic objects (dependencies) in a parsed document. Returns a list of tuples."""
    syntatic_dependencies_count = Counter(token.dep_ for token in doc)
    return syntatic_dependencies_count.most_common(n)


def subjects_by_verb_count(doc, n=10, verb="to hear"):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects_of_verb = []

    for token in doc:
        if token.lemma_ == "hear" and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subjects_of_verb.append(child.text.lower())

    return Counter(subjects_of_verb).most_common(n)


def subjects_by_verb_pmi(doc, n=10):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    choose the number of times words occur overall, or specifically as syntactic subjects
    PMI(x, y) = log(P(x, y) / (P(x) * P(y)))
    P(X): frequency of the subject word in the entire text
    P(Y): frequency of any form/tense of a verb "to hear" in the entire text
    P(x,y): frequency of subject appearing as subject of "hear" """

    hear_subject_count = Counter()
    token_count = Counter()
    hear_verb_count = 0
    total_token_count = 0

    for token in doc:
        if token.is_alpha:
            word = token.text.lower()
            token_count[word] += 1
            total_token_count += 1

        if token.lemma_ == "hear" and token.pos_ == "VERB":
            hear_verb_count += 1
            for child in token.children:
                if child.dep_ == "nsubj":
                    hear_subject_count[child.text.lower()] += 1

    pmi_result = {}
    for subject, xy_count in hear_subject_count.items():
        p_xy = xy_count / hear_verb_count
        p_x = token_count[subject] / total_token_count if total_token_count > 0 else 0
        p_y = hear_verb_count / total_token_count if total_token_count > 0 else 0

        if p_x > 0 and p_y > 0:
            pmi = log(p_xy / (p_x * p_y))
            pmi_result[subject] = pmi

    pmi_list = list(pmi_result.items())
    pmi_list.sort(key=lambda pair: pair[1], reverse=True)
    most_common_syntatic_objects_by_pmi = pmi_list[:n]

    return most_common_syntatic_objects_by_pmi


if __name__ == "__main__":
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(
        path=novels_directory
    )  # this line will fail until you have completed the read_novels function above.
    print(df.head())

    nltk.download("cmudict")
    parse(df=df, out_name="part_one_parsed_texts.pickle")
    print(df.head())
    print(get_ttrs(df))

    print("---")
    print("Type-token Ratios for :\n")
    for title, ttr in get_ttrs(df).items():
        print(f"{title}: TTR = {ttr:.3f}\n")

    print("Flesch-Kincaid Reading Grade Level for :\n")
    for title, fk_grade_level in get_fks(df).items():
        print(f"{title}: {fk_grade_level:.3f}\n")

    df = pd.read_pickle(Path.cwd() / "pickles" / "part_one_parsed_texts.pickle")
    print(df.parsed)

    # ------------------------------------
    ## Part 1f.i
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]
        top_ten_syntactic_objects = most_common_syntatic_objects(doc)
        print(f"\n{title}:")
        print([dep for dep, count in top_ten_syntactic_objects])
    print("---")
    print("Most common subject of the verb:")

    ## Part 1f.ii
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]

        top_subjects_by_verb = subjects_by_verb_count(doc)

        print(f"\n{title}:")
        for subject, count in top_subjects_by_verb:
            print(f"  {subject}: {count}")

    ## Part 1f.ii
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]

        top_pmi = subjects_by_verb_pmi(doc, n=10)

        print(f"\n{title}:")
        for subject, pmi_result in top_pmi:
            print(f"  {subject}: PMI = {pmi_result:.2f}")
