import sys
import os
import math
import json
import random
import textwrap
import re
import collections
import pandas as pd
from collections import defaultdict, Counter

#step 1
class Trigram_LM:
    def __init__(self, sentences):
        #initialize counters
        self.unigram_counts=collections.Counter()
        self.bigram_counts=collections.Counter()
        self.trigram_counts=collections.Counter()
        self.total_tokens=0
        self.vocab_size=0

        for sentence in sentences:
            #tokenize
            tokens = sentence.split()
            #handling the math for the first 2 words in the sentence
            tokens= ['*start*','*start*']+tokens

            self.total_tokens+=len(tokens) #update total tokens count

            for i in range(len(tokens)):
                #count unigrams
                w_n=tokens[i]
                self.unigram_counts[w_n]+=1
                #count bigrams
                if i>=1:
                    w_prev=tokens[i-1]
                    self.bigram_counts[(w_prev,w_n)]+=1
                if i>=2:
                    w_prev2=tokens[i-2]
                    self.trigram_counts[(w_prev2,w_prev,w_n)]+=1

        #vocab size
        self.vocab_size+=len(self.unigram_counts)


    def calculate_prob_of_sentence(self, sentence):
        lambda3=0.8  #trigram
        lambda2=0.1999  #bigram
        lambda1=1-lambda3-lambda2   #unigrams
        tokens=sentence.split()
        tokens= ['*start*','*start*']+tokens
        total_log_probs=0

        for i in range(2,len(tokens)):
            w_n=tokens[i]
            w_prev1=tokens[i-1]
            w_prev2=tokens[i-2]

            #calculation probabilities with Laplace smoothing
            #unigram probs
            unigram_probs= (self.unigram_counts[w_n]+1)/(self.total_tokens+self.vocab_size)
            #bigram probs
            bigram_probs= (self.bigram_counts[(w_prev1,w_n)]+1)/(self.unigram_counts[w_prev1]+self.vocab_size)
            #trigram probs
            trigram_probs=(self.trigram_counts[(w_prev2,w_prev1,w_n)]+1)/(self.bigram_counts[(w_prev2,w_prev1)]+self.vocab_size)

            #sum log probabilities with linear interpolation
            total_log_probs+=math.log(lambda1*unigram_probs+lambda2*bigram_probs+lambda3*trigram_probs)

        return total_log_probs


    def generate_next_token(self, sentence):
        lambda3 = 0.8  # trigram
        lambda2 = 0.1999  # bigram
        lambda1 = 1 - lambda3 - lambda2  # unigrams
        tokens = sentence.split()
        tokens= ['*start*','*start*']+tokens
        # last two words in the sentence
        w_prev1 = tokens[-1]
        w_prev2 = tokens[-2]
        max_log_probs=-float('inf')
        best_word=None

        #calculating denominators
        uni_denom=self.total_tokens+self.vocab_size
        bi_denom=self.unigram_counts[w_prev1]+self.vocab_size
        tri_denom=self.bigram_counts[(w_prev2,w_prev1)]+self.vocab_size

        for word in self.unigram_counts:
            if word=='*start*' : continue

            w_n=word

            # unigram probs
            unigram_probs = (self.unigram_counts[w_n] + 1) / uni_denom
            # bigram probs
            bigram_probs = (self.bigram_counts[(w_prev1, w_n)] + 1) / bi_denom
            # trigram probs
            trigram_probs = (self.trigram_counts[(w_prev2, w_prev1, w_n)] + 1) / tri_denom

            word_log_probs=math.log(lambda1*unigram_probs+lambda2*bigram_probs+lambda3*trigram_probs)
            if(word_log_probs>max_log_probs):
                best_word=w_n
                max_log_probs=word_log_probs

        return (best_word,max_log_probs)

#step 2- Collacation
PUNCT_TOKENS = {
    ".", ",", "!", "?", ":", ";",
    "(", ")", "[", "]", "{", "}",
    "״", "\"", "'", "%",
    "<<", ">>", "<", ">", "–", "-",
}


def clean_token(tok: str) -> str:
    # remove leading/trailing punctuation characters
    return re.sub(r'^[<>\-–\(\)\"\'«»]+|[<>\-–\(\)\"\'«»]+$', '', tok)

def remove_punctuation_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    clean_tokens = []
    for tok in tokens:
        tok = clean_token(tok)
        if tok and tok not in PUNCT_TOKENS:
            clean_tokens.append(tok)
    return " ".join(clean_tokens)


def get_k_n_t_collocations(k: int,
                           n: int,
                           t: int,
                           corpus: pd.DataFrame,
                           metric_type: str) -> list:
    """
    return the top-k n-gram collocations that appear at least t times
    according to the selected metric ('frequency' or 'tfidf')
    corpus is a dataframe with at least:
      - 'protocol_name': document id (protocol)
      - 'sentence_text': tokenized sentence (tokens separated by space)
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if metric_type not in {"frequency", "tfidf"}:
        raise ValueError("metric_type must be 'frequency' or 'tfidf'")

    if "protocol_name" not in corpus.columns or "sentence_text" not in corpus.columns:
        raise ValueError("corpus must contain 'protocol_name' and 'sentence_text' columns")

    # unique documents (protocols)
    doc_ids = corpus["protocol_name"].unique().tolist()
    n_docs = len(doc_ids)

    # total_counts: how many times each n-gram appears in the whole corpus
    total_counts = Counter()
    # doc_freq: in how many different documents each n-gram appears
    doc_freq = Counter()

    # first pass: compute per-document counts and global stats
    for doc_id, group in corpus.groupby("protocol_name"):
        doc_counts = Counter()
        total_ngrams_in_doc = 0

        for sent in group["sentence_text"]:
            if not isinstance(sent, str):
                continue
            tokens = sent.split()
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                doc_counts[ngram] += 1
                total_ngrams_in_doc += 1

        # if document has no n-grams we just skip it
        if total_ngrams_in_doc == 0:
            continue

        # update global counters
        for ngram, c in doc_counts.items():
            total_counts[ngram] += c
            doc_freq[ngram] += 1

    # if metric is just frequency, we are basically done
    if metric_type == "frequency":
        # filter by threshold t
        candidates = [
            (ngram, count)
            for ngram, count in total_counts.items()
            if count >= t
        ]
        # sort by count desc, then lexicographically for stability
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top = [ng for ng, _ in candidates[:k]]
        return top

    # from here metric_type == "tfidf"
    if n_docs == 0:
        return []

    # compute idf for each n-gram
    idf_values = {}
    for ngram, df in doc_freq.items():
        # df is at least 1 so this is safe
        idf_values[ngram] = math.log(n_docs / df)

    # second pass: compute tf-idf for each n-gram per document and sum
    tfidf_sum = defaultdict(float)

    for doc_id, group in corpus.groupby("protocol_name"):
        doc_counts = Counter()
        total_ngrams_in_doc = 0

        for sent in group["sentence_text"]:
            if not isinstance(sent, str):
                continue
            tokens = sent.split()
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                doc_counts[ngram] += 1
                total_ngrams_in_doc += 1

        if total_ngrams_in_doc == 0:
            continue

        # add tf * idf for this document
        for ngram, c in doc_counts.items():
            if ngram not in idf_values:
                continue
            tf = c / total_ngrams_in_doc
            tfidf_sum[ngram] += tf * idf_values[ngram]

    # now compute average tf-idf over all documents
    scored = []
    for ngram, total_count in total_counts.items():
        if total_count < t:
            continue
        if ngram not in tfidf_sum:
            continue
        avg_tfidf = tfidf_sum[ngram] / n_docs
        scored.append((ngram, avg_tfidf))

    # sort by average tf-idf desc, then lexicographically
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = [ng for ng, _ in scored[:k]]
    return top


def build_full_and_no_punc_corpora(corpus_df: pd.DataFrame):
    """
    build two dataframes:
    - full corpus: original sentences
    - no punctuation corpus: same rows but without punctuation tokens
    """
    full_corpus = corpus_df.copy()

    no_punc_corpus = corpus_df.copy()
    no_punc_corpus["sentence_text"] = no_punc_corpus["sentence_text"].apply(
        remove_punctuation_tokens
    )

    return full_corpus, no_punc_corpus


def write_collocations_file(corpus_df: pd.DataFrame, output_path: str):
    """
    create the txt.collocations_knesset file in the exact required format
    """
    full_corpus, no_punc_corpus = build_full_and_no_punc_corpora(corpus_df)

    with open(output_path, "w", encoding="utf-8") as f:
        # two-gram collocations
        f.write("Two-gram collocations:\n")
        f.write("Frequency:\n")

        f.write("Full corpus:\n")
        two_full_freq = get_k_n_t_collocations(
            k=10, n=2, t=10, corpus=full_corpus, metric_type="frequency"
        )
        for coll in two_full_freq:
            f.write(coll + "\n")
        f.write("\n")

        f.write("No punctuation corpus:\n")
        two_nopunc_freq = get_k_n_t_collocations(
            k=10, n=2, t=10, corpus=no_punc_corpus, metric_type="frequency"
        )
        for coll in two_nopunc_freq:
            f.write(coll + "\n")
        f.write("\n")

        f.write("TF-IDF:\n")

        f.write("Full corpus:\n")
        two_full_tfidf = get_k_n_t_collocations(
            k=10, n=2, t=10, corpus=full_corpus, metric_type="tfidf"
        )
        for coll in two_full_tfidf:
            f.write(coll + "\n")
        f.write("\n")

        f.write("No punctuation corpus:\n")
        two_nopunc_tfidf = get_k_n_t_collocations(
            k=10, n=2, t=10, corpus=no_punc_corpus, metric_type="tfidf"
        )
        for coll in two_nopunc_tfidf:
            f.write(coll + "\n")
        f.write("\n")

        # four-gram collocations
        f.write("Four-gram collocations:\n")
        f.write("Frequency:\n")

        f.write("Full corpus:\n")
        four_full_freq = get_k_n_t_collocations(
            k=10, n=4, t=5, corpus=full_corpus, metric_type="frequency"
        )
        for coll in four_full_freq:
            f.write(coll + "\n")
        f.write("\n")

        f.write("No punctuation corpus:\n")
        four_nopunc_freq = get_k_n_t_collocations(
            k=10, n=4, t=5, corpus=no_punc_corpus, metric_type="frequency"
        )
        for coll in four_nopunc_freq:
            f.write(coll + "\n")
        f.write("\n")

        f.write("TF-IDF:\n")

        f.write("Full corpus:\n")
        four_full_tfidf = get_k_n_t_collocations(
            k=10, n=4, t=5, corpus=full_corpus, metric_type="tfidf"
        )
        for coll in four_full_tfidf:
            f.write(coll + "\n")
        f.write("\n")

        f.write("No punctuation corpus:\n")
        four_nopunc_tfidf = get_k_n_t_collocations(
            k=10, n=4, t=5, corpus=no_punc_corpus, metric_type="tfidf"
        )
        for coll in four_nopunc_tfidf:
            f.write(coll + "\n")
        f.write("\n")

#step 3.1
#input: (a list of sentences, a percent of tokens that we'll replace)
def mask_tokens_in_sentences(sentences,x):
    masked_sentences = []

    for sentence in sentences:
        tokens=sentence.split()
        n_tokens=len(tokens)
        mask_size=max(1,int(n_tokens*x)) #mask size

        #sample token indices randomly
        masked_indices=random.sample(range(n_tokens),mask_size)
        #masking the sentence
        for i in masked_indices:
            tokens[i]='[*]'

        #join back the sentence and add to the list
        masked_sentences.append(" ".join(tokens))

    return masked_sentences


#helper function for 3.3 - predicting masked token
def predict_sentence(model,masked_sent):
    tokens=masked_sent.split()
    predcted_tokens_list=[]
    for i in range(len(tokens)):
        if tokens[i]=='[*]':
            context=" ".join(tokens[:i])
            best_word, val =model.generate_next_token(context)
            tokens[i]=best_word
            predcted_tokens_list.append(best_word)

    full_predected_tokens=",".join(predcted_tokens_list)
    return " ".join(tokens), full_predected_tokens


#loading and testing
def load_corpus(file_path):
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # CHECK THIS KEY: If your HW1 used "sentence" or "content", change "text" below
                if 'sentence_text' in data:
                    sentences.append(data['sentence_text'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return []
    return sentences


# --- Main Execution Block ---
if __name__ == '__main__':
    #paths
    if len(sys.argv)<3:
        sys.exit(1)
    corpus_path = sys.argv[1]
    output_dir = sys.argv[2]

    #step 2 output- collacations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        corpus_df = pd.read_json(corpus_path, lines=True)
        collocations_out = os.path.join(output_dir, "knesset_collocations.txt")
        write_collocations_file(corpus_df, collocations_out)
    except Exception as e:
        pass

    all_sentences = load_corpus(corpus_path)

    #build model with all sentences
    model_full = Trigram_LM(all_sentences)

    # train no punctuation model
    puncs = set([',', '.', ':', '?', '!', '-', '(', ')', '"', "'", ';', '–'])

    no_punc_sentences = []
    for s in all_sentences:
        tokens = s.split()
        clean_tokens = [t for t in tokens if t not in puncs]
        no_punc_sentences.append(" ".join(clean_tokens))
    model_no_punc = Trigram_LM(no_punc_sentences)

    #step 3.2
    #filter for sentences with at least 5 words
    long_sentences=[s for s in all_sentences if len(s.split())>=5]
    #randomly select 10 long sentences
    if len(long_sentences) < 10:
        original_sampled = long_sentences
    else:
        original_sampled = random.sample(long_sentences, 10)

    masked_sampled=mask_tokens_in_sentences(original_sampled,0.1)

    #save the files
    with open(os.path.join(output_dir, 'original_sampled_sents.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(original_sampled))

    with open(os.path.join(output_dir, 'masked_sampled_sents.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(masked_sampled))

    #step 3.3
    results_output= []
    for i in range(len(original_sampled)):
        original=original_sampled[i]
        masked=masked_sampled[i]
        #predict with full model
        full_sentence, full_tokens= predict_sentence(model_full,masked)
        #probability of full sentence in full corpus -num1
        prob_full_full=model_full.calculate_prob_of_sentence(full_sentence)

        # predict with model no-puntuation
        no_punc_sentence, no_punc_tokens= predict_sentence(model_no_punc,masked)
        # probability of no_punc in no_punc corpus- num4
        prob_no_punc_no_punc=model_no_punc.calculate_prob_of_sentence(no_punc_sentence)

        #probability of full sentence in no_punc corpus -num2
        prob_full_no_punc=model_no_punc.calculate_prob_of_sentence(full_sentence)

        #probability of no_punc in full corpus- num3
        prob_no_punc_full=model_full.calculate_prob_of_sentence(no_punc_sentence)

        block= textwrap.dedent(f"""
        original_sentence: {original}
        masked_sentence: {masked}
        full_sentence: {full_sentence}
        full_tokens: {full_tokens}
        no_punc_sentence: {no_punc_sentence}
        no_punc_tokens: {no_punc_tokens}
        probability of full sentence in full corpus: {prob_full_full:.2f}
        probability of full sentence in no_punc corpus: {prob_full_no_punc:.2f}
        probability of no_punc sentence in full corpus: {prob_no_punc_full:.2f}
        probability of no_punc sentence in no_punc corpus: {prob_no_punc_no_punc:.2f}
        """).strip()
        results_output.append(block)

    results_path = os.path.join(output_dir, 'sampled_sents_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_output))
