import collections
import math
import json

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
        print("training complete.")
        print(f"vocab size: {self.vocab_size},total tokens: {self.total_tokens}")

    def calculate_prob_of_sentence(self, sentence):
        lambda3=0.8  #trigram
        lambda2=0.19  #bigram
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
        lambda2 = 0.19  # bigram
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

#step 2


# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Load the data
    corpus_path = r'C:\Users\Samar\PycharmProjects\PythonProject\knesset_corpus.jsonl'
    print(f"Loading corpus from {corpus_path}...")

    all_sentences = load_corpus(corpus_path)

    if not all_sentences:
        print("Error: No sentences found. Please check the JSON key name inside load_corpus().")
    else:
        # 2. Train the Model (Using first 1000 for testing purposes)
        # CHANGE TO: model = Trigram_LM(all_sentences) for the final run
        print("Training model (Testing on all sentences)...")
        model = Trigram_LM(all_sentences)

        # 3. Test Prediction
        test_context = "חברי הכנסת"
        print(f"\nPredicting next token after: '{test_context}'")
        next_word, prob = model.generate_next_token(test_context)
        print(f"Predicted: {next_word} (Log Prob: {prob:.2f})")

        # 4. Test Sentence Probability
        test_sent = "חברי הכנסת הצביעו בעד החוק"
        score = model.calculate_prob_of_sentence(test_sent)
        print(f"Probability of '{test_sent}': {score:.2f}")

        # Real sentence (Common in Knesset)
        good_sent = "אני מבקש להודות ליושב ראש"
        score1 = model.calculate_prob_of_sentence(good_sent)

        # Nonsense sentence (Random words)
        bad_sent = "שולחן אכל אתמול את החוק"
        score2 = model.calculate_prob_of_sentence(bad_sent)

        print(f"Good Sentence Score: {score1:.2f}")  # Expect roughly -20 to -40
        print(f"Bad Sentence Score: {score2:.2f}")  # Expect roughly -50 to -70 (Lower is worse)

        # "Head of..." -> Should predict "Government" (הממשלה) or "Committee" (הוועדה)
        context = "ראש"
        pred, _ = model.generate_next_token(context)
        print(f"After 'ראש', model predicts: {pred}")

        # "Supercalifragilistic" is not in the Knesset corpus
        weird_sent = "החוק הוא סופרקאליפרג'יליסטיק"
        try:
            s = model.calculate_prob_of_sentence(weird_sent)
            print(f"Unknown word handled safely. Score: {s:.2f}")
        except:
            print("CRASHED on unknown word!")
