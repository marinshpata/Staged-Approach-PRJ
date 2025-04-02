class SimpleWordTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        self.idx2word = {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
        self.vocab_size = 4

    def build_vocab(self, lines):
        for line in lines:
            words = line.strip().split()
            for w in words:
                w = w.lower()
                if w not in self.word2idx:
                    idx = self.vocab_size
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    self.vocab_size += 1

    def encode(self, text):
        words = text.strip().split()
        ids = [1]  #<sos> - start
        for w in words:
            w = w.lower()
            ids.append(self.word2idx.get(w, 3))  #<unk> - unknown, if not found
        ids.append(2)  #<eos> - end
        return ids

    def decode(self, idx_list):
        words = []
        for i in idx_list:
            if i in [0,1,2]:
                continue
            words.append(self.idx2word.get(i, "<unk>"))
        return " ".join(words)

# resources such as this https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
# and https://huggingface.co/learn/nlp-course/en/chapter2/4#word-based, https://huggingface.co/learn/nlp-course/en/chapter6/8 have been useful in the learning proccess for the purposes of this part of the project