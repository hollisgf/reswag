"""
A script for generating random data for word segmentation tasks / statistical
learning.
"""
import argparse, sys, string, random

class WordGenerator:
    """Implements a class for generating unique words, according to some
       specified parameters.
    """
    def generate_word(self):
        """generates a single new word"""
        pass

    def generate_words(self, n):
        return [ self.generate_word() for i in xrange(n) ]

class PseudoVietnameseWordGenerator(WordGenerator):
    """Generates words for a pseudo-vietnamese language. For details, see:
    
       Baayen, R. H., Shaoul, C., Willits, J., & Ramscar, M. (2016). Comprehension without segmentation: A proof of concept with naive discriminative learning. Language, Cognition and Neuroscience, 31(1), 106-128.
Chicago	
    """
    def __init__(self):
        self.prev_words = set() # words we have previously generated
    
    def generate_syllable(self):
        """Syllables in this artificial language have a CCVC structure.
        """
        pos1 = "ptkbdg"
        pos2 = "fsxvzG"
        pos3 = "aeiou"
        pos4 = "ptkbdgfsxvzGrlh"

        syll = random.choice(pos1) + random.choice(pos2) + random.choice(pos3) + random.choice(pos4)
        return syll
    
    def generate_word(self):
        """Words are 1 or 2 syllables, with 10/90 ratio.
        """
        # 10% chance of monosyllabic word; 90% chance bisyllabic
        word = self.generate_syllable()
        if random.random() <= 0.90:
            word += self.generate_syllable()

        if word not in self.prev_words:
            self.prev_words.add(word)
            return word
        # try to generate another
        return self.generate_word()
    
class UniqueLetterWordGenerator(WordGenerator):
    """Generates words of length n. All characters in a word are unique to
       that word.
    """
    def __init__(self, length_distribution):
        """generates words according to a length distribution (dict of lengths
           to weights). Can be a single integer, in which case all words will
           be of that length.
        """
        if type(length_distribution) == int:
            length_distribution = { length_distribution : 1.0 }
        self.dist = self.normalize_distribution(length_distribution)
        
        # prepare our sequence of possible characters
        self.char_sequence = string.letters + string.digits + string.punctuation.replace("#", "")
        self.seq_start = 0

    def normalize_distribution(self, dist):
        """normalize a distribution so all vals sum to 1.0
        """
        valsum = 0
        for val in dist.itervalues():
            valsum += val

        newdist = { }
        for key, val in dist.iteritems():
            newdist[key] = val / valsum

        return newdist
        
    def choose_length(self):
        """Choose a length for the word to be, according to our distribution
        """
        # choose a probability
        prob      = random.random()
        weightsum = 0
        for key,val in self.dist.iteritems():
            weightsum += val
            if weightsum >= prob:
                return key
        return None
        
    def generate_word(self):
        length = self.choose_length()
        slice  = self.char_sequence[self.seq_start:(self.seq_start+length)]
        self.seq_start += length
        return "".join(slice)

class TokenGenerator:
    """Genereates tokens from a dictionary of types, over a distribution of
       frequencies.
    """
    def __init__(self):
        self.freqs      = { }
        self.normalized = False
    
    def register(self, word, weight):
        """Register new word type into the generator with the specified weight.
           Weight determines its chances of being generated, compared to other
           words.
        """
        if self.normalized:
            raise Exception("Cannot register new types after normalization.")
        self.freqs[word] = weight
    
    def generate(self):
        # normalize our weights
        if self.normalized == False:
            weightsum = 0.0
            for k,v in self.freqs.iteritems():
                weightsum += v
            for k in self.freqs.keys():
                self.freqs[k] /= weightsum
            self.normalized = True

        rnd = random.random()
        weightsum = 0.0
        for k,v in self.freqs.iteritems():
            weightsum += v
            if weightsum >= rnd:
                return k
        return k


    
def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Interface for generating data for statistical learning simulation.')
    parser.add_argument("triallength", type=int, default=4, help="How many words per trial?")
    parser.add_argument("trials", type=int, default=1000, help="How many trials to generate?")
    parser.add_argument("--wordlength", type=int, default=3, help="how many 'units' long are words?")
    parser.add_argument("--numwords", type=int, default=4, help="number of words to generate and randomly sample from.")
    parser.add_argument("--language", type=str, default="basic", help="Which language model are we using?")
    parser.add_argument("--distribution", type=str, default="uniform", help="which sampling distribution are we using for tokens?")
    
    args = parser.parse_args()

    word_generator  = {
        "basic"      : UniqueLetterWordGenerator(args.wordlength),
        "viet"       : PseudoVietnameseWordGenerator(),
        "vietnamese" : PseudoVietnameseWordGenerator(),
        }[args.language]

    distro = {
        "uniform" : lambda : 1,
        "lognorm" : lambda : random.lognormvariate(4,2),
        }[args.distribution]
    
    # generate our words
    words = word_generator.generate_words(args.numwords)

    # register them in our token generator
    token_generator = TokenGenerator()
    for word in words:
        token_generator.register(word, distro())
    
    # print all words in the language as a comment header
    print "###", " ".join(words)

    # generate our learning
    for i in xrange(args.trials):
        trial = [ token_generator.generate() for j in xrange(args.triallength) ]
        print "".join(trial)

    

if __name__ == "__main__":
    sys.exit(main())
    
