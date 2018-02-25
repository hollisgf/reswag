"""
A variety of helper tools for using the R-W model and vector approximation
as a naive discriminative learner (see Baayen et al, 2011).

Contains tools for:
  * generating ngrams out of ordered lists.
  * Reading text from files in various chunk sizes.
  * Converting text to an iterable stream of (cue, outcome) learning events.

Baayen, R. H., Milin, P., Durdevic, D. F., Hendrix, P., & Marelli, M. (2011). An amorphous model for morphological processing in visual comprehension based on naive discriminative learning. Psychological review, 118(3), 438.
"""



################################################################################
# helper functions
################################################################################
def generate_ngrams(sequence, n=3):
    """takes a sequence and returns ngrams of length n"""
    ngrams = [ ]
    for i in xrange(len(sequence)-n+1):
        ngrams.append(sequence[i:(i+n)])
    return ngrams

class FileReader:
    """iterates over text units in a corpus. Can iterate over lines or 
       documents.
    """
    def __init__(self, fname, readmode="line", space_char="#", newline_char="|", head_buffer="", tail_buffer="", comment="###", EOD="---END.OF.DOCUMENT---"):
        self.fname = fname
        self.space_char    = space_char
        self.newline_char  = newline_char
        self.comment       = comment
        self.readmode      = readmode
        self.head_buffer   = head_buffer
        self.tail_buffer   = tail_buffer
        self.EOD           = EOD

        if self.readmode not in ("line", "doc", "document"):
            raise Exception("reader mode must be line or document.")
        if self.readmode != "line":
            self.readmode = "doc"

    def __iter__(self):
        fl   = open(self.fname, 'r')
        doc  = ""
        line = fl.readline()
        while line != "":
            if line.startswith(self.comment):
                pass
            elif line.startswith(self.EOD):
                if self.readmode == "doc":
                    yield self.head_buffer + doc + self.tail_buffer
                    doc = ""
            else:
                # do some preprocessing
                if line[-1] == "\n":
                    line = line[:-1]
                line = line.replace(" ", self.space_char)
                line = line + self.newline_char

                # if we have a line with content, yield it
                if len(line) > 0:
                    if self.readmode == "line":
                        yield self.head_buffer + line + self.tail_buffer
                    else:
                        doc += line
            line = fl.readline()
            
        if self.readmode == "doc" and len(doc) > 0:
            yield self.head_buffer + doc + self.tail_buffer
        fl.close()

class LineSegmenter:
    """Attempts to segment a string based on surprisal of successive cues,
       given an NDL model.
    """
    def __init__(self, model, window, nsize, segmentation_threshold=0.92):
        self.model                  = model
        self.segmentation_threshold = segmentation_threshold
        self.window                 = window   # how big is our sliding window?
        self.nsize                  = nsize    # what is our ngram size?

    def segment(self, text):
        # turn the text into cues
        ngrams   = generate_ngrams(text, self.nsize)
        segments = [ ]

        # we cannot segment the first nsize + window - 1 letters, because
        # there are no outcomes in that range yet.
        segment_start = 0
        segment_end   = self.nsize + self.window - 1
        
        for ngram_i in xrange(len(ngrams) - self.window):
            cues    = ngrams[ngram_i:(ngram_i+self.window)]
            outcome = ngrams[ngram_i+self.window]

            surprise= 1.0 - self.model.activation(cues, outcome)
            if surprise >= self.segmentation_threshold:
                segment     = text[segment_start:segment_end]
                segments.append(segment)
                segment_start = segment_end
                
            segment_end += 1

        segments.append(text[segment_start:])
        return segments
        
        
        
################################################################################
# experience channels
################################################################################
class ExperienceChannel:
    """Provides a transformation of a corpus into an iterable sequence of cues 
       and outcomes to learn from. Different channel types may use different 
       cues (letters, ngrams, words) and different outcomes (ngrams, lexomes).
    """
    def __iter__(self):
        pass

class NgramToLexomeChannel(ExperienceChannel):
    """Uses cues within a sliding window (letters or ngrams) to predict lexomes
       (word referents) that are fully or partially present in that window as 
       well.
    """
    ############
    # FINISH ME
    ############
    pass
    
class ForwardPredictionChannel(ExperienceChannel):
    """Cues and outcomes are of the same type (word, letter, ngram). Uses all
       of the cues available in a sliding window to predict the next outcome
       beyond that window.
    """
    def __init__(self, corpora, unit="ngram", readmode="line", window=2, space_char="#", newline_char="|", buffer_ends=True, nsize=None):
        self.corpora        = corpora
        if type(self.corpora) == str:
            self.corpora = [ self.corpora ]
        self.unit           = unit
        self.window         = window
        self.buffer_ends    = buffer_ends
        self.space_char     = space_char
        self.newline_char   = newline_char
        self.nsize          = nsize
        self.readmode       = readmode

        if unit not in ("word", "letter", "ngram"):
            raise Exception("unit must be word, letter, or ngram.")

        if unit == "ngram" and (nsize == None or nsize < 1):
            raise Exception("an nsize > 0 must be specified when the cue unit is an ngram.")

    def __iter__(self):
        # figure out how many spaces we need to use to buffer text
        # in order to be able to be exposed to all ngrams in the text
        buf      = ""
        buf_size = self.window + self.nsize
        if len(self.space_char) > 0:
            buf  = self.space_char * buf_size
        
        for corpus in self.corpora:
            for line in FileReader(corpus, readmode=self.readmode, space_char=self.space_char, newline_char=self.newline_char, head_buffer=buf, tail_buffer=buf):
                # turn the sequence into words
                if self.unit == 'word':
                    line = line.split()
                # otherwise, we need to do some processing
                else:        
                    # convert the sequence into ngrams
                    if self.unit == 'ngram':
                        line = generate_ngrams(line, n=self.nsize)
                    else:
                        line = list(line)
        
                # now iterate over the sequence and return our units
                if self.window >= len(line):
                    continue
                for i in xrange(len(line) - self.window):
                    cues    = line[i:(i+self.window)]
                    outcome = [line[i+self.window]]
                    yield (cues, outcome)
                    
                    
class NgramToWordChannel(ExperienceChannel):
    """each learning event encompasses a single word. Cues are letters or
       ngrams spanning that word, and the word is the outcome. Ignore any
       words in the channel with occurrences < mincount.
    """
    def __init__(self, corpora, maxgrain, mincount=0, flanking_spaces=True, space_char="#"):
        self.corpora  = corpora
        self.maxgrain = maxgrain
        self.flanking_spaces = flanking_spaces
        self.space_char = space_char
        self.cue_map  = { }
        self.valid_outcome = set()
        self.check_outcome_valid = False

        # find words to ingore
        if mincount > 0:
            self.__build_outcome_list(mincount)

    def __build_outcome_list(self, mincount=0):
        """only use outcomes that occur >= mincount times
        """
        counts = { }
        for cues, outcomes in self:
            for outcome in outcomes:
                try:
                    counts[outcome] += 1
                except:
                    counts[outcome]  = 1

        for outcome, count in counts.iteritems():
            if count >= mincount:
                self.valid_outcome.add(outcome)
                
        self.check_outcome_valid = True
        
    def __iter__(self):
        for corpus in self.corpora:
            for line in FileReader(corpus, space_char=" ", newline_char=""):
                for word in line.split():
                    outcome = word.upper()

                    if self.check_outcome_valid and outcome not in self.valid_outcome:
                        continue
                    
                    if self.flanking_spaces:
                        word = self.space_char + word + self.space_char
                    try:
                        cues = self.cue_map[outcome]
                    except:
                        cues = [ ]
                        for i in xrange(self.maxgrain):
                            cues.extend(generate_ngrams(word, i+1))
                        self.cue_map[outcome] = cues
                        # ignore single spaces as cues
                        if self.space_char in cues:
                            cues.remove(self.space_char)
                    yield (cues, (outcome,))
