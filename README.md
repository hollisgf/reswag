# reswag
This package provides two implementations of the Rescorla-Wagner model. The
first is its classic form, described in Rescorla & Wagner (1972). The second
its vector approximation, described in Hollis (under review). The vector
approximation model uses a vector-based method that eliminates the need to
update for absent outcomes to reduce the computational cost of the model.
However, it does sacrifice some accuracy in learned association strengths as a
consequence.

The purpose of this package is primarily to provide support for training the
Rescorla-Wagner model in many-outcome situations, such as language learning.
Such applications are computationally costly for the Rescorla-Wagner model,
because the number of computations per update increases linearly with the
number of possible outcomes for any particular learning event.

# Examples
Various scripts have been created to demonstrate applications of the R-W model
to many-outcome situations (primarily, examples of language learning taken
from research performed by Harald Baayen and colleagues).

## Example 1 - segmentation without comprehension
The Rescorla-Wagner model can explain infant attention in statistical learning
paradigms (see Baayen, Shaoul, Willits, & Ramscar, 2016). Changes in infant
attention correspond to high-surprise phoneme sequences in a speech stream.
Simulated training data have been provided in data/statistical_learning[1-3].txt
and data/pseudoviet1.txt . These files each contain sequences of words. Train a
R-W model to condition letters on the next letter in the stream. Sequences of
high surprise (where the cues do not activate the outcome) correspond to word
boundaries. To train a model, run:

python scripts/train_segmentation_ndl.py data/statistical_learning1.txt --unit=letter --window=2 --readmode=doc --space_char="#" --newline_char="" --out=MyModel

For additional script arguments, see:
python scripts/train_segmentation_ndl.py -h

The command will create three files:
  - MyModel.rwv : a binary copy of the trained model that can be reloaded with pickle
  - MyModel.seg : a copy of the input file, with segmentations added by the model
  - MyModel.csv : a surprise value (1-activation) for each letter in the input file, given its preceding cues

The script tries to intelligently find a surprise threshold for segmenting the
stream, but still needs some work. To resegment the input data (or new data)
with a user-specified threshold, you may try running:

python scripts/ndl_segment.py MyModel.rwv 0.66 data/statistical_learning1.txt --unit=letter --window=2 --readmode=doc --space_char="#" --newline_char="" > MyModel.reseg

## Example 2 - Lexical Processing
Baayen (2010) has demonstrated that the the R-W model can be used to simulate
lexical processing times. This can be done through training a model by
conditioning intra-word cues (e.g., letters, bigrams, trigrams, etc...) on the
occurrence of the word. The association strength between a word and its
intra-word cues is an indicator of lexical processing times. Conceptually, it is
the amount of bottom-up support available for a string of letters being a word.
To train a model, run:

python scripts/simulate_ldrts.py data/simple_corpus.txt --out=MyModel

For additional script arguments, see:
python scripts/train_segmentation_ndl.py -h

The command will create three files:
  - MyModel.rwv : a binary copy of the trained model that can be reloaded with pickle
  - MyModel.csv : Activation of words, given their intra-word cues

