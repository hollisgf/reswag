"""
sums all of the cues that map to an outcome (word or lexome) and saves these
vectors in word2vec format for semantic analysis.
"""
import argparse, sys
sys.path.append(".")
import analyses.semantics, models.reswag as reswag



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Converts an NDL model conditioned on words or lexomes to a word2vec model, by summing cue vectors for outcomes.')
    parser.add_argument("model", type=str, help="Path to model.")
    parser.add_argument("min_ngram_grain", type=int, help="What is the min ngram size that contributes to cues?")
    parser.add_argument("max_ngram_grain", type=int, help="What is the max ngram size that contributes to cues?")
    parser.add_argument("flanking_spaces", type=str2bool, help="Are flanking spaces used for determining cues?")
    parser.add_argument("path", type=str, help="Where should this model be saved to?")
    args = parser.parse_args()

    model = None
    if args.model.endswith(".rw"):
        model = reswag.ResWag.load(args.model)
    else:
        model = reswag.VectorResWag.load(args.model)

    analyses.semantics.cues_to_w2v(model, args.min_ngram_grain, args.max_ngram_grain, args.flanking_spaces, args.path, binary=True)
    
    


if __name__ == "__main__":
    main()
    
