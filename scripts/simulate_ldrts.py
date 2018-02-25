#!/usr/bin/env python
"""
Trains a R-W model an an NDL to simulate LDRTs. LDRTs are simulated as the log
reciprocal of the summed association strengths between cues available within a 
word and the word itself. For details, see Baayen (2010):

Baayen, R. H. (2010). Demythologizing the word frequency effect: A discriminative learning perspective. The Mental Lexicon, 5(3), 436-461.

Author: Geoff Hollis
email : hollis-at-ualberta-dot-ca

It has been released under the Creative Commons Attribution 4.0 International
license: http://creativecommons.org/licenses/by/4.0/ .
"""
import argparse, math, time, sys
sys.path.append(".")
import models.reswag as reswag, models.ndl_tools as ndl_tools

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser = argparse.ArgumentParser(description='Interface for training an NDL model.')
    parser.add_argument("corpora", type=str, nargs='*', help="Paths to .txt files containing corpora to (sequentially) train the model on.")
    parser.add_argument("--modeltype", type=str, default="rwv", choices=["rw", "rwv"], help="rw or rwv. Default rwv.")
    parser.add_argument("--alpha", type=float, default=0.02, help="CS salience, (0, 1]. Default 0.02.")
    parser.add_argument("--beta", type=float, default=1.0, help="US salience, (0, 1]. Default 1.00.")
    parser.add_argument("--flanking_spaces", type=str2bool, default=True, choices=[True,False], help="Should we use flanking spaces? e.g., dog -> #dog#. Default True.")
    parser.add_argument("--out", type=str, default=None, help="location to save the model to.")
    parser.add_argument("--vectorlength", type=int, default=300, help="only used for vector model; species vector length. Default 300.")
    parser.add_argument("--orthogonal", type=str2bool, default=False, choices=[True,False], help="Are outcome vectors forced to be orthogonal? Default False.")
    parser.add_argument("--outcomes_also_cues", type=str2bool, default=False, choices=[True,False], help="Can outcome vectors also be updated through learning? Relevant only for RWV. Default False")
    parser.add_argument("--cuegrain", type=int, default=2, help="what is the maximum grain of our cue size. 1 = letters, 2 = letters+bigrams, etc... . Default 2.")
    parser.add_argument("--iterations", type=int, default=1, help="how many passes over the corpus do we make? Default 1.")
    parser.add_argument("--mincount", type=int, default=0, help="how many times does an outcome need to occur to be considered for training. Default 0.")

    args = parser.parse_args()
    if args.out == None or "." in args.out:
        raise Exception("Must specify --out path with no extension.")

    if not args.modeltype in ("rw", "rwv"):
        raise Exception("Modeltype must be rw or rwv.")

    outpath  = args.out + "." + args.modeltype
    logpath  = args.out + ".csv"
    
    # prepare our model
    if args.modeltype == "rw":
        model = reswag.ResWag(alpha=args.alpha, beta=args.beta)
    else:
        model = reswag.VectorResWag(alpha=args.alpha, beta=args.beta, vectorlength=args.vectorlength, outcomes_also_cues=args.outcomes_also_cues, force_orthogonal=args.orthogonal)
        
    # build our iterator information
    cuegrain        = args.cuegrain
    flanking_spaces = args.flanking_spaces

    # create our iterator
    events = ndl_tools.NgramToWordChannel(corpora=args.corpora, maxgrain=cuegrain, flanking_spaces=flanking_spaces, mincount=args.mincount)
    
    # train and save the model
    for i in xrange(args.iterations):
        model.process_events(events)
    model.save(outpath)

    # report simulated lexical processing times for all encountered outcomes
    fl = open(logpath, "w")
    fl.write("Word,Activation\n")
    completed_outcomes = set()
    for cues, outcomes in events:
        outcome = outcomes[0]
        if outcome in completed_outcomes:
            continue
        completed_outcomes.add(outcome)
        assoc = model.activation(cues, outcome)
        fl.write("%s,%f\n" % (outcome.lower(), assoc))
    fl.close()

if __name__ == "__main__":
    main()
    
