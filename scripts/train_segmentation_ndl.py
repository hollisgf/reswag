#!/usr/bin/env python
"""
Trains a R-W model as an NDL that learns to segment a stream of input into 
chunks on the basis of finding outcomes that are not well-predicted by the
preceding cues (indicating a segmentation point). This does reasonably well at
being able to identify word boundaries in unsegmented text. For details, see:

Baayen, R. H., Shaoul, C., Willits, J., & Ramscar, M. (2016). Comprehension without segmentation: A proof of concept with naive discriminative learning. Language, Cognition and Neuroscience, 31(1), 106-128.
Chicago	
"""
import sys, argparse, math, scipy.stats
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
    parser.add_argument("--modeltype", type=str, default="rwv", help="rw or rwv.")
    parser.add_argument("--alpha", type=float, default=0.1, help="CS salience.")
    parser.add_argument("--beta", type=float, default=1.0, help="US salience.")
    parser.add_argument("--window", type=int, default=2, help="window size for learning.")
    parser.add_argument("--unit", type=str, default="ngram3", help="what is our cue unit; can be letter, ngramX or word.")
    parser.add_argument("--readmode", type=str, default="line", help="read by lines or docs?")
    parser.add_argument("--space_char", type=str, default="", help="what character should be use to represent spaces. Defaults to removing spaces.")
    parser.add_argument("--newline_char", type=str, default="|", help="what character should be use to represent newlines.")
    parser.add_argument("--out", type=str, default=None, help="location to save the model to.")
    parser.add_argument("--test",type=str, default=None, help="path to a test corpus; if None, use first training corpus.")
    parser.add_argument("--vectorlength", type=int, default=300, help="only used for vector model; species vector length.")
    parser.add_argument("--outcomes_also_cues", type=str2bool, default=False, help="Can outcome vectors also be updated through learning? Relevant only for VNDL")
    parser.add_argument("--segmentation_threshold", type=float, default=None, help="How surprising does an event have to be to result in a segmentation?")

    args = parser.parse_args()
    if args.out == None or "." in args.out:
        raise Exception("Must specify --out path with no extension.")

    if not args.modeltype in ("rw", "rwv"):
        raise Exception("Modeltype must be ndl or vndl.")

    outpath  = args.out + "." + args.modeltype
    logpath  = args.out + ".csv"
    segpath  = args.out + ".seg"
    
    # prepare our model
    if args.modeltype == "rw":
        model = reswag.ResWag(alpha=args.alpha, beta=args.beta)
    else:
        model = reswag.VectorResWag(alpha=args.alpha, beta=args.beta, vectorlength=args.vectorlength, outcomes_also_cues=args.outcomes_also_cues)
        
    # build our iterator information
    window     = args.window
    space_char = args.space_char
    newline_char= args.newline_char

    unit       = args.unit
    nsize      = None
    if unit.startswith("ngram"):
        nsize = int(unit.strip("ngram"))
        unit  = "ngram"
    elif unit == "letter":
        nsize = 1
        unit  = "ngram"

    # create our event iterator
    events = ndl_tools.ForwardPredictionChannel(args.corpora, unit=unit, window=window, space_char=space_char, newline_char=newline_char, nsize=nsize, readmode=args.readmode)

    # train and save the model
    model.process_events(events)
    model.save(outpath)

    # print our surprise at each event to file for analysis; also collect
    # surprise values to determine a good threshold for segmentation.
    testfl = args.test
    if testfl != None:
        events = ndl_tools.ForwardPredictionChannel(args.test, unit=unit, window=window, space_char=space_char, nsize=nsize, readmode=args.readmode)

    # run the tests
    fl = open(logpath, "w")
    
    fl.write("Outcome,NewChar,Surprise\n")
    surprisals = [ ]
    last_outcome = None
    for cues,outcomes in events:
        # check for new line, if so print cues to keep sequence continuous
        outcome  = outcomes[0]
        if last_outcome == None or last_outcome not in cues:
            for letter in cues[0][:-1]:
                fl.write("NA,%s,NA\n" % letter)
            for cue in cues:
                fl.write("%s,%s,NA\n" % (cue,cue[-1]))

        surprise = 1.0 - model.activation(cues, outcome)
        surprisals.append(surprise)
        fl.write("%s,%s,%1.4f\n" % (outcome, outcome[-1], surprise))
        last_outcome = outcome
    fl.close()

    seg_threshold = args.segmentation_threshold
    if seg_threshold == None:
        # calculate the modal surprise (rounded to a reasonable decimal value)
        decimals = 3
        if len(surprisals) > 100000:
            decimals = 4
        surprisals = [ round(val, decimals) for val in surprisals ]
        modal_surprise, modal_count = scipy.stats.mode(surprisals)
        seg_threshold = modal_surprise[0]
    
    segfls = [ args.test ]
    if args.test == None:
        segfls = args.corpora

    # run the segmentation test
    fl = open(segpath, 'w')
    buf_size = window + nsize
    buf      = ("#" if space_char=="" else space_char) * buf_size
    segmenter = ndl_tools.LineSegmenter(model, args.window, nsize, segmentation_threshold=seg_threshold)
    for corpus in segfls:
        for line in ndl_tools.FileReader(corpus, space_char=space_char, newline_char=newline_char, readmode=args.readmode, head_buffer=buf, tail_buffer=buf):
            fl.write(" ".join(segmenter.segment(line)) + "\n")
    fl.close()
    


if __name__ == "__main__":
    main()
    
