#!/usr/bin/env python
"""
A general framework for training the RW model as an NDL model. Specifies a 
corpus, cue types, outcome types, and uses these details to train a model.
"""
import argparse, sys
sys.path.append(".")
import models.reswag as reswag, models.ndl_tools as ndl_tools
import analyses.surprise, analyses.lexical_processing



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Interface for training an NDL model.')
    parser.add_argument("label", type=str, help="a string identifier for saving the model. Form is path/to/identifier ; .rw/.rwv and parameter settings will be appended to file name.")
    parser.add_argument("cue_type", type=str, default=None, choices=["letter", "ngram", "word"], help="What is the cue type being conditioned?")
    parser.add_argument("outcome_type", type=str, default=None, choices=["letter", "ngram", "word", "lexome"], help="What are cues being conditioned on?")
    parser.add_argument("window", type=int, default=2, help="How many cues should be used for conditioning?")
    parser.add_argument("corpora", type=str, nargs='*', help="Paths to .txt files containing corpora to (sequentially) train the model on.")
    parser.add_argument("--ngram_grain", type=str, default=1, help="Specify a size of ngram to use. When conditioning on words/lexomes, a range can be provided, e.g., 1-3. Default 1.")
    parser.add_argument("--flanking_spaces", type=str2bool, default=True, choices=[True,False], help="When conditioning ngrams on words or lexomes, specify whether spaces at the beginning and end of a word should count for building cues. Default True.")
    parser.add_argument("--modeltype", type=str, default="rwv", choices=["rw", "rwv"], help="rw or rwv. Default rwv.")
    parser.add_argument("--alpha", type=float, default=0.10, help="CS salience, (0, 1]. Default 0.02. Alpha * cue number should always be less than 1.0.")
    parser.add_argument("--beta", type=float, default=1.0, help="US salience, (0, 1]. Default 1.00.")
    parser.add_argument("--vectorlength", type=int, default=300, help="only used for vector model; species vector length. Default 300.")
    parser.add_argument("--vectortype", type=str, default="random", choices=["random", "ortho", "sensory"], help="What is our generation method for vectors? Random values, orthogonal, or sensory (tries to preserve similarities between words as vector correlations). Default random.")
    parser.add_argument("--outcomes_also_cues", type=str2bool, default=False, choices=[True,False], help="Can outcome vectors also be updated through learning? Relevant only for RWV. Default False")
    parser.add_argument("--iterations", type=int, default=1, help="how many passes over the corpus do we make? Default 1.")
    parser.add_argument("--mincount", type=int, default=0, help="how many times does an outcome need to occur to be considered for training. Default 0.")
    parser.add_argument("--force_train", type=str2bool, default=False, choices=[True,False], help="Override sanity checking for model runnability.")
    parser.add_argument("--readmode", type=str, default="line", choices=["line","doc"], help="Should the corpus be read line by line, or document by document? Defaults to line.")
    parser.add_argument("--space_char", type=str, default=None, help="When cues are ngrams or letters, what character should represent spaces? Can strip spaces with an empty string. Default '#' for letter cues and ' ' for word cues.")
    parser.add_argument("--newline_char", type=str, default=None, help="When cues are ngrams or letters, what character should represent a newline? Can strip newlines with an empty string. Default |.")
    parser.add_argument("--buffer_ends", type=str2bool, default=True, choices=[True,False], help="When cues are ngrams or letters, buffer the beginnings and endings of text units {lines,docs} with spaces so all letters receive conditioning. Default True.")
    args = parser.parse_args()

    # where are we saving the model?
    savepath = args.label
    
    # sanity checking
    if args.force_train == False and args.alpha * args.window >= 0.5:
        raise Exception("You are trying to train a model with too large of an alpha for your window size. It is advised that alpha * window be <= 0.5. Your is currently %f. Run with --force_train=True to override" % (args.alpha * args.window))

    # start figuring out what our experience channel will be
    cue_type     = args.cue_type
    outcome_type = args.outcome_type
    space_char   = args.space_char
    newline_char = args.newline_char

    # figure out default space and newline chars
    if space_char == None:
        if cue_type == "word":
            space_char = " "
        elif cue_type in ("letter", "ngram"):
            space_char = "#"
    if newline_char == None:
        if cue_type == "word":
            newline_char = " |"
        elif cue_type in ("letter", "ngram"):
            newline_char = "|"
    
    # if we are using letter or ngram cues, figure out the details
    min_ngram_size = 0
    max_ngram_size = 0
    if cue_type == "letter":
        if not outcome_type == "letter":
            cue_type = "ngram"
        min_ngram_size=1
        max_ngram_size=1

        savepath += "_letter"
    elif cue_type == "ngram":
        if type(args.ngram_grain) == int:
            min_ngram_size = args.ngram_grain
            max_ngram_size = args.ngram_grain
        elif "-" in args.ngram_grain:
            min_ngram_size, max_ngram_size = args.ngram_grain.split("-")
            min_ngram_size = max(1, int(min_ngram_size))
            max_ngram_size = max(1, int(max_ngram_size))
            if min_ngram_size > max_ngram_size:
                min_ngram_size, max_ngram_size = max_ngram_size, min_ngram_size
        else:
            min_ngram_size = int(args.ngram_grain)
            max_ngram_size = min_ngram_size

        if min_ngram_size == max_ngram_size:
            savepath += "_ngram%d" % min_ngram_size
        else:
            savepath += "_ngram%d-%d" % (min_ngram_size, max_ngram_size)
    elif cue_type == "word":
        savepath += "_word"
    
    # can't condition a range of ngrams on a specific ngram size
    if args.outcome_type == "ngram" and min_ngram_size != max_ngram_size:
        raise Exception("When conditioning ngrams on ngrams, only one grain size may be used.")
    savepath += "_to_" + args.outcome_type
    savepath += "_window%d" % args.window
            
    forward_prediction_channel = False
    ng_to_word_channel         = False
    lexome_channel             = False
                
    # if our cue type is the same as our outcome type, we are by default using
    # a forward prediction channel, conditioning cues in a sliding window on
    # the next item.
    if cue_type == outcome_type:
        forward_prediction_channel = True
    # otherwise check if we are conditioning ngrams on words
    elif cue_type == "ngram" and outcome_type == "word":
        ng_to_word_channel = True
    # finally, check if we are conditioning on lexomes
    elif outcome_type == "lexome" and cue_type in ("ngram", "word"):
        lexome_channel = True

    # build the experience channel
    events = None
    if lexome_channel:
        raise Exception("Lexome conditioning is under development.")
    elif ng_to_word_channel:
        events = ndl_tools.NgramToWordChannel(corpora=args.corpora, mingrain=min_ngram_size, maxgrain=max_ngram_size, flanking_spaces=args.flanking_spaces, mincount=args.mincount)
    elif forward_prediction_channel:
        events = ndl_tools.ForwardPredictionChannel(corpora=args.corpora, unit=outcome_type, window=args.window, space_char=space_char, newline_char=newline_char, buffer_ends=args.buffer_ends, nsize=min_ngram_size, readmode=args.readmode)
    else:
        raise Exception("Your cue/outcome combination did not make sense. See documentation for help.")

    # add other information to the file description
    if args.mincount > 0:
        savepath += "_mincount%d" % args.mincount
    
    savepath += "_a%0.2f_b%0.2f" % (args.alpha, args.beta)

    if outcome_type == "word":
        savepath += "_flanking_spaces%s" %("T" if args.flanking_spaces==True else "F")
        
    if args.modeltype == "rwv":
        savepath += "_len%d_%s" % (args.vectorlength, args.vectortype)
        if args.outcomes_also_cues:
            ############
            # FINISH ME
            ############
            pass
            
    if args.iterations > 1:
        savepath += "_iter%d" % (args.iterations)

    ext       = "." + args.modeltype
    savepath += ext


    # create the model
    model = None
    if args.modeltype == "rw":
        model = reswag.ResWag(alpha=args.alpha, beta=args.beta)
    else:
        model = reswag.VectorResWag(alpha=args.alpha, beta=args.beta, vectorlength=args.vectorlength, vectortype=args.vectortype)

    # train the model
    for i in xrange(args.iterations):
        model.process_events(events)

    # save the model
    model.save(savepath)

    # run various tests
    if forward_prediction_channel and outcome_type in ("ngram", "letter", "word"):
        # print event surprisal
        path = savepath.replace(ext, ".surprise")
        analyses.surprise.event_surprisal(model, events, unit=outcome_type, outpath=path)
        
        # segment the first corpus
        if outcome_type != "word":
            path = savepath.replace(ext, ".seg")
            analyses.surprise.segmentation(model, corpus=args.corpora[0], window=args.window, nsize=max_ngram_size, segmentation_threshold=0.67, readmode=args.readmode, space_char=space_char, newline_char=newline_char, outpath=path)

    if ng_to_word_channel:
        # print word activation and vector magnitude
        path = savepath.replace(ext, ".process")
        analyses.lexical_processing.processing_measures(model, min_ngram_size=min_ngram_size, max_ngram_size=max_ngram_size, flanking_spaces=args.flanking_spaces, outpath=path)

        
if __name__ == "__main__":
    main()
