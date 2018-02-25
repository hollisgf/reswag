#!/usr/bin/env python
"""
Tools for doing word segmentation with NDL

Author: Geoff Hollis
email : hollis-at-ualberta-dot-ca

It has been released under the Creative Commons Attribution 4.0 International
license: http://creativecommons.org/licenses/by/4.0/ .
"""
import sys, argparse
sys.path.append(".")
import models.ndl_tools as ndl_tools, models.reswag as reswag

def main():
    parser = argparse.ArgumentParser(description='Interface for performing segmentation with a R-W model trained as an NDL.')
    parser.add_argument("model", type=str, help="path to .rw or .rwv model that has been trained to do text segmentation.")
    parser.add_argument("segmentation_threshold", type=float, default=None, help="How surprising does an event have to be to result in a segmentation? Values of 0.9 - 0.95 work for most real corpora; lower values are typically needed for artificial data.")
    parser.add_argument("corpora", type=str, nargs='*', help="Paths to .txt files containing corpora to segment.")
    parser.add_argument("--window", type=int, default=2, help="window size for predicting. Default 2.")
    parser.add_argument("--unit", type=str, default="ngram3", help="what is our cue unit; can be letter, ngramX or word. Default ngram3.")
    parser.add_argument("--readmode", type=str, default="line", choices=["line", "doc"], help="read by lines or docs? Default line.")
    parser.add_argument("--space_char", type=str, default="#", help="what character should be use to represent spaces? Defaults to #. Spaces can be removed with the empty string.")
    parser.add_argument("--newline_char", type=str, default="|", help="what character should be use to represent newlines. Defaults to |. Newlines can be removed with the empty string")

    args = parser.parse_args()

    # load our model
    model = None
    if args.model.endswith(".rwv"):
        model = reswag.VectorResWag.load(args.model)
    else:
        model = reswag.ResWag.load(args.model)

    # get our ngram size
    unit       = args.unit
    nsize      = None
    if unit.startswith("ngram"):
        nsize = int(unit.strip("ngram"))
        unit  = "ngram"
    elif unit == "letter":
        nsize = 1
        unit  = "ngram"
        
    # create our segmenter
    segmenter = ndl_tools.LineSegmenter(model, args.window, nsize, segmentation_threshold=args.segmentation_threshold)

    # run the segmentation test
    buf_size = args.window + nsize
    buf      = ("#" if args.space_char=="" else args.space_char) * buf_size
    for corpus in args.corpora:
        for line in ndl_tools.FileReader(corpus, space_char=args.space_char, newline_char=args.newline_char, readmode=args.readmode, head_buffer=buf, tail_buffer=buf):
            print " ".join(segmenter.segment(line))
    
    

if __name__ == "__main__":
    sys.exit(main())
    
