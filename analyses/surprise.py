"""
Various tools for checking model performance on segmenting streams of text.
"""
import models.ndl_tools as ndl_tools

def event_surprisal(model, events, unit, outpath=None):
    """goes through each event and prints surprise for that event to the file
       in .csv format.
    """
    header = "Cues,Outcome,Surprise"
    needs_newchar = False
    if unit in ("ngram", "letter"):
        header = "Cues,Outcome,NewChar,Surprise"
        needs_newchar = True

    fl    = None
    lines = [ ]
    if outpath != None:
        fl = open(outpath, "w")
        fl.write(header + "\n")
    else:
        lines.append(header)
        
    surprisals = [ ]
    last_outcome = None
    for cues,outcomes in events:
        cue_str     = "_".join(cues)
        outcome_str = "_".join(outcomes)
        surprise    = 1.0 - model.activation(cues, outcomes)

        line = ""
        if needs_newchar:
            newchar = outcomes[0][-1]
            line = "%s,%s,%s,%f" % (cue_str, outcome_str, newchar, surprise)
        else:
            line = "%s,%s,%f" % (cue_str, outcome_str, surprise)

        if outpath != None:
            fl.write(line + "\n")
        else:
            lines.append(line)

    if outpath == None:
        return "\n".join(outpath) + "\n"
    else:
        fl.close()

        
        
def segmentation(model, corpus, window, nsize, segmentation_threshold, readmode="line", space_char="#", newline_char="|", outpath=None):
    """requires an ngram-ngram model. Goes over a corpus line-by-line or 
       doc-by-doc and tries to segment it based on encountering surprising
       sequences of ngrams. Returns output as string, or prints to file if
       outpath != None.
    """
    # create our segmenter
    segmenter = ndl_tools.LineSegmenter(model, window, nsize, segmentation_threshold=segmentation_threshold)

    # run the segmentation test
    fl       = None
    if outpath != None:
        fl = open(outpath, "w")
    lines    = [ ]
    buf_size = window + nsize
    buf      = space_char * buf_size
    for line in ndl_tools.FileReader(corpus, space_char=space_char, newline_char=newline_char, readmode=readmode, head_buffer=buf, tail_buffer=buf):
        line = " ".join(segmenter.segment(line))
        if outpath == None:
            lines.append(line)
        else:
            fl.write(line + "\n")

    if fl != None:
        fl.close()
    else:
        return "\n".join(lines)
