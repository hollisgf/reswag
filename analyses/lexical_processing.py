"""
Various tools for checking model performance on learning cue-word associations.
"""
import models.reswag as reswag, models.ndl_tools as ndl_tools



def processing_measures(model, min_ngram_size, max_ngram_size, flanking_spaces, outpath=None):
    """requires an ngram-word model. Returns str of results, or prints to file
       if outpath provided.
    """
    is_vector_model = hasattr(model, "vectorlength")
    header = "Word,Cues,Activation"
    if is_vector_model:
        header = "Word,Cues,Activation,Magnitude"

    fl    = None
    lines = [ ]
    if outpath != None:
        fl = open(outpath, "w")
        fl.write(header+"\n")
    else:
        lines.append(header)
        
    for outcome in model.outcomes():
        chars   = outcome.lower()
        if flanking_spaces:
            chars = "#" + chars + "#"
            
        cues = [ ]
        for i in xrange(min_ngram_size, max_ngram_size+1):
            cues.extend(ndl_tools.generate_ngrams(chars, i))
        while "#" in cues:
            cues.remove("#")
        
        activation = model.activation(cues, outcome)
        data = [ outcome.lower(), "_".join(cues), activation ]
        if is_vector_model:
            data.append(model.cue_magnitude(cues))

        data = ",".join([ str(d) for d in data ])
        if outpath == None:
            lines.append(data)
        else:
            fl.write(data + "\n")

    if outpath == None:
        return "\n".join(lines) + "\n"
    else:
        fl.close()
