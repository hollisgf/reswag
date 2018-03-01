"""
Tools for extracting semantic vectors out of R-W.
"""
import numpy as np, time, models.ndl_tools as ndl_tools, shutil
from gensim.models import Word2Vec



def cues_to_w2v(model, min_ngram_size, max_ngram_size, flanking_spaces, outpath=None, binary=True):
    """sums all of the cues in an outcome and saves that as a vector for the
       outcome.
    """
    tmpfile  = "temp%f.txt" % time.time()
    fl       = open(tmpfile, 'w')
    outcomes = model.outcomes()

    # write the header
    header_written = False
    
    # go over each outcome and generate cues for it
    for outcome in outcomes:
        # generate cues
        string = outcome.lower()
        if flanking_spaces == True:
            string = "#" + string + "#"
        cues    = [ ]
        for i in range(min_ngram_size, max_ngram_size+1):
            cues.extend(ndl_tools.generate_ngrams(string, i))
        cues = set(cues)
        if "#" in cues:
            cues.remove("#")
            
        # get vectors and sum to single
        vecs = [ model.cue_vectors[cue] for cue in cues ]
        vec  = np.sum(vecs, axis=0)

        if header_written == False:
            fl.write("%d %d\n" % (len(outcomes), len(vec)))
            header_written = True
        
        # print results
        line = [ outcome.lower() ] + [ str(v) for v in vec ]
        fl.write(" ".join(line) + "\n")

    # close the file
    fl.close()

    # move to correct location, or save as binary if necessary
    if binary == True:
        w2vmodel = Word2Vec.load_word2vec_format(tmpfile, binary=False)
        w2vmodel.save_word2vec_format(outpath, binary=True)
        os.remove(tmpfile)
    else:
        shutil.move(tmpfile, outpath)
