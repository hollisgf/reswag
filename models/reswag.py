"""
Provides two implementations of the Rescorla-Wagner model. The first is its
classic form, described in Rescorla & Wagner (1972). The second its vector
approximation, described in Hollis (under review). The vector approximation
method uses a vector-based method that eliminates the need to update for 
absent outcomes to reduce the computational cost of the model. However, it does
sacrifice some accuracy in learned association strengths as a consequence.

Author: Geoff Hollis
email : hollis-at-ualberta-dot-ca

It has been released under the Creative Commons Attribution 4.0 International
license: http://creativecommons.org/licenses/by/4.0/ .
"""
import pickle, numpy as np, ndl_tools
import scipy.spatial.distance as dist


def vector_magnitude(vector):
    # return (vector ** 2.0).sum() ** 0.5 
    return np.sqrt(vector.dot(vector))

class ResWag:
    """Implements the classic R-W model.
    """
    def __init__(self, alpha=0.1, beta=1.0, lamda=1.0):
        """creates a new R-W model.
        """
        # learning parameters
        self.alpha = alpha
        self.beta  = beta
        self.lamda = lamda
        
        # maps cues to vectors of outcome associations
        self.cue_associations = { }

        # maps outcomes to indices, used to reference associations in cue vectors
        self.outcome_indices = { }

        # used for training the model
        self.__zero_vector = np.zeros(0)
        
    def save(self, fname):
        pickle.dump(self, open(fname,'w'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname,'r'))

    @property
    def cue_vectors(self):
        return self.cue_associations
    
    def cues(self):
        return self.cue_associations.keys()

    def outcomes(self):
        return self.outcome_indices.keys()
    
    def __resize_cues(self):
        """Does as it says.
        """
        newsize = len(self.outcome_indices)
        for cue in self.cues():
            assocs = self.cue_associations[cue]
            assocs = np.append(assocs, [0.] * (newsize - len(assocs)))
            self.cue_associations[cue] = assocs

        self.__zero_vector = np.zeros(len(self.outcome_indices))
    
    def __create_cue(self, cue):
        """registers a new cue in the model.
        """
        cue_created = False
        try:
            assocs = self.cue_associations[cue]
        except:
            self.cue_associations[cue] = np.zeros(len(self.outcome_indices))
            cue_created = True
        return cue_created

    def __create_cues(self, cues):
        for cue in cues:
            self.__create_cue(cue)
        
    def __create_outcome(self, outcome, resize=True):
        """registers a new outcome in the model. This should rarely be done on
           the fly, because it requires expansion of the cue association 
           vectors. Register all cues and outcomes prior to training.
        """
        outcome_created = False
        try:
            index = self.outcome_indices[outcome]
        except:
            index = len(self.outcome_indices)
            self.outcome_indices[outcome] = index
            outcome_created = True

        # resize all of our outcome weight vectors
        if resize and outcome_created == True:
            self.__resize_cues()

        if outcome_created:
            return index
        return -1

    def __create_outcomes(self, outcomes):
        """creates multiple outcomes; a little faster for resizing than doing 
           each outcome individually.
        """
        old_num_outcomes = len(self.outcome_indices)
        for outcome in outcomes:
            self.__create_outcome(outcome, resize=False)

        if old_num_outcomes < len(self.outcome_indices):
            self.__resize_cues()
    
    def create_cues_and_outcomes(self, events):
        """populates cues and outcomes based on unique cues and outcomes in
           the events list.
        """
        # collect all of our unique cues and outcomes
        unique_cues     = set()
        unique_outcomes = set()
        for cues, outcomes in events:
            unique_cues.update(cues)
            unique_outcomes.update(outcomes)

        self.__create_outcomes(unique_outcomes)
        self.__create_cues(unique_cues)
    
    def process_events(self, events):
        """events is an iterator over (cue,outcome) pairs. Processes each
           pair with learn_contingency.
        """
        self.create_cues_and_outcomes(events)

        for event in events:
            self.learn_contingency(*event)

    def learn_contingency(self, cues, outcomes):
        """processes one learning event, associating cues with outcomes and
           also learning from negative evidence about absent outcomes. This is
           the heart of the R-W model.
        """
        # cues and outcomes only used once
        cues     = set(cues)
        outcomes = set(outcomes)
        
        # build our update multiplier
        lamda_vector  = self.__zero_vector
        lamda_vector *= 0.0
        
        # find the outcome indices to use for learning.
        for outcome in outcomes:
            lamda_vector[self.outcome_indices[outcome]] = 1.0

        # calculate outcome associations
        assoc_vecs = [ self.cue_associations[cue] for cue in cues ]
        Vtot       = np.sum(assoc_vecs, axis=0)

        # calculate errors
        learning_rate = self.alpha * self.beta
        errors        = (lamda_vector - Vtot)
        deltas        = learning_rate * errors
        
        # update association strengths for cues that appeared
        for cue in cues:
            self.cue_associations[cue] += deltas

    def activation(self, cues, outcomes):
        """Returns the activation of the outcome, given the cues.
        """
        if type(outcomes) == str:
            outcomes = [ outcomes ]
        
        # remove duplicates
        cues  = set(cues)
        
        # calculate outcome associations
        assoc_vecs = [ self.cue_associations[cue] for cue in cues ]
        Vtot       = np.sum(assoc_vecs, axis=0)

        return Vtot[[self.outcome_indices[outcome] for outcome in outcomes]].sum()

    def most_active(self, cues, topn=10):
        """returns the topn most active outcomes, given the set of cues.
           Returns as a list of (cue, activation) pairs.
        """
        # remove duplicates
        cues = set(cues)
        
        # calculate outcome associations
        assoc_vecs = [ self.cue_assocations[cue] for cue in cues ]
        Vtot       = np.sum(assoc_vecs, axis=0)
        
        activities = { }
        for outcome, index in self.outcome_indices.iteritems():
            activities[outcome] = Vtot[index]

        outcomes = activities.keys()
        outcomes.sort(lambda a,b: cmp(activities[a], activities[b]), reverse=True)

        if topn != None:
            outcomes = outcomes[:topn]

        return [ (outcome, activities[outcome]) for outcome in outcomes ]



class VectorResWag:
    """Vector approximation of the R-W model. This model allows for 
       discriminative learning without having to update for absent outcomes. 
       Outcomes are represented as uncorrelated vectors. Cues do not have 
       association strength with an outcome. Instead, they are also represented
       as vectors a cue-outcome association is handled as updating the cue's 
       vector towards the outcome's vector. Because the outcome's vector is 
       uncorrelated with all other outcomes, this neccessarily also decorrelates
       the cue with other outcomes.

       Association strength between cue and outcome is calculated as the sum of
       all values in the cue vector, multiplied by the sign of the values at 
       the same indices in the outcome vector, normalized by the sum of the 
       outcome vector.

       Model is fully described in:
       Hollis (under review).
    """
    def __init__(self, alpha=0.1, beta=1.0, lamda=1.0, vectorlength=1000, outcomes_also_cues=False, vectortype="random"):
        """creates a new model. vectorlength must be sufficiently long
           so that two randomly generated and normalized vectors are highly
           unlikely to have incidental correlation. The longer the vectors, the
           more accurate the model is at approximating behavior of the classic 
           R-W model.
        """
        # set learning parameters
        self.alpha = alpha
        self.beta  = beta
        self.lamda = lamda # determines vector magnitude in the VNDL model
        self.vectorlength = vectorlength
        self.vectortype = vectortype

        if vectortype not in ["random", "ortho", "sensory"]:
            raise Exception("Unrecognized outcome vector type, %s." % vectortype)
        
        # are outcomes static, or can their vectors be updated through learning
        self.outcomes_also_cues = outcomes_also_cues

        # table of outcome and cue vectors
        self.outcome_vectors = { }
        self.__cue_vectors   = { }

        # if sensory vectors are used, build up vectors for outcomes from
        # vectors specific to shorter substrings of the outcome.
        self.vectortype = vectortype
        if vectortype == "sensory":
            self.sense_table = { }
        
    def save(self, fname):
        pickle.dump(self, open(fname,'w'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname,'r'))

    @property
    def cue_vectors(self):
        if self.outcomes_also_cues:
            return self.outcome_vectors
        return self.__cue_vectors

    def cues(self):
        return self.cue_vectors.keys()

    def outcomes(self):
        return self.outcome_vectors.keys()

    def __create_vector(self, outcome):
        """creates and registers an outcome vector
        """
        {
            "random" : self.__create_random_vector,
            "ortho"  : self.__create_orthogonal_vector,
            "sensory": self.__create_sensory_vector,
        }[self.vectortype](outcome)
    
    def __create_sensory_vector(self, outcome):
        # each ngram from length 1-3 contributes to the sensory vector
        str_for_cues = "#" + outcome + "#"
        substrings = [ ]
        for i in xrange(1, min(3, len(str_for_cues)) + 1):
            substrings.extend(ndl_tools.generate_ngrams(str_for_cues, i))
        substrings = set(substrings)
        substrings.remove("#")

        # sum up all of the subparts
        vec = np.zeros(self.vectorlength)
        for string in substrings:
            try:
                vec += self.sense_table[string]
            except:
                newsubvec = np.float64(np.random.randn(self.vectorlength))
                vec      += newsubvec
                self.sense_table[string] = newsubvec

        # normalize and save
        vec = self.lamda * (vec / np.linalg.norm(vec))
        self.outcome_vectors[outcome] = vec
    
    def __create_random_vector(self, outcome):
        vec = np.float64(np.random.randn(self.vectorlength))
        vec = self.lamda * (vec / np.linalg.norm(vec))
        self.outcome_vectors[outcome] = vec
        
    def __create_orthogonal_vector(self, outcome):
        if len(self.outcome_vectors) >= self.vectorlength:
            raise Exception("Cannot create new vector; maximum number of orthogonal vectors already created.")

        vec = np.zeros(self.vectorlength)
        vec[len(self.outcome_vectors)] = 1.0
        self.outcome_vectors[outcome]  = vec
    
    def __create_cue(self, cue):
        """registers a new cue in the model.
        """
        if cue not in self.cue_vectors:
            if self.outcomes_also_cues:
                self.__create_outome(cue)
            else:
                self.cue_vectors[cue] = np.zeros(self.vectorlength)

    def __create_cues(self, cues):
        for cue in cues:
            self.__create_cue(cue)
    
    def __create_outcome(self, outcome):
        """registers a new outcome in the model.
        """
        if outcome in self.outcome_vectors:
            return
        self.__create_vector(outcome)

    def __create_outcomes(self, outcomes):
        for outcome in outcomes:
            self.__create_outcome(outcome)
    
    def create_cues_and_outcomes(self, events):
        """populates cues and outcomes based on unique cues and outcomes in
           the events list.
        """
        # collect all of our unique cues and outcomes
        unique_cues     = set()
        unique_outcomes = set()
        for cues, outcomes in events:
            unique_cues.update(cues)
            unique_outcomes.update(outcomes)

        self.__create_cues(unique_cues)
        self.__create_outcomes(unique_outcomes)
    
    def process_events(self, events):
        """events is an iterator over (cue,outcome) pairs. Processes each
           pair with learn_contingency.
        """
        for event in events:
            self.learn_contingency(*event)

    def learn_contingency(self, cues, outcomes):
        """processes one learning event, associating cues with outcomes.
        """
        # no duplicates
        outcomes = set(outcomes)
        cues     = set(cues)
        
        # make sure our cues and outcomes exist
        for cue in cues:
            if cue not in self.cue_vectors:
                self.__create_cue(cue)
        for outcome in outcomes:
            if outcome not in self.outcome_vectors:
                self.__create_outcome(outcome)

        # build our cue and outcome vectors
        cue_vecs     = [ self.cue_vectors[cue] for cue in cues ]
        outcome_vecs = [ self.outcome_vectors[outcome] for outcome in outcomes ]

        # calculate our V and lamda vectors
        V      = np.sum(cue_vecs,     axis=0)
        lamda  = np.sum(outcome_vecs, axis=0)

        # the magnitude of the sum of correlated vectors is longer than the
        # individual vectors. With many cues, this ends up resulting in huge
        # errors that amplify themselves on future learning outcomes. This is
        # a correction to prevent many-cue situations from amplifying
        # prediction error.
        #magnitude_correction = 1.0 / len(cue_vecs)**0.5
        magnitude_correction = 1.0
        
        # calculate error vector, and learning delta
        error         = (lamda - V)
        delta         = self.alpha * self.beta * error# * magnitude_correction
        
        # update our cue vectors
        for cue in cues:
            self.cue_vectors[cue] += delta

    def cue_magnitude(self, cues):
        """returns the vector magnitude of the cue cues
        """
        if type(cues) == str:
            cues = [ cues ]

        vecs = [ self.cue_vectors[cue] for cue in cues ]
        vec  = np.sum(vecs, axis=0)
        return vector_magnitude(vec)

    def activation(self, cues, outcomes):
        """returns the activation of the outcomes, given the cues
        """
        if type(outcomes) == str:
            outcomes = [ outcomes ]
        
        # remove duplicates
        cues        = set(cues)

        # collapse cue vectors
        cue_vecs    = [ self.cue_vectors[cue] for cue in cues ]
        cue_vec     = np.sum(cue_vecs, axis=0, dtype=np.float64)
        outcome_vecs= [ self.outcome_vectors[outcome] for outcome in outcomes ]
        outcome_vec = np.sum(outcome_vecs, axis=0, dtype=np.float64)

        return (1.0 - dist.cosine(cue_vec, outcome_vec)) * vector_magnitude(cue_vec)

    def most_active(self, cues, topn=10):
        """returns the topn most active outcomes, given the set of cues.
           Returns as a list of (cue, activation) pairs.
        """
        # remove duplicates
        cues = set(cues)
        
        activities = { }
        for outcome,vec in self.outcome_vectors.iteritems():
            activities[outcome] = self.activation(cues, outcome)

        outcomes = activities.keys()
        outcomes.sort(lambda a,b: cmp(activities[a], activities[b]), reverse=True)

        if topn != None:
            outcomes = outcomes[:topn]

        return [ (outcome, activities[outcome]) for outcome in outcomes ]
