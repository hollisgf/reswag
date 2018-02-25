"""
Provides two implementations of the Rescorla-Wagner model. The first is its
classic form, described in Rescorla & Wagner (1972). The second its vector
approximation, described in Hollis (under review). The vector approximation
method uses a vector-based method that eliminates the need to update for 
absent outcomes to reduce the computational cost of the model. However, it does
sacrifice some accuracy in learned association strengths as a consequecne.
"""
import pickle, numpy as np



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
        
        # maps outcomes to vectors of cue weights
        self.outcome_weights = { }

        # maps cues to indices, used to reference weights in outcome vectors
        self.cue_indices = { }

        # used for training the model
        self.__zero_vector = np.zeros(0)
        
    def save(self, fname):
        pickle.dump(self, open(fname,'w'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname,'r'))

    def cues(self):
        return self.cue_indices.keys()

    def outcomes(self):
        return self.outcome_weights.keys()
    
    def __resize_outcomes(self):
        """Does as it says.
        """
        newsize = len(self.cue_indices)
        outcomes= self.outcome_weights.keys()
        for outcome in outcomes:
            weights = self.outcome_weights[outcome]
            weights = np.append(weights, [0.])
            self.outcome_weights[outcome] = weights

        self.__zero_vector = np.zeros(len(self.cue_indices))
        
    def __create_cue(self, cue, resize=True):
        """registers a new cue in the model. This requires expansion of the
           outcome vectors and can be quite costly if called often. In practice,
           all necessary cues should be created prior to model training. 
           
           Returns the cue's index, or -1 if no cue was created.
        """
        cue_created = False
        try:
            index = self.cue_indices[cue]
        except:
            index = len(self.cue_indices)
            self.cue_indices[cue] = index
            cue_created = True

        # resize all of our outcome weight vectors
        if resize and cue_created == True:
            self.__resize_outcomes()

        if cue_created:
            return index
        return -1

    def __create_cues(self, cues):
        """creates multiple cues; a little faster for resizing than doing each
           cue individually.
        """
        old_num_cues = len(self.cue_indices)
        for cue in cues:
            self.__create_cue(cue, resize=False)

        if old_num_cues < len(self.cue_indices):
            self.__resize_outcomes()
    
    def __create_outcome(self, outcome):
        """registers a new outcome in the model. This should NEVER be done
           after model training has started. It will interfere with the model's
           quality, since prior training events will not have been able to 
           learn from negative evidence (the absence of an outcome). 
        
           Returns True/False depending on whether the outcome was created
        """
        outcome_created = False
        try:
            weights = self.outcome_weights[outcome]
        except:
            self.outcome_weights[outcome] = np.zeros(len(self.cue_indices))
            outcome_created = True
        return outcome_created

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
        self.create_cues_and_outcomes(events)

        for event in events:
            self.learn_contingency(*event)

    def learn_contingency(self, cues, outcomes):
        """processes one learning event, associating cues with outcomes and
           also learning from negative evidence about absent outcomes. This is
           the heart of the R-W model.
        """
        # build our update multiplier
        update_vector  = self.__zero_vector
        update_vector *= 0.0
        
        # find the weight indices to use for learning. Cues are only
        # used once.
        for cue in set(cues):
            update_vector[self.cue_indices[cue]] = 1.0
        present_outcomes = set(outcomes)

        # go over all outcomes and update weight vectors
        learning_rate = self.alpha * self.beta
        for outcome in self.outcome_weights.keys():
            weights = self.outcome_weights[outcome]
            lamda   = self.lamda if outcome in present_outcomes else 0.0
            activity= (weights * update_vector).sum()
            
            # save a little computation
            if activity == lamda:
                continue
            
            # calculate weight delta and update weights
            delta    = learning_rate * (lamda - activity)
            weights += delta * update_vector

    def activation(self, cues, outcome):
        """Returns the activation of the outcome, given the cues.
        """
        # remove duplicates
        cues = set(cues)
        
        indices = [ self.cue_indices[cue] for cue in cues ]
        return self.outcome_weights[outcome][indices].sum()

    def most_active(self, cues, topn=10):
        """returns the topn most active outcomes, given the set of cues.
           Returns as a list of (cue, activation) pairs.
        """
        # remove duplicates
        cues = set(cues)
        
        activities = { }
        indices = [ self.cue_indices[cue] for cue in cues ]
        for outcome,vec in self.outcome_weights.iteritems():
            activities[outcome] = vec[indices].sum()

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
    def __init__(self, alpha=0.1, beta=1.0, lamda=1.0, vectorlength=1000, outcomes_also_cues=False, force_orthogonal=False):
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
        self.force_orthogonal = force_orthogonal
        
        # are outcomes static, or can their vectors be updated through learning
        self.outcomes_also_cues = outcomes_also_cues

        # table of outcome and cue vectors
        self.outcome_vectors = { }
        self.__cue_vectors   = { }
        
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
    
    def __generate_vector(self, length, magnitude=None):
        vec = np.float64(np.random.randn(length))
        return (magnitude if magnitude is not None else self.lamda) * (vec / np.linalg.norm(vec))
    
    def __create_cue(self, cue):
        """registers a new cue in the model.
        """
        if cue not in self.cue_vectors:
            if self.outcomes_also_cues:
                self.__create_outome(cue)
            else:
                self.cue_vectors[cue] = np.zeros(self.vectorlength)
                #self.__generate_vector(self.vectorlength)

    def __create_cues(self, cues):
        for cue in cues:
            self.__create_cue(cue)
    
    def __create_outcome(self, outcome):
        """registers a new outcome in the model.
        """
        if outcome in self.outcome_vectors:
            return
        
        if self.force_orthogonal:
            if len(self.outcome_vectors) >= self.vectorlength:
                raise Exception("Cannot create new vector; maximum number of orthogonal vectors already created.")
            vec = np.zeros(self.vectorlength)
            vec[len(self.outcome_vectors)] = 1.0
            self.outcome_vectors[outcome]  = vec
        else:
            self.outcome_vectors[outcome] = self.__generate_vector(self.vectorlength)

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
        present_outcomes = set(outcomes)
        present_cues     = set(cues)

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
        delta         = self.alpha * self.beta * error * magnitude_correction

        # update our cue vectors
        for cue in cues:
            self.cue_vectors[cue] += delta

    def activation(self, cues, outcome):
        """Returns the activation of the outcome, given the cues.
        """
        # remove duplicates
        cues        = set(cues)

        # collapse cue vectors
        cue_vecs    = [ self.cue_vectors[cue] for cue in cues ]
        cue_vec     = np.sum(cue_vecs, axis=0, dtype=np.float64)
        outcome_vec = self.outcome_vectors[outcome]

        activation = (cue_vec * np.sign(outcome_vec)).sum()
        lamda      = np.abs(outcome_vec).sum()
        return activation / lamda

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
