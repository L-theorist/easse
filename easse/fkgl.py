from easse.utils.text import to_words, to_sentences, count_sentences, count_words, count_syllables_in_sentence


# https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
class FKGLScorer:
    def __init__(self, version='easse'):
        self.nb_words = 0
        self.nb_syllables = 0
        self.nb_sentences = 0
        self.version = version.lower()
        if self.version == 'easse':
            self.compute_fkgl = self.compute_fkgl_easse
            self.add = self.add_easse
        elif self.version == 'dress':
            self.compute_fkgl = self.compute_fkgl_dress
            self.add = self.add_dress
        else:
            raise NotImplementedError(f'FKGL Version "{version}" not implemented.')

    def add_easse(self, text):
        for sentence in to_sentences(text):
            self.nb_words += count_words(sentence)
            self.nb_syllables += count_syllables_in_sentence(sentence)
            self.nb_sentences += 1

    @staticmethod
    def compute_fkgl_easse(nb_sentences, nb_words, nb_syllables):
        if nb_sentences == 0 or nb_words == 0:
            return 0
        return max(0, 0.39 * (nb_words / nb_sentences) + 11.8 * (nb_syllables / nb_words) - 15.59)

    def add_dress(self, text):
        # The text is not split into multiple sentences, and not tokenized the same,
        # resulting in '\n' to be counted as a word
        #print('Ours')
        #print(repr(text))
        #print(to_words(text, version='dress'))
        self.nb_words += count_words(text, version='dress')
        self.nb_syllables += count_syllables_in_sentence(text, version='dress')
        self.nb_sentences += count_sentences(text)

    @staticmethod
    def compute_fkgl_dress(nb_sentences, nb_words, nb_syllables):
        if nb_sentences == 0 or nb_words == 0:
            return 0
        # There is a bug in DRESS FKGL implementation, where the average number of words per sentence is casted to int dur to the behaviour of int division in python2
        # https://github.com/XingxingZhang/dress/blob/dda0c92a755e2b209c7f5bb41a51bcc4b0962953/dress/scripts/readability/readability.py#L25
        # It can also return negative scores
        print('Ours')
        print(nb_sentences, nb_words, nb_syllables)
        #print(int(self.nb_words / self.nb_sentences))
        #print(0.39 * (self.nb_words / self.nb_sentences))
        #print(11.8 * (self.nb_syllables / self.nb_words) - 15.59)
        return 0.39 * (int(nb_words / nb_sentences)) + 11.8 * (nb_syllables / nb_words) - 15.59

    def score(self):
        # Flesch-Kincaid grade level
        return self.compute_fkgl(self.nb_sentences, self.nb_words, self.nb_syllables)


def get_fkgl(filepath, version='easse'):
    if version == 'easse':
        return get_fkgl_easse(filepath)
    elif version == 'dress':
        return get_fkgl_dress(filepath)
    else:
        NotImplementedError(f'Version {version} not implemented.')

def get_fkgl_easse(filepath):
    scorer = FKGLScorer(version='easse')
    with open(filepath, 'r') as f:
        for line in f:
            scorer.add(line.strip())
    return scorer.score()


def get_fkgl_dress(filepath):
    scorer = FKGLScorer(version='dress')
    with open(filepath, 'r') as f:
        # File is read all at once, this causes weird cases where one line does not end with a final dot.
        # In this case, this line and the following line are considered as a single very long sentence.
        scorer.add(f.read().strip())
    return scorer.score()


def get_sentence_fkgl(sentence, version='easse'):
    scorer = FKGLScorer(version=version)
    scorer.add(sentence)
    return scorer.score()
