from stemming.porter2 import stem

class Document:

    def tokenize(self):
        punc = """\\.!?,(){}[]"'"""
        wordarray = []
        for c in self.document.lower().split():
            if stem(c.strip()) not in self.corpus.stopwords:
                wordarray.append(stem(c.strip(punc)))
        return wordarray

    
    def __init__(self, corpus, text, tag, parsevector):
        """
        For each document, we construct a vector.  If the first doc is "Diyang is cool," the vector is <1, 1, 1>.
        If the second doc is "Grouperfish is is cool," the vector is <0, 2, 1, 1>.  The vector grows longer with the number
        of distinct words in the corpus.
        """
        self.tag = tag
        self.corpus = corpus
        self.document = text.strip("()[] }$")
        self.vector = {}
        self.finalvector = []
        self.reducedvector = []
        self.parsevector = []
        wordarray = self.tokenize()
        for word in set(wordarray):
            if word not in self.corpus.worddict:
                self.corpus.worddict[word] = len(self.corpus.worddict)
            self.vector[self.corpus.worddict[word]] = wordarray.count(word)

    def computevector(self):
        self.finalvector = self.vector.values()

class Corpus:

    def __init__(self, stopwords):
        self.worddict = {}
        self.docs = []
        self.stopwords = stopwords

    def add(self, document, tag, parsevector):
        """
        Adds a document to a corpus.
        """
        doc = Document(self, document, tag, parsevector)
        self.docs.append(doc)
        return doc

    def equalize(self):
        length = 0
        for doc in self.docs:
            if len(doc.finalvector) > length:
                length = len(doc.finalvector)
        for doc in self.docs:
            while len(doc.finalvector) < length:
                doc.finalvector.append(0)

    def addparse(self):
        for doc in self.docs:
            for i in range(len(doc.parsevector)):
                doc.reducedvector.append(doc.parsevector[i])
