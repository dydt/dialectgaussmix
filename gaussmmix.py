# Written using http://lasa.epfl.ch/sourcecode/ 's implementation of GMM/GMR

import types
import random
import math
import numpy
import numpy.matlib
import tokenizer

class Document():

    def __init__(self, vector, tag):
        self.vector = vector 
# unlike in onlinekmeans, vector is only an array
        self.prob = [] # array of size k
        self.group = None
        self.tag = tag

        
class Mixture():

    def __init__(self):
        self.docs = [] # array of documents?
        self.clusters = [] # array of Groups
        self.priors = [] # array of probabilities

    def add(self, vector, tag):
        doc = Document(vector, tag)
        self.docs.append(doc)

    def initializeclusters(self, k, data):
        """
        Initializes clusters, assigns docs to clusters, and computes the prior.

        parameter 'data' is an array of Documents
        """
        for i in range(k):
            c = Cluster(i)
            c.mean = data[random.randint(0, len(data)-1)].vector
            self.clusters.append(c)
        for doc in data:
            dist = []
            for i in range(k):
                dist.append(self.distance(doc.vector, self.clusters[i].mean))
            mindist = min(dist)
            j = dist.index(mindist)
            doc.group = self.clusters[j]
            self.clusters[j].add(doc)
# Prior is [number in group 1, number in group 2, ... group k] normalized
        prior = [0]*k
        for i in range(k):
            for doc in data:
                if doc.group == self.clusters[i]:
                    prior[i] += 1
        for i in range(len(prior)):
            prior[i]/float(sum(prior))
        self.priors = prior
# Set clusters.sigma
        for i in range(k):
            self.clusters[i].sigma = self.covariance(self.clusters[i].docs)


    def distance(self, vectora, vectorb):
        distance = 0
# Assume vectora and vectorb are same size.
        for i in range(len(vectora)):
            distance += pow(vectora[i] - vectorb[i], 2)
        return distance
        

    def covariance(self, data):
        """
        Calculates the variance matrix of the data.

        data here needs to be a matrix with each doc.vector a row
        TODO: transform the data into the matrix here for consistency?
        """
        vectors = []
        for doc in data:
            vectors.append(doc.vector)
        datmatrix = numpy.matrix(vectors)
        sigma = numpy.matrix(numpy.cov(numpy.concatenate((datmatrix, datmatrix), 0).H))
        for i in range(numpy.shape(sigma)[0]):
            sigma[i, i] += 0.001

        return sigma


    def gausspdfmat(self, datamatrix, mu, sigma):
        """
        datamatrix = matrix with rows being doc.vectors
        Computes the PDF of a datamatrix.T given mu, sigma.
        """
        normmatrix = datamatrix - numpy.matrix(numpy.tile(mu, (len(self.docs), 1)))
        norm = numpy.sum(numpy.multiply(normmatrix*numpy.linalg.inv(sigma), normmatrix), 1)
        sz = numpy.shape(norm)
#         for i in range(sz[0]):
#             for j in range(sz[1]):
#                 if norm[i, j] > 10:
#                     norm[i, j] = 10 
        prob = numpy.exp(-0.5*norm)/ math.sqrt(pow(2*math.pi, datamatrix.shape[1]) * (numpy.linalg.det(sigma) + 0.001))
        return prob

    def finalcluster(self):
        for cluster in self.clusters:
            cluster.clear()
        for doc in self.docs:
            doc.prob = numpy.asarray(doc.prob)[0].tolist()
            print(doc.prob)
            i = doc.prob.index(max(doc.prob))
            self.clusters[i].add(doc)

    def expectmax(self, data, priors):
        """
        EM algorithm for GMM.
        """
        vectormatrix = numpy.matrix([doc.vector for doc in data])
        loglike_threshold = .01
        loglike_old = -100000000
        steps = 0
        datapdfmat = numpy.matlib.zeros((len(data), len(self.clusters)))

        while True:

            # E step
            # Calculate the probability of data point given a cluster
            for i in range(len(self.clusters)):
                datapdfmat[:, i] = self.gausspdfmat(vectormatrix, self.clusters[i].mean, self.clusters[i].sigma) # N x K
            for j in range(len(data)):
                data[j].prob = numpy.asarray(datapdfmat[j, :])
# Calculate the probability of posterior: p(mu, sigma | datum)
# P(mu, sigma | x ) = Prior(mu, sigma) * P(x | mu, sigma)
            posteriors_temp = numpy.multiply(numpy.matrix(numpy.tile(priors, (len(data), 1))),datapdfmat)
            posteriors = posteriors_temp/posteriors_temp.sum(axis=0)
# Calculate cumulative posterior
            cumpost = posteriors.sum(axis = 0)
# M step
# Update the priors
            self.priors = [i/float(len(data)) for i in numpy.asarray(cumpost)]
# Update the centers
            for i in range(len(self.clusters)):
                self.clusters[i].mean = [numpy.ravel(vectormatrix.T*posteriors[:,i])]
                self.clusters[i].mean = self.clusters[i].mean/numpy.array(numpy.asarray(cumpost)[0].tolist()[i])
# Update the covariance matrices
            for i in range(len(self.clusters)):
                datatemp = vectormatrix - numpy.matrix(numpy.tile(self.clusters[i].mean, (len(data), 1)))
                covtemp = numpy.matlib.zeros(numpy.shape(self.clusters[i].sigma))
                for j in range(len(data)):
                    temp = (vectormatrix.T)[:, j] - self.clusters[i].mean.T
                    covtemp = covtemp + numpy.multiply(temp*temp.T, posteriors[j, i])
                self.clusters[i].sigma = covtemp/numpy.array(numpy.asarray(cumpost)[0].tolist()[i])
#                self.clusters[i].sigma = numpy.multiply(numpy.tile(posteriors[:,i].T, (numpy.shape(vectormatrix)[1], 1)), datatemp.T*datatemp)/numpy.array(numpy.asarray(cumpost)[0].tolist()[i])
                for j in range(numpy.shape(self.clusters[i].sigma)[0]):
                    self.clusters[i].sigma[j, j] += 0.001                
# Add small variance for numerical stability
#                self.clusters[i].sigma += numpy.mat(numpy.tile([0.00001], (len(data), len(data))))
# Stopping criterion
# Compute new p(x|mu, sigma)
            #datapdfmat = numpy.matrix(datapdfmat).T # each pdf a column
            for i in range(len(self.clusters)):
                datapdfmat[:,i] = self.gausspdfmat(vectormatrix, self.clusters[i].mean, self.clusters[i].sigma)
            for j in range(len(data)):
                data[j].prob = numpy.asarray(datapdfmat[j, :])
# Compute the log likelihood
            F = posteriors * numpy.mat(priors).T # returns N x 1 matrix
            for j in range(len(self.clusters)):
                if F[j] < 0.000001:
                    F[j] = 0.000001
                F[j] = math.log(F[j])
            logrealmin = numpy.array(F.T)
            loglike = numpy.average(logrealmin)
# Stop process depending on log likelihood
            if abs(loglike/loglike_old) - 1 < loglike_threshold:
                break
            loglike_old = loglike
            steps += 1
            print(steps)
 
    
class Cluster():

    def __init__(self, integer):
        self.docs = []
        self.id = integer
        self.mean = [] # vector
        self.sigma = [] # covariance matrix
        self.tags = []

    def add(self, document):
        """
        Adds a document to a particular cluster.
        """
        self.docs.append(document)
        
    def clear(self):
        self.docs = []
