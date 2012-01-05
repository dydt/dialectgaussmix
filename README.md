My Machine Learning (6.867 Fall '11) project was done with Alyssa Mensch.  We wanted to differentiate written American English and British English.  We did so by first sampling sections from books (scraped from Project Gutenberg) and news articles.  Next, from those sections the features we chose were word frequencies and grammatical structure, and we represented those in vector form.  We then ran PCA to shorten the extremely long vectors, and then used the results in a Gaussian mixure model.  The same process should be able to differentiate any n dialects, and might even be able to differentiate written vs spoken or media (book vs article vs tweet).  Effectiveness untested

Here is some code that I wrote that stands independently of the code Alyssa wrote.  It generates word frequency vectors and creates a Gaussian mixture model.  The GMM functionality is based off the GMM/GMR implementation found at http://lasa.epfl.ch/sourcecode/.  Like most project code done close to the deadline, this code is messy.

I hope to have more information put up here, and perhaps later make this code unmessy.