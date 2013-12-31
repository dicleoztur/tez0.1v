'''
Created on Sep 16, 2012

@author: dicle
'''

import nltk.collocations
import nltk.corpus
import collections

bgm    = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(
    nltk.corpus.brown.words())
scored = finder.score_ngrams( bgm.likelihood_ratio  )



# Group bigrams by first word in bigram.                                        
prefix_keys = collections.defaultdict(list)
for key, scores in scored:
   prefix_keys[key[0]].append((key[1], scores))

# Sort keyed bigrams by strongest association.                                  
for key in prefix_keys:
   prefix_keys[key].sort(key = lambda x: -x[1])

print 'doctor', prefix_keys['doctor'][:5]
print 'baseball', prefix_keys['baseball'][:5]
print 'happy', prefix_keys['happy'][:5]

for k,v in prefix_keys.iteritems():
    print k," ",v
''' output:
dauntless   [('as', 10.308547655867347)]
Lust   [(',', 5.9820507395876934)]
northerly   [('American', 15.243912013779944)]
Spikes   [(',', 5.9820507395876934)]
hardboiled   [("''", 19.535257316824854)]
jawbone   [(',', 5.9820507395876934)]
'''
     
    