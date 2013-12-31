'''
Created on May 4, 2013

@author: dicle
'''


def read_N_papers(resourcename, N=200):
    
    return



def build_docfeatmatrix():
    return




# features

def get_POStagfreq():
    for fileid in fileids:
        path = newspath + os.sep + fileid
        words, date = texter.getnewsitem(path)
        doclemmas = SAKsParser.lemmatize_lexicon(words)
        for (_, literalPOS, root, _) in doclemmas:
            dailyroots.append((date, (root.decode('utf-8'),)))    # decoding'e dikkat
            dailypostags.append((date, literalPOS))
            '''if literalPOS == "ADJ":
                print literalPOS
                dailyadjectives.append((date, (literalPOS,)))
                
                