'''
Created on Feb 24, 2014

@author: dicle
'''

import re



# combNO_F_groupNO-featNO -> [csubjtfidf, tabscount,texcl,..]
def decode_combcode(combcode):
    return
def encode_combname(combname):
    return

# featsetname = _combNO_F_groupNO-featNO_NC-nclasses
def parse_featsetname(featsetname):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(featsetname)
    combcode = "".join(groupnofeatnopairs)
    
    nclasses = featsetname.split('_NC-')[-1]
    
    return combcode, nclasses