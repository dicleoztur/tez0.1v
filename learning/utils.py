'''
Created on Feb 24, 2014

@author: dicle
'''

import re


from txtprocessor import listutils


# combNO_F_groupNO-featNO -> [csubjtfidf, tabscount,texcl,..]
def decode_combcode(combcode, combitemsmap):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(combcode)
    
    #combitems_name_map = listutils.initialize_dict(sorted(combitemsmap.keys()), val=[])
    combitems_name_map = {}
    for k in sorted(combitemsmap.keys()):
        combitems_name_map[k] = []
    
    for indexedpair in groupnofeatnopairs:
        pairliteral = indexedpair[1:]
        pairitems = pairliteral.split("-")
        pairitems = map(lambda x : int(x), pairitems)
        groupno = pairitems[0]
        featno = pairitems[1]
        
        groupname = featgroup_name_byindex(combitemsmap, groupno)
        featname = combitemsmap[groupname][featno]
        featname = featname.__class__.__name__
        
        combitems_name_map[groupname].append(featname)
        
    
    return combitems_name_map
    #return "".join(groupnofeatnopairs), groupnofeatnopairs


def encode_combname(combname):
    return



# featsetname = _combNO_F_groupNO-featNO_NC-nclasses
def parse_featsetname(featsetname):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(featsetname)
    combcode = "".join(groupnofeatnopairs)
    
    nclasses = featsetname.split('_NC-')[-1]
    
    return combcode, nclasses



def featgroup_name_byindex(featmap, ind):
    l = sorted(featmap.keys())
    return l[ind]

def featgroup_index_byname(featmap, name):
    l = sorted(featmap.keys())
    try:
        return l.index(name)
    except:
        return -1


def print_features(featmap, matrix):
    print "features"
    _, ncols = matrix.shape
    
    for i in range(ncols):
        groupname = featgroup_name_byindex(featmap, i)
        print groupname," : ",
        groupvect = matrix[:, i]
        for featno in groupvect:
            if featno > -1:
                print featmap[groupname][featno],", ",
            else:
                print "group excluded"
        print





if __name__ == "__main__":
    
    s = "comb0_F_0-0_1-0_2-0_3-0_4-0"  #_5-0_6-0_7-0_8-0"
    combitemsmap = {"c_subj" : ["tfidf", "binary"],
                    "t_abs" : ["count", "tfidf"],
                    "c_adj" : ["count", "ratio"],
                    "t_excl" : ["excl"],
                    "t_adv" : ["ratio"]
                    }
    
    print decode_combcode(s, combitemsmap)
    
    
    