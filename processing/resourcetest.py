'''
Created on Aug 6, 2012

@author: dicle
'''

from NewsResource import NewsResource




name = "radikal"
rootlink_item = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID="
rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE="


#item
markerTitle1 = '<title>'
markerTitle2 = '</title>'

markerText1 = '<div id="metin2" class="fck_li">'
markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'


#id
idlimit1 = "<div class=\"cat-news\"><ol";
idlimit2 = "var Geri = 'Geri'";

pattern1 = r";ArticleID=[0-9]{6,9}"
pattern2 = r'[0-9]{6,9}'


r1 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, markerTitle1, markerTitle2, markerText1, markerText2)

r1.toscreen()
