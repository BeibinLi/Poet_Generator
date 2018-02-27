import re, json
import hanzi

#%% Global Constant
MIN_LEN = 2 # minimum length for each sentence in the poet

SPECIAL_CHINESE_CHARS = "●◇ＰＯ!！？｡。．:＂《》＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
SPECIAL_CHINESE_CHARS += hanzi.punctuation

SPECIAL_ENGLISH_CHARS = "|\?|\d|\[|\]|\."

#PATTERN_SPECIAL_CHARS = "|".join( SPECIAL_CHINESE_CHARS ) + SPECIAL_ENGLISH_CHARS
PATTERN_SPECIAL_CHARS = "[^" + hanzi.CHARACTERS + "]"

#CHAR2IDX = {}
#IDX2CHAR = {}

MAX_LINE = 20

#%%
def xml2txt( filename =  "100k_poets.xml", save = True, verbose = True ):

    f = open( filename, "r", encoding = "utf-8" )
    
    all_poets = {}   
    all_sentences = []
    
    temp_title = None
    
    for line in f:
        # sanitize the input: remove special characters, whitespace, etc.
        line = re.sub(r';|\&|lt|br|gt|\s|\\n', '', line)  

    
        m1 = re.findall( r'<title>(.*)</title>', line )
        m2 = re.findall( r'<content>(.*)注释(.*)</content>', line )
    
    
        if m1: temp_title = m1[0]
        if m2:
            poet = m2[0][0]
            intro = m2[0][1] # In this program, we just ignore the introduction to the poet
        
            all_poets[ temp_title ] = poet
            
            # Break the poet into small sentences, and add to SENTENCES
            sentences = re.split( PATTERN_SPECIAL_CHARS, poet )
            
            if len( sentences ) > MAX_LINE: continue # there are too many "sentences" in the poet, it might not be a poet.
    
            # Remove short sentences
            sentences = list(filter(lambda x:len(x)>MIN_LEN, sentences ))
       
            all_sentences += sentences           
            
    f.close()
    
    
    unique_chars = set( "".join(all_sentences) ) 
    print( "There are %d sentences with %d unique characters in the dataset" % 
            ( len( all_sentences),  len( unique_chars )  ) )
    

    if save:
        with open( "poets.json", "w", encoding = "utf-8" ) as out:
            json.dump( all_poets, out  )

        with open( "poets.txt", "w", encoding = "utf-8" ) as out:
            for i in all_sentences: out.write( i + "\n" )
        
    
    
    all_characters = sorted( list(unique_chars) )
    
    all_characters.append( "\n" )

    with open( "unique_chars.txt", "w", encoding = "utf-8" ) as out:
        out.write( "".join( all_characters ) )
    
    return all_sentences, all_characters




if __name__ == "__main__":
    all_sentences, all_characters = xml2txt()
