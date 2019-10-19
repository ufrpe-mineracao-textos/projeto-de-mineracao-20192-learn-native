from util import util
import os
import re
import math
import string
import nltk
from nltk.tokenize import RegexpTokenizer

# --- Tirando referências -----


path = r'datasets/'
data_list = os.listdir(path)
prep = util.PrepData(path)

#prep.label_data('text-label.csv')
dataset = prep.get_datasets()
data = dataset[data_list[0]]
scripture = data['Scripture']

sufix_freq = {}
signature = {}
letter_freq = {}
raw_text = ' '.join(scripture).lower()

tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)

tokens = tokenizer.tokenize(raw_text)

SIZE = len(tokens)
STEP = 1/100
percent = 0 
temp = 0

for token in tokens:

    if temp is int(SIZE*STEP):
        percent += 1
        temp = 0
        print('#', end='')

    for l in token:

        try:
            freq = letter_freq.get(l.lower())
            freq += 1
            letter_freq[l.lower()] = freq
        
        except TypeError:
            
            freq = 1
            letter_freq[l.lower()] = freq
    
    for size in range(1, 6):

        if len(token) > size+3:
            sufix = token[-size:]
            
            try:
                freq = sufix_freq.get(sufix)
                freq += 1
                sufix_freq[sufix] = freq
            except TypeError:
                freq = 1
                sufix_freq[sufix] = freq
    temp += 1
                
keys = list(letter_freq.keys())
print(keys[9], letter_freq[keys[0]])

print('Stemming words: ', end='#')

temp = 1


raw_text = ' '.join(tokenizer.tokenize(raw_text))

sufixes = ' '.join(sufix_freq.keys())

sufixes = tokenizer.tokenize(sufixes)

SIZE = len(sufixes)
STEP = 1/100
percent = 0 
temp = 0
for sufix in sufixes:

    if temp is int(SIZE*STEP):
        percent += 1
        temp = 0
        print('#', end='')
        
    try:
        search = re.findall(r'\w+'+str(sufix), raw_text)
    except re.error:
        print("\n Sufixo: ", sufix)
        breakpoint()

    try:
        for word in set(search.group(0)):
            stem = word.lower().replace(sufix, '')
            signature[sufix] = (stem, word)
    except AttributeError:
        pass

    temp += 1

print('Finish!')
keys = list(signature.keys())
print(keys[0], signature[keys[0]])
print(signature[keys[0]])




print()
#prep.align_verses()
#prep.clean_data(get_noise(), True)


