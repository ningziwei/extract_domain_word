import time
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

def read_zh_words(word_file):
    f = open(word_file, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()
    lines = [line.strip() for line in lines]
    words = ' '.join(lines).split(' ')
    return set(words)

def extract_word(wv_from_text, dom_words):
    key_to_index = wv_from_text.key_to_index
    
    my_word_list = {}
    my_vector_list = []
    for key in key_to_index:
        if len(key) == 1:
            my_word_list[key] = 1
            my_vector_list.append(wv_from_text[key])
    
    wv_from_text.init_sims(replace=True)
    for word in tqdm(dom_words):
        if len(word)<2: continue
        if word in key_to_index:
            vec = wv_from_text[word]
            sim_list = wv_from_text.most_similar(positive=[vec], topn=30)
            sim_list = [s for s in sim_list if s[1]>0.7]
            for key, _ in sim_list:
                if key not in my_word_list:
                    my_word_list[key] = 1
                    my_vector_list.append(wv_from_text[key])

    dim = wv_from_text.vector_size
    domain_wv = KeyedVectors(dim)
    my_word_list = list(my_word_list.keys())
    domain_wv.index_to_key = my_word_list
    domain_wv.key_to_index = {key: idx for idx, key in enumerate(my_word_list)}
    domain_wv.vectors = np.array(my_vector_list)

    return domain_wv

def extract_and_save_word(word_vec_file, dom_file, save_file, no_header=False):
    print('loading embedding vec...')
    tic = time()
    wv_from_text = KeyedVectors.load_word2vec_format(
        word_vec_file, binary=False, no_header=no_header)
    toc = time()
    print('loaded, cost {:.2f}s'.format(toc-tic))
    dom_words = read_zh_words(dom_file)
    print('word num:', len(dom_words))
    domain_wv = extract_word(wv_from_text, dom_words)
    domain_wv.save_word2vec_format(save_file)


if __name__=='__main__':
    word_vec_file = '/data1/nzw/Pretrain/tencent-ailab-embedding-zh-d200-v0.2.0.txt'
    # word_vec_file = '/data1/nzw/Pretrain/yangjie_word_char_mix.txt'
    dom_file = '/data2/nzw/BGY/data/建筑领域分词语料_segment_split.txt'
    save_file = '/data1/nzw/Pretrain/tencent-arch-zh-d200-v0.2.0.txt'
    extract_and_save_word(
        word_vec_file, dom_file, save_file, no_header='_mix.txt' in word_vec_file)

