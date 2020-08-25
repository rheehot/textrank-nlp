# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

# import logging
import re
import unicodedata
from typing import List, Tuple

from nltk.stem.porter import PorterStemmer
from transformers import BertTokenizer, BertTokenizerFast

stemmer = PorterStemmer()
# logger = logging.getLogger()
digit = '<digit>'


def replace_digit(tokens: List[str]) -> List[str]:
    return [w if not re.match('^\d+$', w) else digit for w in tokens]


def get_token(text: str) -> List[str]:
    text = re.sub(r'[\r\n\t]', ' ', text)

    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)

    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text)))

    # replace digits to special token <digit>
    return replace_digit(tokens)


def norm_phrase_to_char(phrase_list):
    norm_phrases = set()
    for phrase in phrase_list:
        p = " ".join([w.strip() for w in phrase if len(w.strip()) > 0])
        if len(p) < 1: continue
        norm_phrases.add(unicodedata.normalize('NFD', p))

    norm_stem_phrases = []
    for norm_chars in norm_phrases:
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
        norm_stem_phrases.append((norm_chars, stem_chars))

    return norm_stem_phrases


def norm_doc_to_char(word_list):
    norm_char = unicodedata.normalize('NFD', " ".join(word_list))
    stem_char = " ".join([stemmer.stem(w.strip()) for w in norm_char.split(" ")])

    return norm_char, stem_char


def find_stem_answer(word_list, ans_list):
    norm_doc_char, stem_doc_char = norm_doc_to_char(word_list)
    norm_stem_phrase_list = norm_phrase_to_char(ans_list)

    tot_ans_str = []
    tot_start_end_pos = []

    for norm_ans_char, stem_ans_char in norm_stem_phrase_list:

        norm_stem_doc_char = " ".join([norm_doc_char, stem_doc_char])

        if norm_ans_char not in norm_stem_doc_char and stem_ans_char not in norm_stem_doc_char:
            continue
        else:
            norm_doc_words = norm_doc_char.split(" ")
            stem_doc_words = stem_doc_char.split(" ")

            norm_ans_words = norm_ans_char.split(" ")
            stem_ans_words = stem_ans_char.split(" ")

            assert len(norm_doc_words) == len(stem_doc_words)
            assert len(norm_ans_words) == len(stem_ans_words)

            # find postions
            tot_pos = []

            for i in range(0, len(stem_doc_words) - len(stem_ans_words) + 1):

                Flag = False

                if norm_ans_words == norm_doc_words[i:i + len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == norm_doc_words[i:i + len(stem_ans_words)]:
                    Flag = True

                elif norm_ans_words == stem_doc_words[i:i + len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == stem_doc_words[i:i + len(stem_ans_words)]:
                    Flag = True

                if Flag:
                    tot_pos.append([i, i + len(norm_ans_words) - 1])
                    assert (i + len(stem_ans_words) - 1) >= i

            if len(tot_pos) > 0:
                tot_start_end_pos.append(tot_pos)
                tot_ans_str.append(norm_ans_char.split())

    assert len(tot_ans_str) == len(tot_start_end_pos)
    # print(word_list, norm_doc_char.split(" "), sep=', ')
    # assert len(word_list) == len(norm_doc_char.split(" "))

    if len(tot_ans_str) == 0:
        return None
    return {'keyphrases': tot_ans_str, 'start_end_pos': tot_start_end_pos}


def get_bert_tokenizer(model_name: str='bert-base-uncased') -> Tuple[BertTokenizer, int]:
    """
    bert.resize_token_embeddings(orig_num_tokens + num_added_tokens)
    """
    decode_sep = '__;__'
    sos = '<sos>'
    eos = '<eos>'

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': [decode_sep, sos, eos]}

    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer, orig_num_tokens + num_added_tokens