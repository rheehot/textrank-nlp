# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import sys
import os

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

import argparse
import json
import logging
import re

from typing import (List, Tuple, Dict, Union)
from src.keyword.data.token import get_token, find_stem_answer
from tqdm import tqdm

logger = logging.getLogger()


def load_data(output_path: str,
              src_fields=None,
              trg_fields=None) -> List[Tuple[str, str, List[str]]]:
    if src_fields is None:
        src_fields = ['title', 'abstract']

    if trg_fields is None:
        trg_fields = ['keyword']

    data = []

    with open(output_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f)):
            json_object = json.loads(line)

            title = ' '.join([json_object[src_fields[0]]])
            abstract = ' '.join([json_object[src_fields[1]]])
            keyword = [(re.split(';', json_object[f])) for f in trg_fields][0]

            # data.append({
            #     'title':title,
            #     'abstract':abstract,
            #     'keyword':keyword
            # })

            data.append((title, abstract, keyword))

        f.close()

    return data


def data2feature(data,
                 max_src_seq_len: int,
                 max_trg_seq_len: int,
                 valid_check: bool=True,
                 lower: bool=True) -> List[Dict]:
    preprocessed_features = []
    null_ids, absent_ids = 0, 0

    for idx, (title, src, trgs) in tqdm(enumerate(data)):
        src_filter_flag = False
        src_tokens = get_token(src)

        # max_seq_len 을 충족하지 못하면 넘긴다.
        if len(src_tokens) > max_src_seq_len:
            src_filter_flag = True

        if valid_check and src_filter_flag:
            continue

        trgs_tokens = []

        for trg in trgs:
            trg_filter_flag = False
            trg = trg.lower()

            # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
            trg = re.sub(r'\(.*?\)', '', trg)
            trg = re.sub(r'\[.*?\]', '', trg)
            trg = re.sub(r'\{.*?\}', '', trg)

            # FILTER 2: ingore all the phrases that contains strange punctuations, very DIRTY data!
            puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', trg)

            trg_tokens = get_token(trg)

            if len(puncts) > 0:
                continue

            if len(trg_tokens) > max_trg_seq_len:
                trg_filter_flag = True

            if valid_check and trg_filter_flag:
                continue

            if valid_check and (len(trg_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d',
                                                                 trg_tokens[0].strip())) or (
                    len(trg_tokens) > 1 and re.match(r'\d\d\w\d\d', trg_tokens[1].strip())):
                continue

            trgs_tokens.append(trg_tokens)

        if valid_check and len(trgs_tokens) == 0:
            continue

        if lower:
            src_tokens = [token.lower() for token in src_tokens]

        present_phrases = find_stem_answer(word_list=src_tokens, ans_list=trgs_tokens)

        if present_phrases is None:
            null_ids += 1
            continue

        if len(present_phrases['keyphrases']) != len(trgs_tokens):
            absent_ids += 1

        feature = {
            'doc_words': ' '.join(src_tokens),
            'keyphrases': ' __;__ '.join([' '.join(phrase) for phrase in present_phrases['keyphrases']])
        }

        preprocessed_features.append(feature)

    logger.info('Null : number = {} '.format(null_ids))
    logger.info('Absent : number = {} '.format(absent_ids))

    print(preprocessed_features)

    return preprocessed_features

def save_preprocess_data(data_list, filename) -> None:
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    logger.info("Success save file to %s \n" % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dataset', type=str, required=True,
                        help="The path to the source dataset (raw json).")
    parser.add_argument('--preprocess_path', type=str, required=True,
                        help="The path to save preprocess data")
    parser.add_argument('--max_src_seq_len', type=int, default=256,
                        help="Maximum document sequence length")
    parser.add_argument('--max_trg_seq_len', type=int, default=6,
                        help="Maximum keyphrases sequence length to keep.")

    args = parser.parse_args()


    features = data2feature(
        # load_data('../rsc/kp20k/kp20k_validation.json'),
        load_data(args.source_dataset),
        args.max_src_seq_len,
        args.max_trg_seq_len,
        True,
        True)

    # if 'train' not in args.ground_truth_path:
    #     save_ground_truths(features, args.ground_truth_path, 'keyphrases')
    #
    save_preprocess_data(features, args.preprocess_path)
    # data2feature(load_data('../rsc/kp20k/kp20k_training.json'), 256, 10, True, True)
