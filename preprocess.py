import argparse
import collections
import logging
import os
import re
import pickle

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    src_dict = build_dictionary([args.train_prefix + '.' + args.source_lang])
    tgt_dict = build_dictionary([args.train_prefix + '.' + args.target_lang])

    src_dict.finalize(threshold=args.threshold_src, num_words=args.num_words_src)
    src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
    logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    tgt_dict.finalize(threshold=args.threshold_tgt, num_words=args.num_words_tgt)
    tgt_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
    logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    def make_split_datasets(lang, dictionary):
        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang), dictionary)
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '.' + lang, os.path.join(args.dest_dir, 'valid.' + lang), dictionary)
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + '.' + lang, os.path.join(args.dest_dir, 'test.' + lang), dictionary)

    make_split_datasets(args.source_lang, src_dict)
    make_split_datasets(args.target_lang, tgt_dict)


def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()
    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    main(args)
