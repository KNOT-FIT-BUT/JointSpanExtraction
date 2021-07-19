import string
from typing import List, Tuple

import spacy
import logging
from spacy.tokenizer import Tokenizer


def create_custom_tokenizer(nlp):
    custom_prefixes = [r'[0-9]+', r'\~', r'\–', r'\—', r'\$']
    custom_infixes = [r'[!&:,()]', r'\.', r'\-', r'\–', r'\—', r'\$']
    custom_suffixes = [r'\.', r'\–', r'\—', r'\$']
    default_prefixes = list(nlp.Defaults.prefixes) + custom_prefixes
    default_prefixes.remove(r'US\$')
    default_prefixes.remove(r'C\$')
    default_prefixes.remove(r'A\$')
    all_prefixes_re = spacy.util.compile_prefix_regex(tuple(default_prefixes))

    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))

    suffix_re = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + custom_suffixes))

    rules = dict(nlp.Defaults.tokenizer_exceptions)
    for c in range(ord("a"), ord("z") + 1):
        if f"{chr(c)}." in rules:
            rules.pop(f"{chr(c)}.")

    return Tokenizer(nlp.vocab, rules,
                     prefix_search=all_prefixes_re.search,
                     infix_finditer=infix_re.finditer, suffix_search=suffix_re.search,
                     token_match=None)


_spacy_en = spacy.load('en_core_web_sm')
# _spacy_en_vanilla = spacy.load('en_core_web_sm')
_spacy_en.tokenizer = create_custom_tokenizer(_spacy_en)



def tokenize(text: string, tokenizer=_spacy_en):
    tokens = [tok for tok in _spacy_en.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def tokenize_and_join(text: string, jointoken="█"):
    return jointoken.join(tokenize(text)[1])


def tokenize_spacy_wordpiece(subwordtokenizer, text):
    spacy_tokens, _ = tokenize(text)
    token_positions = []
    final_tokenized = []
    for token in spacy_tokens:
        subtokens = subwordtokenizer.tokenize(token.text)
        token_accumulated_offset = 0
        final_tokenized += subtokens
        # I found no way to keep track of UNK tokens :(
        if subwordtokenizer.unk_token in subtokens:
            for st in subtokens:
                token_positions.append((token.idx, token.idx + len(token.text)))
            continue

        for st in subtokens:
            start_ix = token.idx + token_accumulated_offset
            subtoken_len = len(st.replace('##', ''))
            end_ix = start_ix + subtoken_len
            token_positions.append([start_ix, end_ix])
            token_accumulated_offset += subtoken_len
        if not (token_positions[-1][1] == token.idx + len(token.text)):
            # Unfortunately BERT tokenizer throws away unicode symbols
            # we cannot correct keep track on words with unicode subtokens
            token_positions[-1][1] = token.idx + len(token.text)
    return final_tokenized, token_positions


# Borrowed from AllenNLP
# https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/util.py#L47
def char_span_to_token_span(token_offsets: List[Tuple[int, int]],
                            character_span: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).
    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.
    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.
    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether the token spans match the input character spans exactly.  If this is ``False``, it
        means there was an error in either the tokenization or the annotated character span.
    """
    # We have token offsets into the passage from the tokenizer; we _should_ be able to just find
    # the tokens that have the same offsets as our span.
    error = False
    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
        start_index += 1

    if start_index >= len(token_offsets):
        error = True
        return token_offsets[-1], error

    # start_index should now be pointing at the span start index.
    if token_offsets[start_index][0] > character_span[0]:
        # In this case, a tokenization or labeling issue made us go too far - the character span
        # we're looking for actually starts in the previous token.  We'll back up one.
        logging.debug("Bad labelling or tokenization - start offset doesn't match")
        start_index -= 1
    if token_offsets[start_index][0] != character_span[0]:
        error = True
    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
        end_index += 1
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logging.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logging.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    s = "Hello world."
    # split $US
    c = "The effects of the \"Twin Shocks\"—the Soviet entry and the atomic bombing—were profound. On 10 August the \"sacred decision\" was made by Japanese Cabinet to accept the Potsdam terms on one condition: the \"prerogative of His Majesty as a Sovereign Ruler\". At noon on 15 August, after the American government's intentionally ambiguous reply, stating that the \"authority\" of the emperor \"shall be subject to the Supreme Commander of the Allied Powers\", the Emperor broadcast to the nation and to the world at large the rescript of surrender, ending the Second World War."
    c2 = "The b. 1485a. more \"fit\"— US$ College of ~Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively."
    logging.info(tokenize(c))
    # p = nlp(c)
    # r = []
    # for sentence in p.sents:
    #     for w in sentence:
    #         r.append(w.text)
    # logging.info(r)
    # logging.info(tokenize(s, tokenizer=_spacy_en_vanilla))
