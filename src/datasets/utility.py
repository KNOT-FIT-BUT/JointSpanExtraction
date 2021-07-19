import sys, os, shutil
from urllib import request

TRAIN_V1_URL_SQUAD = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/train-v1.1.json'
DEV_V1_URL_SQUAD = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json'
TRAIN_F_SQUAD = "train-v1.1.json"
VALIDATION_F_SQUAD = "dev-v1.1.json"

TRAIN_V2_URL_SQUAD = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/train-v2.0.json'
DEV_V2_URL_SQUAD = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v2.0.json'
TRAIN_F_SQUAD2 = "train-v2.0.json"
VALIDATION_F_SQUAD2 = "dev-v2.0.json"

JOIN_TOKEN = "█"

TRAIN_F_MRQA_SQUAD = "SQuAD.train.jsonl"
VALIDATION_F_MRQA_SQUAD = "SQuAD.dev.jsonl"

TRAIN_F_MRQA_NQ = "NaturalQuestionsShort.train.jsonl"
VALIDATION_F_MRQA_NQ = "NaturalQuestionsShort.dev.jsonl"

TRAIN_F_MRQA_NEWSQA = "NewsQA.train.jsonl"
VALIDATION_F_MRQA_NEWSQA = "NewsQA.dev.jsonl"

TRAIN_F_MRQA_TRIVIAQA = "TriviaQA-web.train.jsonl"
VALIDATION_F_MRQA_TRIVIAQA = "TriviaQA-web.dev.jsonl"

TRAIN_F_MRQA_SEARCHQA = "SearchQA.train.jsonl"
VALIDATION_F_MRQA_SEARCHQA = "SearchQA.dev.jsonl"


def check_for_download(path, url):
    if not os.path.exists(path):
        os.makedirs(path)
        try:
            download_url(url, TRAIN_V1_URL_SQUAD)
            download_url(url, DEV_V1_URL_SQUAD)
        except BaseException as e:
            sys.stderr.write(f'Download failed, removing directory {path}\n')
            sys.stderr.flush()
            shutil.rmtree(path)
            raise e


def download_url(path, url):
    sys.stderr.write(f'Downloading from {url} into {path}\n')
    sys.stderr.flush()
    request.urlretrieve(url, path)


def windowize(paragraph_len, max_window_len, window_stride):
    l = []
    r = 0, min(max_window_len, paragraph_len)
    l.append(r)
    while not r[1] == paragraph_len:
        s = r[1] - window_stride
        r = s, min(s + max_window_len, paragraph_len)
        l.append(r)
    return l


class SquadJointResult:
    def __init__(self, unique_id, joint_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.joint_logits = joint_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def make_sure_min_length(t, pad_token='<pad>', pad_up_to=5):
    # pad to at least length of shortest filter,
    # (5) in case of bidaf is kept as default
    while (len(t)) < pad_up_to:
        t.append(pad_token)
    return t


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    if sll < 1:
        return results
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results
