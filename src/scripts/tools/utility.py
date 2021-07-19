import inspect
import random
import unicodedata

import numpy as np
import torch
import os
import logging
import logging.config
import datetime
import socket
import yaml
from functools import wraps


def disable_logging(func):
    def disable_logging_decorator(*args, **kwargs):
        logger = logging.getLogger()
        logger.disabled = True
        result = func(*args, **kwargs)
        logger.disabled = False
        return result

    return disable_logging_decorator


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


class DotDict(dict):
    """
    A dictionary with dot notation for key indexing
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

    # These are needed when python pickle-ing is done
    # for example, when object is passed to another process
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sum_parameters(model):
    return sum(p.view(-1).sum() for p in model.parameters() if p.requires_grad)


def autoinit(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @autoinit
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    Implementation taken from https://stackoverflow.com/a/1389216/2174310
    """
    names, varargs, keywords, defaults, kwonlyargs, kwonlydefaults, ann = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])
        func(self, *args, **kargs)

    return wrapper


def report_parameters(model):
    num_pars = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    num_sizes = {name: p.shape for name, p in model.named_parameters() if p.requires_grad}
    return num_pars, num_sizes


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def xavier_param_init(model):
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')


def logsumexp(inputs, dim=None, keepdim=False, weights=None):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        weights: A set of weights to multiply exponentiated quantities with
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    if weights is not None:
        outputs = s + ((inputs - s).exp() * weights).sum(dim=dim, keepdim=True).log()
    else:
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def mkdir(s):
    if not os.path.exists(s):
        os.makedirs(s)


def plot_rank_histogram(h, figpath=".visualisations/topk_plot.png"):
    import matplotlib.pyplot as plt
    plt.xlabel("K")
    plt.ylabel("topK-HIT")
    plt.plot(h, c='green')
    if figpath is not None:
        plt.savefig(figpath)
    plt.show()


def get_device(t):
    return t.get_device() if t.get_device() > -1 else torch.device("cpu")


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        extra_name="",
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, stamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, filename=os.path.join(logpath, stamp))


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
