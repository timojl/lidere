import os
from os.path import join, realpath, dirname


class LogWriter(object):

    def __getattr__(self, k):
        return print

log = LogWriter()


_PATHS = {
    'DATA_REPO_ROOT': '~/dataset_repository',  # not necessary when S3 is used
    'FEATURE_CACHE_ROOT': realpath(join(dirname(__file__), '..', 'feature_cache')),
}

if 'LOCAL_TMPDIR' in os.environ:
    _PATHS['DATA_ROOT'] = os.environ['LOCAL_TMPDIR']
elif 'SLURM_JOB_ID' in os.environ and 'USER' in os.environ:
    _PATHS['DATA_ROOT'] = '/local/jobs/' + os.environ['USER'] + '_' + os.environ['SLURM_JOB_ID']
else:
    _PATHS['DATA_ROOT'] = '~/datasets'



def get_path(name):
    if name not in _PATHS:
        raise KeyError(f'{name} is not a valid path name.')

    if name in os.environ:
        return os.environ[name]
    else:
        return os.path.expanduser(_PATHS[name])
    
def get_dataset_path(dataset_name, *sub_path):
    return join(get_path('DATA_ROOT'), dataset_name, *sub_path)

