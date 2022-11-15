import multiprocessing
from datetime import datetime
from multiprocessing import Process
from deepspeed.launcher import runner as ds_runer
from torch.distributed import launch as pt_runner
from settings.config import Config
import sys
import socket
from ipc import resultServer

conf = Config.from_file('settings/hyperparameters.json')
print(conf.to_json())

q = multiprocessing.Queue()
p = Process(target=resultServer, args=[conf, q])
p.start()

sys.argv.extend(['train.py', '--deepspeed_config', 'settings/ds_config.json'])
conf.extend_argv()
setting_argv = sys.argv.copy()

for run_num in range(conf.exps):
    sys.argv.clear()
    sys.argv.extend(setting_argv)
    sys.argv.extend(["--run_num", str(run_num + 1)])
    ds_runer.main()

rfq = q.get()
