from deepspeed.launcher import runner as ds_runer
from settings.config import Config
import sys

conf = Config.from_file('settings/hyperparameters.json')
print(conf.to_json())

sys.argv.extend(['train.py', '--deepspeed_config', 'settings/ds_config.json'])
conf.extend_argv()
setting_argv = sys.argv.copy()

for run_num in range(conf.exps):
    print("running")
    sys.argv.clear()
    sys.argv.extend(setting_argv)
    sys.argv.extend(["--run_num", str(run_num + 1)])
    ds_runer.main()