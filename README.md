
# The source code of the "Graph neural networks and deep reinforcement learning in job-shop scheduling"

**Edit:** Organised the configuration files for better reproducibility of results. Added only configurations from 
the final steps of the experiments!

To perform any of the experiments you must define the configuration file following examples in `configuration/experiments/**/emperiment.yml` files.
Then to execute the experiment just call `cli.py`, i.e.

```
python cli.py --configuration PATH_TO_EXPERIMENT_FILE
```

The results of the tournament runs are stored in the pickled version of the class from `environment/statistics.py`.
To load the file use the `load` method of Statistics class.

```
from environment.statistics import Statistics

statistics = Statistics.load(PATH_TO_PICKLED_FILE)
```

The requirements of the work are present in `requirements.txt`.

Trained models and experiment results are available [here](https://campuscvut-my.sharepoint.com/:u:/g/personal/hayeuyur_cvut_cz/EQ7TgHnCjbVIvDmWBltOo5ABQH5YcKSm6CRa0k33InaY8A?e=ul41Pa). Additionally, in `evaluation` folder you can find
the results of the tournaments of the trained models.

If you want to launch any of the experiments from the evaluation archive, then you must follow the next procedure

1.  Copy the configuration folder to the 'diploma_thesis/configuration/experiments/jsp' folder
2.  Update reference paths in either 'experiment.yml' or 'experiment/0.yml' files from the copied folder. For instance,
    in the following yml slice, the path ''configuration/experiments/jsp/BEST/experiments/1/mr_machine.yml' must be updated to
    'configuration/experiments/jsp/YOUR_FOLDER/experiments/1/mr_machine.yml'

```yml
dqn_2: &dqn_2
  base_path: 'configuration/experiments/jsp/BEST/experiments/1/mr_machine.yml'
  template: '../../../../../../mods/machine/model/marl_dqn/baseline'
  mod_dirs:
    - 'configuration/mods/machine/mods'
  mods:
    - *default_mods
```

