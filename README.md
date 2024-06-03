
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
