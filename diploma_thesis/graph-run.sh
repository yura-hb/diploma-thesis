

python cli.py --configuration 'configuration/experiments/jsp/GRAPH-NN/experiments/2/0/experiment.yml' &

python cli.py --configuration 'configuration/experiments/jsp/GRAPH-NN/experiments/2/1/experiment.yml' &

wait
#
python cli.py --configuration 'configuration/experiments/jsp/GRAPH-NN/experiments/2/2/experiment.yml' &
#
python cli.py --configuration 'configuration/experiments/jsp/GRAPH-NN/experiments/2/3/experiment.yml' &

python cli.py --configuration 'configuration/experiments/jsp/GRAPH-NN/experiments/2/4/experiment.yml' &

wait