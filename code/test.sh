#time=$(date "+%Y-%m-%d-%H:%M:%S")
time=$(date "+%Y-%m-%d-%H-%M-%S")
# process data
#python data.py --mode valid --logfile "${time}.log"

# itemcf recall
python recall_itemcf.py --mode valid --logfile "${time}.log"

# binetwork recall
python recall_binetwork.py --mode valid --logfile "${time}.log"

# w2v recall
python recall_w2v.py --mode valid --logfile "${time}.log"

# hot recall
python recall_hot.py --mode valid --logfile "${time}.log"

# merge recall
python recall.py --mode valid --logfile "${time}.log"

# rank feature
python rank_feature.py --mode valid --logfile "${time}.log"

# lgb train
python rank_lgb.py --mode valid --logfile "${time}.log"
