import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
print('world.device',world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1
start = time.time()
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
max_recall,max_precision,max_ndcg,max_epoch = 0,0,0,0
try:
    for epoch in range(world.TRAIN_epochs):
        #start = time.time()
        if epoch %10 == 0:
            print('epoch',epoch,time.time()-start)
            results,max_recall,max_precision,max_ndcg,max_epoch = Procedure.Test(dataset, Recmodel, epoch, max_recall,max_precision,max_ndcg,max_epoch, w, world.config['multicore'])
        output_information,aver_loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        #if epoch %10 == 0:
            #print('epoch',aver_loss)
        torch.save(Recmodel.state_dict(), weight_file)
    print('max_epoch:',max_epoch)
    print('max_recall:',max_recall)
    print('max_precision:',max_precision)
    print('max_ndcg:',max_ndcg)
finally:
    if world.tensorboard:
        w.close()