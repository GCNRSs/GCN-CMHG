import world
import dataloader
import dataloader_douban
import dataloader_Flixster
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','ml1m_dense', 'Movies_and_TV', 'yelp2018']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    print('123456789')
    dataset = dataloader.LastFM()
elif world.dataset == 'douban':
    dataset = dataloader_douban.ML()
else:
    dataset = dataloader_Flixster.ML()


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}