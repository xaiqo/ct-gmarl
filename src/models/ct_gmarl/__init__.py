from src.models.ct_gmarl.agent import CTGMARLAgent
from src.models.factory import ModelFactory

ModelFactory.register('ct_gmarl', CTGMARLAgent)
