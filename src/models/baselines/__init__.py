from src.models.baselines.rmappo import RMAPPOAgent
from src.models.baselines.qmix import QMIXAgent
from src.models.factory import ModelFactory

ModelFactory.register("rmappo", RMAPPOAgent)
ModelFactory.register("qmix", QMIXAgent)
ModelFactory.register("vdn", RMAPPOAgent)
