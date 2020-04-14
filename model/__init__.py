from .base_model import BaseModel
from .ganimation import GANimationModel
from .stargan import StarGANModel


def create_model(opt): # 创建模型
    if opt.model == "ganimation":
        instance = GANimationModel()
    elif opt.model == "stargan":
        instance = StarGANModel()
    else:
        instance = BaseModel()
    instance._initialize(opt)
    instance.setup()
    return instance
