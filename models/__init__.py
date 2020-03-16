import importlib

def create_model(config):
    lib = importlib.import_module('models.{}'.format(config.model))
    model = lib.Model(config)
    
    return model
