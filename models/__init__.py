def create_model(config):
    try:
        exec('from .{} import Model'.format(config.model))
    except Exception as ex:
        print('Model [{}] not recognized: '.format(config.model), ex)

    model = Model(config)
    return model
