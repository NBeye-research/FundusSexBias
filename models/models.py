import torch
import torchvision
import timm

#自建模型，注册函数
custom_models = {}
def register(name):
    def decorator(cls):
        custom_models[name] = cls
        return cls
    return decorator


def make(arch, pretrained = True, n_classes = 1000, img_size = 224, fine_tuning=None):
    if arch is None:
        return None
    torchvisison_list_models = [name for name in dir(torchvision.models) if not name.startswith('_')]
    if arch in timm.list_models():
        model = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)

    elif arch in torchvisison_list_models:
        model = getattr(torchvision.models, arch)(pretrained=pretrained, num_classes=n_classes)
    else:   
        #自建模型，根据注册的key进行关联
        model = custom_models[arch](pretrained=pretrained, num_classes=n_classes, img_size=img_size, fine_tuning=fine_tuning)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])
    model.load_state_dict(model_sv[name + '_sd'])
    # print('load model finish. method {}, temp {}'.format(model.method, model.temp))
    return model

