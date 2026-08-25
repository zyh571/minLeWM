import torch
import functools 

from model.lewm import LeWorldModel
from model.encoder import ViT
from model.predictor import AdaLNTransformer
from model.decoder import DecoderTransformer

def report_success(func):
    @functools.wraps(func)
    def wrapper():
        func()
        print(f"{func.__name__} passed")
    return wrapper

@report_success
def test_lewm_shape():
    o = torch.randn(2, 3+1, 3, 224, 224)
    a = torch.randn(2, 3+1, 10)
    model = LeWorldModel()
    z_pred, z_targ, z = model(o, a)
    #print(z_pred.shape, z_targ.shape, z.shape)
    assert z_pred.shape == (2, 3, 192), "lewm (z_pred) shape mismatch"
    assert z_targ.shape == (2, 3, 192), "lewm (z_targ) shape mismatch"
    assert z.shape == (2, 4, 192), "lewm (z) shape mismatch"

@report_success
def test_encoder_shape():
    x = torch.randn(2, 3, 224, 224)
    vit = ViT()
    assert vit(x).shape == (2, 257, 192), "encoder shape mismatch"

@report_success
def test_predictor_shape():
    x = torch.randn(2, 3, 192)
    a = torch.randn(2, 3, 10)
    predictor = AdaLNTransformer()
    assert predictor(x, a).shape == (2, 3, 192), "predictor shape mismatch"

@report_success 
def test_decoder_shape():
    cls = torch.randn(2, 192)
    decoder = DecoderTransformer()
    assert decoder(cls).shape == (2, 3, 224, 224), "decoder shape mismatch"

if __name__ == '__main__':
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()