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
def test_lewm_forward():
    model = LeWorldModel()
    o = torch.randn(2, 3+1, 3, 224, 224)
    a = torch.randn(2, 3+1, 10)
    out = model(o, a)
    assert set(out) == {"loss", "pred_loss", "sigreg_loss"}, out.keys()
    for k, v in out.items():
        assert v.dim() == 0, f"{k} is not scalar: {v.shape}"
        assert torch.isfinite(v), f"{k} is {v}"
    assert out["loss"].requires_grad

@report_success
def test_lewm_backward():
    model = LeWorldModel()
    o = torch.randn(2, 3+1, 3, 224, 224)
    a = torch.randn(2, 3+1, 10)
    out = model(o, a)
    out["loss"].backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead[:5]}"

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