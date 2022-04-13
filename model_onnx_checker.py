import onnx
import onnxruntime
import torch
import numpy as np

MODEL_PATH = './data/data_from_kaggle/StanfordDogs.onnx'


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()

    else:
        return tensor.cpu().numpy()


onnx_model = onnx.load(MODEL_PATH)
onnx.checker.check_model(onnx_model)
x = torch.load('./data/data_from_kaggle/x.pt',
               map_location=torch.device('cpu'))
torch_out = torch.load('./data/data_from_kaggle/torch_out.pt',
                       map_location=torch.device('cpu'))
ort_session = onnxruntime.InferenceSession(MODEL_PATH)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
np.testing.assert_allclose(to_numpy(torch_out),
                           ort_outs[0],
                           rtol=1e-03,
                           atol=1e-05)
print('OK!')
