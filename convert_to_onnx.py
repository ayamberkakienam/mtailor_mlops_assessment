import torch.onnx
from pytorch_model import BasicBlock, Classifier


if __name__ == "__main__":
    # import pytorch model
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    pytorch_model.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    pytorch_model.eval()

    # create tensor for exporting
    batch_size = 1  # just a random number
    tensor = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # convert and export as onnx
    torch.onnx.export(
        pytorch_model, tensor, "onnx_model.onnx", export_params=True, opset_version=10
    )
