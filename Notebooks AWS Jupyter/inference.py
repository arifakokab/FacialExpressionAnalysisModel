
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

def model_fn(model_dir):
    model = torch.load(f"{model_dir}/mobV2_full.pth", map_location="cpu")
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    raise Exception("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        probs = F.softmax(output, dim=1)
    return probs

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return prediction.numpy().tolist()
    raise Exception("Unsupported content type: {}".format(content_type))
