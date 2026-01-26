import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Identity()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "resnet_feature_extractor.pth"
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image).cpu()
