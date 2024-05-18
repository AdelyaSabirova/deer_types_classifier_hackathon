import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import get_model

class CLASS_p:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted = predicted.item()
            class_names = ['Кабарга', 'Косуля', 'Олень']
            return class_names[predicted], confidence
