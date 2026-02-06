import os
import torch
from torchvision import transforms
from PIL import Image

class Perceptor:

    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def analyze_image(self, path):
        try:
            img = Image.open(path).convert('RGBA')
            img = img.resize((64, 64))
            t_img = transforms.ToTensor()(img)
            rgb = t_img[:3, :, :]
            alpha = t_img[3, :, :]
            total_alpha = torch.sum(alpha)
            if total_alpha > 0:
                weighted_sum = torch.sum(rgb * alpha, dim=[1, 2])
                mean_color = weighted_sum / total_alpha
            else:
                mean_color = torch.tensor([0.5, 0.5, 0.5])
            r, g, b = (mean_color[0].item(), mean_color[1].item(), mean_color[2].item())
            complexity = torch.std(rgb).item()
            return [r, g, b, complexity]
        except Exception as e:
            print(f'Error analyzing image {path}: {e}')
            return [0.0, 0.0, 0.0, 0.0]

    def analyze_text(self, path):
        try:
            with open(path, 'r') as f:
                text = f.read().lower()
            length_score = min(len(text) / 1000.0, 1.0)
            pos_words = ['good', 'happy', 'bright', 'yes', 'love']
            neg_words = ['bad', 'sad', 'dark', 'no', 'hate']
            score = 0.5
            for w in text.split():
                if w in pos_words:
                    score += 0.05
                if w in neg_words:
                    score -= 0.05
            return [max(0, min(1, score)), length_score, 0.0, 0.0]
        except Exception as e:
            print(f'Error analyzing text {path}: {e}')
            return [0.0, 0.0, 0.0, 0.0]

    def scan_folder(self, folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    yield self.analyze_image(path)
                elif file.lower().endswith('.txt'):
                    yield self.analyze_text(path)