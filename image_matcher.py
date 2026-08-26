from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class ImageMatcher:
    def __init__(self):
        weights = models.ResNet50_Weights.DEFAULT

        self.model = models.resnet50(weights=weights)

        # 마지막 분류층 제거
        self.model.fc = torch.nn.Identity()

        self.model.eval()

        self.transform = weights.transforms()

    def extract_feature(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            feature = self.model(image_tensor)

        # 벡터 정규화
        feature = F.normalize(feature, p=2, dim=1)

        return feature

    def compare_images(self, image_path1: str, image_path2: str):
        feature1 = self.extract_feature(image_path1)
        feature2 = self.extract_feature(image_path2)

        similarity = F.cosine_similarity(
            feature1,
            feature2
        ).item()

        return similarity


if __name__ == "__main__":
    matcher = ImageMatcher()

    print("ImageMatcher 준비 완료")