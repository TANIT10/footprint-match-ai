from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


class DINOv2Matcher:

    def __init__(self):
        print("DINOv2 모델을 불러오는 중...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )

        print(
            f"DINOv2 준비 완료 / 사용 장치: {self.device}"
        )

    def extract_feature(self, image_path: str):
        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            feature = self.model(
                image_tensor
            )

        feature = F.normalize(
            feature,
            p=2,
            dim=1
        )

        return feature

    def compare_images(
        self,
        image_path1: str,
        image_path2: str
    ):
        feature1 = self.extract_feature(
            image_path1
        )

        feature2 = self.extract_feature(
            image_path2
        )

        similarity = F.cosine_similarity(
            feature1,
            feature2
        ).item()

        return similarity


if __name__ == "__main__":
    matcher = DINOv2Matcher()

    print(
        "DINOv2Matcher 정상 실행 완료"
    )