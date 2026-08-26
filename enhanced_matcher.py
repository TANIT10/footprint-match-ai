from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class EnhancedImageMatcher:

    def __init__(self):
        weights = models.ResNet50_Weights.DEFAULT

        self.model = models.resnet50(weights=weights)

        # 마지막 이미지 분류층 제거
        # 우리는 "무슨 동물인가"가 아니라
        # 이미지의 특징 벡터가 필요함
        self.model.fc = torch.nn.Identity()

        self.model.eval()

        self.base_transform = weights.transforms()

    def _make_center_crop(self, image: Image.Image):
        """
        사진 가장자리의 배경 영향을 조금 줄이기 위해
        이미지 중앙 80% 영역을 잘라낸다.
        """

        width, height = image.size

        crop_width = int(width * 0.80)
        crop_height = int(height * 0.80)

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2

        right = left + crop_width
        bottom = top + crop_height

        return image.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

    def _extract_single_feature(self, image: Image.Image):
        """
        PIL 이미지 한 장에서 특징 벡터를 추출한다.
        """

        image_tensor = (
            self.base_transform(image)
            .unsqueeze(0)
        )

        with torch.no_grad():
            feature = self.model(image_tensor)

        feature = F.normalize(
            feature,
            p=2,
            dim=1
        )

        return feature

    def extract_feature(self, image_path: str):
        """
        한 사진을 여러 방식으로 바라본 특징을 평균내서
        최종 특징 벡터를 만든다.

        사용 이미지:
        1. 원본
        2. 좌우 반전
        3. 중앙 확대
        4. 중앙 확대 + 좌우 반전
        """

        image = Image.open(image_path).convert("RGB")

        center_crop = self._make_center_crop(image)

        flipped_image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        flipped_crop = center_crop.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        image_versions = [
            image,
            flipped_image,
            center_crop,
            flipped_crop
        ]

        features = []

        for version in image_versions:
            feature = self._extract_single_feature(
                version
            )

            features.append(feature)

        combined_feature = torch.mean(
            torch.stack(features),
            dim=0
        )

        combined_feature = F.normalize(
            combined_feature,
            p=2,
            dim=1
        )

        return combined_feature

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
    matcher = EnhancedImageMatcher()

    print(
        "EnhancedImageMatcher 준비 완료"
    )