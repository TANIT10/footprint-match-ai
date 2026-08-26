from pathlib import Path
from statistics import mean

from image_matcher import ImageMatcher


BASE_DIR = Path("validation_images")
SAME_DIR = BASE_DIR / "same"
DIFFERENT_DIR = BASE_DIR / "different"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_files(folder: Path):
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


matcher = ImageMatcher()

same_images = get_image_files(SAME_DIR)
different_images = get_image_files(DIFFERENT_DIR)

if len(same_images) < 2:
    raise ValueError(
        "same 폴더에는 같은 동물 사진이 최소 2장 이상 필요합니다."
    )

if len(different_images) < 1:
    raise ValueError(
        "different 폴더에는 다른 동물 사진이 최소 1장 이상 필요합니다."
    )

reference_image = same_images[0]

print()
print("========================================")
print("발자국 이미지 매칭 검증")
print("========================================")
print(f"기준 사진: {reference_image.name}")
print()


same_scores = []

print("[ 같은 동물 비교 ]")

for image_path in same_images[1:]:
    similarity = matcher.compare_images(
        str(reference_image),
        str(image_path)
    )

    score = similarity * 100
    same_scores.append(score)

    print(
        f"{reference_image.name} ↔ "
        f"{image_path.name} : "
        f"{score:.2f}"
    )


different_scores = []

print()
print("[ 다른 동물 비교 ]")

for image_path in different_images:
    similarity = matcher.compare_images(
        str(reference_image),
        str(image_path)
    )

    score = similarity * 100
    different_scores.append(score)

    print(
        f"{reference_image.name} ↔ "
        f"{image_path.name} : "
        f"{score:.2f}"
    )


print()
print("========================================")
print("검증 결과 요약")
print("========================================")

print(
    f"같은 동물 평균: {mean(same_scores):.2f}"
)

print(
    f"같은 동물 최저: {min(same_scores):.2f}"
)

print(
    f"같은 동물 최고: {max(same_scores):.2f}"
)

print()

print(
    f"다른 동물 평균: {mean(different_scores):.2f}"
)

print(
    f"다른 동물 최저: {min(different_scores):.2f}"
)

print(
    f"다른 동물 최고: {max(different_scores):.2f}"
)

print()
print(
    "※ 이 점수는 같은 동물일 확률이 아니라 "
    "이미지 특징 유사도입니다."
)
print(
    "※ 실제 알림 기준은 더 많은 데이터를 테스트한 뒤 정합니다."
)