from pathlib import Path
from statistics import mean

from image_matcher import ImageMatcher
from enhanced_matcher import EnhancedImageMatcher


BASE_DIR = Path("validation_images")
SAME_DIR = BASE_DIR / "same"
DIFFERENT_DIR = BASE_DIR / "different"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def get_image_files(folder: Path):
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def run_validation(model_name, matcher, reference_image, same_images, different_images):
    print()
    print("========================================")
    print(model_name)
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

    same_average = mean(same_scores)
    same_min = min(same_scores)
    same_max = max(same_scores)

    different_average = mean(different_scores)
    different_min = min(different_scores)
    different_max = max(different_scores)

    separation = same_average - different_average
    overlap = different_max - same_min

    print()
    print("[ 요약 ]")
    print(f"같은 동물 평균: {same_average:.2f}")
    print(f"같은 동물 최저: {same_min:.2f}")
    print(f"같은 동물 최고: {same_max:.2f}")
    print()
    print(f"다른 동물 평균: {different_average:.2f}")
    print(f"다른 동물 최저: {different_min:.2f}")
    print(f"다른 동물 최고: {different_max:.2f}")
    print()
    print(f"평균 점수 차이: {separation:.2f}")

    if same_min > different_max:
        print(
            f"구간 분리 상태: 완전 분리 "
            f"(여유 {same_min - different_max:.2f})"
        )
    else:
        print(
            f"구간 분리 상태: 겹침 발생 "
            f"(겹침 {overlap:.2f})"
        )

    return {
        "same_average": same_average,
        "same_min": same_min,
        "same_max": same_max,
        "different_average": different_average,
        "different_min": different_min,
        "different_max": different_max,
        "separation": separation,
        "overlap": overlap
    }


same_images = get_image_files(SAME_DIR)
different_images = get_image_files(DIFFERENT_DIR)

if len(same_images) < 2:
    raise ValueError(
        "validation_images/same 폴더에는 "
        "같은 동물 사진이 최소 2장 이상 필요합니다."
    )

if len(different_images) < 1:
    raise ValueError(
        "validation_images/different 폴더에는 "
        "다른 동물 사진이 최소 1장 이상 필요합니다."
    )

reference_image = same_images[0]

basic_matcher = ImageMatcher()
enhanced_matcher = EnhancedImageMatcher()


basic_result = run_validation(
    "기존 ResNet50 모델",
    basic_matcher,
    reference_image,
    same_images,
    different_images
)

enhanced_result = run_validation(
    "개선형 ResNet50 모델",
    enhanced_matcher,
    reference_image,
    same_images,
    different_images
)


print()
print("========================================")
print("최종 비교")
print("========================================")

print(
    f"기존 모델 평균 분리도: "
    f"{basic_result['separation']:.2f}"
)

print(
    f"개선 모델 평균 분리도: "
    f"{enhanced_result['separation']:.2f}"
)

print()

if enhanced_result["separation"] > basic_result["separation"]:
    print("평균 분리도 기준: 개선 모델 우세")
elif enhanced_result["separation"] < basic_result["separation"]:
    print("평균 분리도 기준: 기존 모델 우세")
else:
    print("평균 분리도 기준: 동일")

print()

if enhanced_result["overlap"] < basic_result["overlap"]:
    print("같은/다른 동물 겹침 기준: 개선 모델 우세")
elif enhanced_result["overlap"] > basic_result["overlap"]:
    print("같은/다른 동물 겹침 기준: 기존 모델 우세")
else:
    print("같은/다른 동물 겹침 기준: 동일")

print()
print(
    "※ 아직 최종 모델이나 알림 임계값을 "
    "확정하는 단계는 아닙니다."
)