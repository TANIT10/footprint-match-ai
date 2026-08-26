from pathlib import Path
from statistics import mean

from image_matcher import ImageMatcher
from enhanced_matcher import EnhancedImageMatcher
from dinov2_matcher import DINOv2Matcher


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


def run_validation(
    model_name,
    matcher,
    reference_image,
    same_images,
    different_images
):
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

    overlap = max(
        0,
        different_max - same_min
    )

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

    print(f"평균 분리도: {separation:.2f}")

    if same_min > different_max:
        print(
            "구간 분리 상태: 완전 분리"
        )
        print(
            f"분리 여유: "
            f"{same_min - different_max:.2f}"
        )
    else:
        print(
            "구간 분리 상태: 겹침 발생"
        )
        print(
            f"겹침 크기: {overlap:.2f}"
        )

    return {
        "model_name": model_name,
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
        "같은 동물 사진이 최소 2장 필요합니다."
    )


if len(different_images) < 1:
    raise ValueError(
        "validation_images/different 폴더에는 "
        "다른 동물 사진이 최소 1장 필요합니다."
    )


reference_image = same_images[0]


print()
print("========================================")
print("발자국 AI 모델 3종 비교 시작")
print("========================================")
print(f"기준 사진: {reference_image.name}")
print(f"같은 동물 비교 사진: {len(same_images) - 1}장")
print(f"다른 동물 비교 사진: {len(different_images)}장")


print()
print("1/3 기존 ResNet50 준비 중...")

basic_matcher = ImageMatcher()

basic_result = run_validation(
    "기존 ResNet50",
    basic_matcher,
    reference_image,
    same_images,
    different_images
)


print()
print("2/3 개선형 ResNet50 준비 중...")

enhanced_matcher = EnhancedImageMatcher()

enhanced_result = run_validation(
    "개선형 ResNet50",
    enhanced_matcher,
    reference_image,
    same_images,
    different_images
)


print()
print("3/3 DINOv2 준비 중...")

dinov2_matcher = DINOv2Matcher()

dinov2_result = run_validation(
    "DINOv2",
    dinov2_matcher,
    reference_image,
    same_images,
    different_images
)


results = [
    basic_result,
    enhanced_result,
    dinov2_result
]


print()
print("========================================")
print("최종 비교표")
print("========================================")

for result in results:
    print()
    print(result["model_name"])
    print(
        f"  같은 동물 평균 : "
        f"{result['same_average']:.2f}"
    )
    print(
        f"  다른 동물 평균 : "
        f"{result['different_average']:.2f}"
    )
    print(
        f"  평균 분리도     : "
        f"{result['separation']:.2f}"
    )
    print(
        f"  구간 겹침       : "
        f"{result['overlap']:.2f}"
    )


best_separation = max(
    results,
    key=lambda result: result["separation"]
)

best_overlap = min(
    results,
    key=lambda result: result["overlap"]
)


print()
print("========================================")
print("자동 분석")
print("========================================")

print(
    "평균 분리도가 가장 큰 모델: "
    f"{best_separation['model_name']}"
)

print(
    "같은/다른 동물 겹침이 가장 작은 모델: "
    f"{best_overlap['model_name']}"
)

print()
print(
    "※ 자동 분석 결과만으로 최종 모델을 "
    "확정하지 않습니다."
)

print(
    "※ 실제 개별 점수와 어려운 사진의 결과까지 "
    "확인한 뒤 최종 모델을 선택합니다."
)

print(
    "※ 현재 점수는 같은 동물일 확률이 아니라 "
    "이미지 특징 유사도입니다."
)