from pathlib import Path
from statistics import mean

from dinov2_matcher import DINOv2Matcher


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


class PostMatcher:

    def __init__(self):
        print("게시글 매칭 AI를 준비하는 중...")

        self.matcher = DINOv2Matcher()

        print("게시글 매칭 AI 준비 완료")

    def get_image_files(self, folder_path: str):
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"폴더를 찾을 수 없습니다: {folder}"
            )

        images = sorted(
            [
                file
                for file in folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

        if len(images) == 0:
            raise ValueError(
                f"이미지가 없습니다: {folder}"
            )

        return images

    def compare_posts(
        self,
        missing_folder: str,
        sighted_folder: str
    ):
        missing_images = self.get_image_files(
            missing_folder
        )

        sighted_images = self.get_image_files(
            sighted_folder
        )

        print()
        print("========================================")
        print("찾아요 ↔ 봤어요 게시글 비교")
        print("========================================")

        print(
            f"찾아요 사진 수: {len(missing_images)}장"
        )

        print(
            f"봤어요 사진 수: {len(sighted_images)}장"
        )

        total_comparisons = (
            len(missing_images)
            * len(sighted_images)
        )

        print(
            f"총 비교 조합: {total_comparisons}개"
        )

        print()

        comparison_results = []

        for missing_image in missing_images:
            for sighted_image in sighted_images:

                similarity = self.matcher.compare_images(
                    str(missing_image),
                    str(sighted_image)
                )

                score = similarity * 100

                comparison_results.append(
                    {
                        "missing_image": missing_image.name,
                        "sighted_image": sighted_image.name,
                        "score": score
                    }
                )

                print(
                    f"{missing_image.name} ↔ "
                    f"{sighted_image.name} : "
                    f"{score:.2f}"
                )

        scores = [
            result["score"]
            for result in comparison_results
        ]

        sorted_results = sorted(
            comparison_results,
            key=lambda result: result["score"],
            reverse=True
        )

        highest_score = sorted_results[0]["score"]

        top_count = min(
            3,
            len(sorted_results)
        )

        top_scores = [
            result["score"]
            for result in sorted_results[:top_count]
        ]

        top_average = mean(top_scores)
        all_average = mean(scores)

        # 현재는 임계값을 정하지 않는다.
        # 여러 사진을 비교했을 때 어떤 값들이 나오는지
        # 확인하기 위한 임시 게시글 종합 점수다.
        #
        # 최고점 하나에만 의존하지 않도록
        # 최고점과 상위 3개 평균을 함께 사용한다.
        combined_score = (
            highest_score * 0.4
            + top_average * 0.6
        )

        print()
        print("========================================")
        print("게시글 매칭 결과")
        print("========================================")

        print(
            f"최고 유사도: {highest_score:.2f}"
        )

        print(
            f"상위 {top_count}개 평균: "
            f"{top_average:.2f}"
        )

        print(
            f"전체 조합 평균: {all_average:.2f}"
        )

        print(
            f"게시글 종합 점수: {combined_score:.2f}"
        )

        print()
        print("[ 가장 비슷했던 사진 조합 ]")

        for index, result in enumerate(
            sorted_results[:top_count],
            start=1
        ):
            print(
                f"{index}. "
                f"{result['missing_image']} ↔ "
                f"{result['sighted_image']} "
                f": {result['score']:.2f}"
            )

        print()
        print(
            "※ 게시글 종합 점수는 같은 동물일 "
            "확률이 아닙니다."
        )

        print(
            "※ HIGH / 후보 표시 / 제외 기준은 "
            "추후 실제 데이터 검증 후 정합니다."
        )

        return {
            "highest_score": highest_score,
            "top_average": top_average,
            "all_average": all_average,
            "combined_score": combined_score,
            "comparisons": sorted_results
        }


if __name__ == "__main__":

    post_matcher = PostMatcher()

    result = post_matcher.compare_posts(
        missing_folder=(
            "post_match_test/missing_post"
        ),
        sighted_folder=(
            "post_match_test/sighted_post"
        )
    )