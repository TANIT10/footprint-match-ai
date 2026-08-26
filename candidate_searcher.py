from pathlib import Path

from post_matcher import PostMatcher


BASE_DIR = Path("candidate_search_test")

MISSING_POST_DIR = BASE_DIR / "missing_post"
CANDIDATES_DIR = BASE_DIR / "sighted_candidates"


class CandidateSearcher:

    def __init__(self):
        print()
        print("========================================")
        print("닮은 발자국 후보 검색 AI 준비")
        print("========================================")

        self.post_matcher = PostMatcher()

    def get_candidate_folders(self):
        if not CANDIDATES_DIR.exists():
            raise FileNotFoundError(
                f"후보 폴더를 찾을 수 없습니다: {CANDIDATES_DIR}"
            )

        candidate_folders = sorted(
            [
                folder
                for folder in CANDIDATES_DIR.iterdir()
                if folder.is_dir()
            ]
        )

        if len(candidate_folders) == 0:
            raise ValueError(
                "봤어요 후보 게시글 폴더가 없습니다."
            )

        return candidate_folders

    def search(self):
        candidate_folders = self.get_candidate_folders()

        print()
        print("========================================")
        print("🐾 닮은 발자국 후보 검색 시작")
        print("========================================")

        print(
            f"찾아요 게시글: {MISSING_POST_DIR.name}"
        )

        print(
            f"비교할 봤어요 게시글 수: "
            f"{len(candidate_folders)}개"
        )

        search_results = []

        for index, candidate_folder in enumerate(
            candidate_folders,
            start=1
        ):
            print()
            print("########################################")
            print(
                f"후보 {index}/{len(candidate_folders)} "
                f": {candidate_folder.name}"
            )
            print("########################################")

            result = self.post_matcher.compare_posts(
                missing_folder=str(MISSING_POST_DIR),
                sighted_folder=str(candidate_folder)
            )

            search_results.append(
                {
                    "candidate": candidate_folder.name,
                    "combined_score": result["combined_score"],
                    "highest_score": result["highest_score"],
                    "top_average": result["top_average"],
                    "all_average": result["all_average"]
                }
            )

        ranked_results = sorted(
            search_results,
            key=lambda item: item["combined_score"],
            reverse=True
        )

        print()
        print("========================================")
        print("🐾 닮은 발자국")
        print("발자국이 비슷한 발견 제보를 모아봤어요.")
        print("========================================")

        for rank, result in enumerate(
            ranked_results,
            start=1
        ):
            print()
            print(
                f"{rank}위 - {result['candidate']}"
            )

            print(
                f"게시글 종합 점수: "
                f"{result['combined_score']:.2f}"
            )

            print(
                f"최고 유사도: "
                f"{result['highest_score']:.2f}"
            )

            print(
                f"상위 3개 평균: "
                f"{result['top_average']:.2f}"
            )

            print(
                f"전체 조합 평균: "
                f"{result['all_average']:.2f}"
            )

        print()
        print("========================================")
        print("후보 검색 완료")
        print("========================================")

        print()
        print(
            "※ 현재는 HIGH / 일반 후보 / 제외 "
            "기준을 아직 적용하지 않습니다."
        )

        print(
            "※ 실제 데이터 검증 후 "
            "푸시 알림 기준을 결정합니다."
        )

        print(
            "※ 점수는 같은 동물일 확률이 아니라 "
            "이미지 특징 유사도 기반 점수입니다."
        )

        return ranked_results


if __name__ == "__main__":
    searcher = CandidateSearcher()

    results = searcher.search()