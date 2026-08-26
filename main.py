from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, UploadFile

from post_matcher import PostMatcher


app = FastAPI(
    title="Footprint Match AI",
    description="발자국 실종동물 이미지 매칭 AI API",
    version="1.0.0"
)


post_matcher = PostMatcher()


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


@app.get("/")
def root():
    return {
        "service": "Footprint Match AI",
        "status": "running",
        "message": "발자국 실종동물 매칭 AI 서버가 정상 실행 중입니다."
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/compare-posts")
async def compare_posts(
    missing_images: Annotated[
        list[UploadFile],
        File(description="찾아요 게시글 이미지")
    ],
    sighted_images: Annotated[
        list[UploadFile],
        File(description="봤어요 게시글 이미지")
    ]
):
    if not missing_images:
        return {
            "success": False,
            "message": "찾아요 게시글 사진이 필요합니다."
        }

    if not sighted_images:
        return {
            "success": False,
            "message": "봤어요 게시글 사진이 필요합니다."
        }

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        missing_dir = temp_path / "missing"
        sighted_dir = temp_path / "sighted"

        missing_dir.mkdir()
        sighted_dir.mkdir()

        for index, upload_file in enumerate(
            missing_images,
            start=1
        ):
            filename = upload_file.filename or ""

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                return {
                    "success": False,
                    "message": (
                        "지원하지 않는 이미지 형식입니다: "
                        f"{filename}"
                    )
                }

            save_path = (
                missing_dir
                / f"missing_{index}{extension}"
            )

            contents = await upload_file.read()

            with open(save_path, "wb") as file:
                file.write(contents)

        for index, upload_file in enumerate(
            sighted_images,
            start=1
        ):
            filename = upload_file.filename or ""

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                return {
                    "success": False,
                    "message": (
                        "지원하지 않는 이미지 형식입니다: "
                        f"{filename}"
                    )
                }

            save_path = (
                sighted_dir
                / f"sighted_{index}{extension}"
            )

            contents = await upload_file.read()

            with open(save_path, "wb") as file:
                file.write(contents)

        result = post_matcher.compare_posts(
            missing_folder=str(missing_dir),
            sighted_folder=str(sighted_dir)
        )

        return {
            "success": True,
            "missing_image_count": len(missing_images),
            "sighted_image_count": len(sighted_images),

            "highest_score": round(
                result["highest_score"],
                2
            ),

            "top_average": round(
                result["top_average"],
                2
            ),

            "all_average": round(
                result["all_average"],
                2
            ),

            "combined_score": round(
                result["combined_score"],
                2
            ),

            "message": (
                "점수는 같은 동물일 확률이 아니라 "
                "이미지 특징 유사도 기반 점수입니다."
            )
        }