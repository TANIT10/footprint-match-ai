from image_matcher import ImageMatcher


matcher = ImageMatcher()

image1 = "test_images/animal1.jpg"
image2 = "test_images/animal2.jpg"

similarity = matcher.compare_images(image1, image2)

print("===== 발자국 이미지 비교 테스트 =====")
print(f"원본 유사도 점수: {similarity:.4f}")
print(f"참고용 표시 점수: {similarity * 100:.2f}")