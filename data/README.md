# data/

이 폴더에 **`dataset.zip`** 을 넣어주세요. (GitHub 100MB 제한으로 레포에 포함되지 않습니다.)

- 다운로드 링크: **<여기에 OneDrive/Google Drive 공유 링크>**
- 최종 경로: `data/dataset.zip`

`dataset.zip` 안에는 착장 이미지(`dataset/images/*`)와 태그 시트(`outfits.xlsx`)가 들어 있으며,
앱은 압축을 풀지 않고 zip에서 직접 읽습니다.

> 임베딩 캐시(`../embeddings/`)는 레포에 포함되어 있어, dataset.zip이 없어도
> ②벡터 임베딩·③추천 과정 단계는 동작합니다. 다만 ④결과의 **착장 이미지 표시**에는
> dataset.zip 이 필요합니다.
