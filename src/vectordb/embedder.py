import os 

from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class Embedder:
  """
  텍스트를 숫자 벡터로 변환하는 임베딩 클래스.

  '임베딩'이란 텍스트의 의미를 숫자 배열(벡터)로 표현하는 기술입니다.
  의미가 비슷한 텍스트는 벡터 공간에서 가까운 위치에 놓이게 되므로,
  유사한 문서를 검색하거나 비교할 때 활용됩니다.

  사용 예:
    embedder = Embedder()
    vector = embedder.embed_question("전세 보증금은 얼마인가요?")
  """

  def __init__(self, model_name: str = _EMBEDDING_MODEL) -> None:
    """
    Embedder를 초기화합니다.

    모델을 처음 불러올 때 인터넷에서 다운로드가 발생할 수 있습니다.
    이후 실행부터는 로컬 캐시를 사용하므로 빠릅니다.

    Args:
      model_name: 사용할 임베딩 모델 이름.
                  기본값은 환경변수 EMBEDDING_MODEL에서 읽어옵니다.
                  예: "paraphrase-multilingual-MiniLM-L12-v2"
    """
    self._model = SentenceTransformer(model_name)
    self.vector_size = self._model.get_sentence_embedding_dimension()

  def embed(self, texts: list[str]) -> list[list[float]]:
    """
    여러 텍스트를 한꺼번에 벡터로 변환합니다.

    문서를 대량으로 벡터DB에 저장할 때 사용합니다.
    한 번에 여러 텍스트를 처리하므로 반복 호출보다 효율적입니다.

    Args:
      texts: 변환할 텍스트 목록.
             예: ["임대차 계약이란...", "보증금 반환 기한은..."]

    Returns:
      각 텍스트에 대응하는 벡터 목록.
      반환 형태: [[0.12, -0.34, ...], [0.56, 0.78, ...], ...]
      리스트의 순서는 입력 texts의 순서와 동일합니다.
    """
    vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()

  def embed_question(self, text: str) -> list[float]:
    """
    단일 텍스트를 벡터로 변환합니다.

    사용자가 입력한 질문을 검색에 사용할 벡터로 바꿀 때 주로 사용합니다.
    내부적으로 embed()를 호출하며, 텍스트 한 개만 처리합니다.

    Args:
      text: 변환할 텍스트 한 개.
            예: "집주인이 보증금을 돌려주지 않으면 어떻게 하나요?"

    Returns:
      텍스트를 표현하는 float 벡터 (1차원 리스트).
      예: [0.12, -0.34, 0.78, ...]
    """
    return self.embed([text])[0]