# Git 전략 및 커밋 컨벤션

이 문서는 RAG 시스템 프로젝트의 Git 사용 전략과 커밋 메시지 작성 규칙을 정의합니다.

## 개요

팀원들이 일관된 방식으로 Git을 사용하여 효율적인 협업을 할 수 있도록 가이드라인을 제공합니다.

## 브랜치 전략

### 브랜치 구조
```
main                    # 실제 서비스 운영 브랜치
develop                 # 개발 통합 브랜치
feature/기능명          # 새로운 기능 개발 브랜치
hotfix/버그명           # 긴급 오류 수정 브랜치
release/버전명          # 출시 준비 브랜치
```

### 브랜치 명명 규칙
- **feature/**: 새로운 기능 개발
  - 예: `feature/rag-구조-개선`, `feature/PDF-로더-통합`
- **hotfix/**: 긴급 오류 수정
  - 예: `hotfix/벡터-서비스-오류`, `hotfix/로그인-오류`
- **release/**: 출시 준비
  - 예: `release/v1.0.0`, `release/v1.1.0`

## 커밋 메시지 컨벤션

### 기본 형식
```
<유형>(<범위>): <제목>

<본문>

<푸터>
```

### 유형 종류
- **feat**: 새로운 기능 추가
- **fix**: 오류 수정
- **docs**: 문서 수정
- **style**: 코드 포맷팅 (기능 변경 없음)
- **refactor**: 코드 구조 개선
- **test**: 테스트 코드 추가/수정
- **chore**: 빌드 과정, 도구 변경

### 범위 (선택사항)
- **core/**: 핵심 모듈
- **api/**: API 관련
- **services/**: 서비스 레이어
- **utils/**: 유틸리티
- **config/**: 설정 파일

### 커밋 메시지 예시

#### 좋은 예시
```bash
feat(핵심): PDF 로더를 새로운 구조로 통합

- 기존 loaders/에서 core/ingestion/loaders/로 PDF 로더 이동
- 포괄적인 오류 처리 및 타입 힌트 추가
- 개발자 정보가 포함된 적절한 문서화 구현
- main.py 및 테스트 파일의 import 경로 업데이트

개발자: 박민수 (2024-07-10), 우성민 (2024-07-12)
```

#### 나쁜 예시
```bash
PDF 로더 업데이트
수정
fix stuff
```

## 작업 흐름

### 1. 기능 개발 워크플로우
```bash
# 1. develop 브랜치에서 시작
git checkout develop
git pull origin develop

# 2. 기능 브랜치 생성
git checkout -b feature/새로운-기능

# 3. 작업 수행
# ... 코드 작성 ...

# 4. 변경사항 스테이징
git add .

# 5. 커밋
git commit -m "feat(범위): 새로운 기능 설명"

# 6. 원격 저장소에 푸시
git push origin feature/새로운-기능

# 7. Pull Request 생성
# GitHub/GitLab에서 Pull Request 생성
```

### 2. 긴급 수정 워크플로우
```bash
# 1. main 브랜치에서 hotfix 브랜치 생성
git checkout main
git checkout -b hotfix/긴급-수정

# 2. 수정 작업
# ... 오류 수정 ...

# 3. 커밋 및 푸시
git add .
git commit -m "fix(범위): 긴급 오류 수정"
git push origin hotfix/긴급-수정

# 4. Pull Request 생성 (main과 develop 모두에 병합)
```

## Pull Request 규칙

### Pull Request 제목
```
[유형] 간단한 설명
예: [feat] PDF 로더 구조 개선
```

### Pull Request 설명 템플릿
```markdown
## 변경사항
- [변경사항 1]
- [변경사항 2]

## 테스트 방법
1. [테스트 단계 1]
2. [테스트 단계 2]

## 관련 이슈
- 관련 이슈 번호 또는 링크

## 체크리스트
- [ ] 코드가 정상 작동하는지 확인
- [ ] 테스트가 통과하는지 확인
- [ ] 문서가 업데이트되었는지 확인
- [ ] 불필요한 파일이 포함되지 않았는지 확인
```

## 코드 리뷰 가이드

### 리뷰어 체크리스트
- [ ] 코드가 요구사항을 만족하는가?
- [ ] 코드가 읽기 쉽고 이해하기 쉬운가?
- [ ] 적절한 오류 처리가 있는가?
- [ ] 테스트가 충분한가?
- [ ] 문서가 업데이트되었는가?

### 리뷰어 피드백 규칙
- 건설적인 피드백 제공
- 구체적인 개선 제안
- 긍정적인 부분도 언급
- 개인적 공격 금지

## 커밋 전 체크리스트

### 필수 확인 사항
- [ ] 코드가 정상 작동하는지 확인
- [ ] 테스트가 통과하는지 확인
- [ ] 불필요한 파일이 포함되지 않았는지 확인
- [ ] 커밋 메시지가 컨벤션을 따르는지 확인
- [ ] 문서가 업데이트되었는지 확인

### 선택 확인 사항
- [ ] 코드 리뷰를 받았는지 확인
- [ ] 성능에 영향을 주는 변경사항인지 확인
- [ ] 보안 관련 변경사항인지 확인

## 자주 사용하는 Git 명령어

### 기본 명령어
```bash
# 브랜치 확인
git branch

# 브랜치 생성 및 이동
git checkout -b feature/새로운-기능

# 변경사항 확인
git status

# 변경사항 스테이징
git add .

# 커밋
git commit -m "feat(범위): 설명"

# 푸시
git push origin 브랜치명
```

### 고급 명령어
```bash
# 커밋 히스토리 확인
git log --oneline

# 특정 파일의 변경사항 확인
git diff 파일명

# 브랜치 병합
git merge 브랜치명

# 브랜치 삭제
git branch -d 브랜치명
```

## 문제 해결

### 자주 발생하는 문제

#### 1. 커밋 메시지 수정
```bash
# 마지막 커밋 메시지 수정
git commit --amend -m "새로운 커밋 메시지"
```

#### 2. 잘못된 브랜치에서 작업한 경우
```bash
# 현재 변경사항을 올바른 브랜치로 이동
git stash
git checkout 올바른-브랜치
git stash pop
```

#### 3. 충돌 해결
```bash
# 충돌 파일 수정 후
git add .
git commit -m "fix: 충돌 해결"
```

## 참고 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**마지막 업데이트**: 2024-07-12  
**작성자**: 우성민  
**문서 버전**: v1.0 