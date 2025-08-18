### How to deal with the prompts

About defining the prompts and switching to exact one in runtime

- 규칙:
    - 슬러그: `도메인-역할` 소문자 케밥케이스: 
        - ex) gov-policy, acad-research, industry-analyst, startup-pre, exec-brief, staff-report, student-ug
    - 사람 라벨: UI/로그용 짧은 한글/영문, 예) “정책입안자”, “Executive Brief”
    - 버전: v1, 프롬프트 변경시 v2로 올리고 이전 버전 유지
    - 별칭(Alias): 자연어 라우팅 입력 대비용, 예) “정책”, “공무원”, “policy_maker” → gov-policy
    - 상수화: Enum으로 강제, 중앙 레지스트리로 관리


    