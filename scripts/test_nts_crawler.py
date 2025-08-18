#!/usr/bin/env python3
"""
국세청 크롤러 테스트 스크립트

수정된 NTS 크롤러의 동작을 테스트합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ingestion.collector.nts_crawler import NTSCrawler, crawl_nts_once
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 변경
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nts_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Playwright 로그도 활성화
playwright_logger = logging.getLogger('playwright')
playwright_logger.setLevel(logging.INFO)


async def test_nts_crawler():
    """NTS 크롤러 테스트"""
    print("=" * 60)
    print("국세청 크롤러 테스트 시작")
    print("=" * 60)
    
    try:
        # 개정세법해설 게시판 테스트
        print("\n1. 개정세법해설 게시판 테스트")
        print("-" * 40)
        
        crawler = NTSCrawler(board_type="tax_law_revision", output_dir="test_output/nts")
        
        # 문서 발견 테스트
        documents = await crawler.discover_documents()
        print(f"발견된 문서 수: {len(documents)}")
        
        if documents:
            print("\n최신 문서 5개:")
            for i, doc in enumerate(documents[:5]):
                print(f"  {i+1}. {doc.title}")
                print(f"     URL: {doc.url}")
                print(f"     파일: {doc.file_path or '없음'}")
                print(f"     날짜: {doc.last_modified}")
                print()
        
        # 다운로드 테스트 (첫 번째 문서만)
        if documents and documents[0].file_path:
            print("2. 다운로드 테스트")
            print("-" * 40)
            
            result = await crawler.download_contents_file(documents[0])
            print(f"다운로드 결과: {result}")
        
        print("테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        return False
    
    return True


async def test_all_boards():
    """모든 게시판 테스트"""
    print("=" * 60)
    print("모든 국세청 게시판 테스트")
    print("=" * 60)
    
    boards = ["tax_law_revision", "tax_field_guide", "tax_guide_book"]
    
    for board in boards:
        try:
            print(f"\n{board} 게시판 테스트")
            print("-" * 40)
            
            result = await crawl_nts_once(board, f"test_output/nts/{board}")
            print(f"결과: {result}")
            
        except Exception as e:
            logger.error(f"{board} 테스트 실패: {e}")


async def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        await test_all_boards()
    else:
        await test_nts_crawler()


if __name__ == "__main__":
    asyncio.run(main()) 