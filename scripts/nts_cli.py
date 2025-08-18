#!/usr/bin/env python3
"""
국세청 크롤러 CLI

Command Line Interface for NTS (National Tax Service) document crawler.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ingestion.collector.nts_crawler import (
    NTSCrawler, 
    crawl_nts_once, 
    start_nts_daily_monitor, 
    monitor_all_boards
)


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nts_crawler.log')
        ]
    )


async def run_command(args):
    """명령어 실행"""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == "run":
            await run_crawler(args)
        elif args.command == "monitor":
            await monitor_crawler(args)
        elif args.command == "test":
            await test_crawler(args)
        elif args.command == "status":
            show_status(args)
        else:
            print(f"알 수 없는 명령어: {args.command}")
            return False
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        return True
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        return False
    
    return True


async def run_crawler(args):
    """크롤러 1회 실행"""
    print("=" * 60)
    print(f"국세청 크롤러 실행: {args.board}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    if args.board == "all":
        print("모든 게시판 크롤링 시작...")
        results = {}
        
        for board_type in NTSCrawler.NTS_URLS.keys():
            print(f"\n{board_type} 크롤링 중...")
            try:
                result = await crawl_nts_once(
                    board_type, 
                    f"{args.output}/{board_type}"
                )
                results[board_type] = result
                print(f"✓ {board_type}: {result}")
            except Exception as e:
                print(f"✗ {board_type}: 실패 - {e}")
                results[board_type] = {"error": str(e)}
        
        # 총계 출력
        total_found = sum(r.get("total_found", 0) for r in results.values() if isinstance(r, dict) and "error" not in r)
        total_new = sum(r.get("new_documents", 0) for r in results.values() if isinstance(r, dict) and "error" not in r)
        total_downloaded = sum(r.get("successful_downloads", 0) for r in results.values() if isinstance(r, dict) and "error" not in r)
        
        print(f"\n전체 결과:")
        print(f"  발견된 문서: {total_found}개")
        print(f"  새 문서: {total_new}개")
        print(f"  다운로드 성공: {total_downloaded}개")
        
    else:
        result = await crawl_nts_once(args.board, args.output)
        print(f"\n크롤링 결과:")
        print(f"  발견된 문서: {result.get('total_found', 0)}개")
        print(f"  새 문서: {result.get('new_documents', 0)}개")
        print(f"  다운로드 성공: {result.get('successful_downloads', 0)}개")
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"  실행 시간: {duration:.1f}초")


async def monitor_crawler(args):
    """크롤러 지속 모니터링"""
    print("=" * 60)
    print(f"국세청 크롤러 모니터링 시작: {args.board}")
    print(f"간격: {args.interval}시간")
    print("=" * 60)
    print("중단하려면 Ctrl+C를 누르세요")
    
    if args.board == "all":
        await monitor_all_boards(args.output, args.interval)
    else:
        await start_nts_daily_monitor(args.board, args.output, args.interval)


async def test_crawler(args):
    """크롤러 테스트"""
    print("=" * 60)
    print("국세청 크롤러 연결 테스트")
    print("=" * 60)
    
    boards = [args.board] if args.board != "all" else list(NTSCrawler.NTS_URLS.keys())
    
    for board in boards:
        print(f"\n{board} 테스트 중...")
        try:
            crawler = NTSCrawler(board_type=board, output_dir=args.output)
            documents = await crawler.discover_documents()
            
            if documents:
                print(f"✓ 연결 성공 - {len(documents)}개 문서 발견")
                print(f"  최신 문서: {documents[0].title}")
            else:
                print("✗ 문서를 찾을 수 없음")
                
        except Exception as e:
            print(f"✗ 연결 실패: {e}")


def show_status(args):
    """상태 정보 표시"""
    print("=" * 60)
    print("국세청 크롤러 상태")
    print("=" * 60)
    
    # 출력 디렉토리 확인
    output_path = Path(args.output)
    if output_path.exists():
        print(f"출력 디렉토리: {output_path.absolute()}")
        
        # 각 게시판별 상태
        for board_type in NTSCrawler.NTS_URLS.keys():
            board_path = output_path / board_type
            if board_path.exists():
                files = list(board_path.glob("*.pdf"))
                print(f"  {board_type}: {len(files)}개 파일")
            else:
                print(f"  {board_type}: 다운로드 없음")
    else:
        print(f"출력 디렉토리 없음: {output_path}")
    
    # 게시판 URL 정보
    print(f"\n지원 게시판:")
    for board_type, url in NTSCrawler.NTS_URLS.items():
        print(f"  {board_type}: {url}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="국세청 문서 크롤러",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python nts_cli.py run --board tax_law_revision
  python nts_cli.py run --board all
  python nts_cli.py monitor --board all --interval 12
  python nts_cli.py test --board tax_law_revision
  python nts_cli.py status
        """
    )
    
    # 서브커맨드
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # run 명령어
    run_parser = subparsers.add_parser('run', help='1회 크롤링 실행')
    run_parser.add_argument('--board', 
                           choices=['tax_law_revision', 'tax_field_guide', 'tax_guide_book', 'all'],
                           default='tax_law_revision',
                           help='크롤링할 게시판')
    run_parser.add_argument('--output', default='output/nts', help='출력 디렉토리')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    # monitor 명령어  
    monitor_parser = subparsers.add_parser('monitor', help='지속적 모니터링')
    monitor_parser.add_argument('--board',
                               choices=['tax_law_revision', 'tax_field_guide', 'tax_guide_book', 'all'],
                               default='tax_law_revision',
                               help='모니터링할 게시판')
    monitor_parser.add_argument('--output', default='output/nts', help='출력 디렉토리')
    monitor_parser.add_argument('--interval', type=int, default=24, help='체크 간격(시간)')
    monitor_parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    # test 명령어
    test_parser = subparsers.add_parser('test', help='연결 테스트')
    test_parser.add_argument('--board',
                            choices=['tax_law_revision', 'tax_field_guide', 'tax_guide_book', 'all'],
                            default='tax_law_revision',
                            help='테스트할 게시판')
    test_parser.add_argument('--output', default='output/nts', help='출력 디렉토리')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    # status 명령어
    status_parser = subparsers.add_parser('status', help='상태 확인')
    status_parser.add_argument('--output', default='output/nts', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 비동기 실행
    success = asyncio.run(run_command(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 