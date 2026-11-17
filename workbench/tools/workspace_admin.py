"""Offline workspace backup/recovery. Stop the workspace service before using writes."""
import argparse
import json
from pathlib import Path
from app.storage import Storage
from app.maintenance import MaintenanceRepository


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('usage')
    backup=sub.add_parser('backup');backup.add_argument('--project',required=True)
    restore=sub.add_parser('restore');restore.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args()
    if not (args.data_dir/'platform.db').is_file():parser.error('Existing platform.db required; do not point at an empty source directory.')
    repo=MaintenanceRepository(Storage(args.data_dir/'platform.db'))
    result=repo.usage() if args.action=='usage' else repo.backup(args.project) if args.action=='backup' else repo.restore(args.archive)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
