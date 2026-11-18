"""CLI used by the local product to recover a frozen-model loading failure."""
from pathlib import Path
import argparse
import json
from types import SimpleNamespace

from engine.executor import Interpreter
from engine.final_evaluation import retry_legacy_main_finalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--expected-status-sha256', required=True)
    args = parser.parse_args()
    state = json.loads((args.workspace/'final_evaluation/status.json').read_text(encoding='utf-8'))
    journal = json.loads(args.journal.read_text(encoding='utf-8'))
    selected = next(n for n in journal['nodes'] if n['id'] == state['selection']['node_id'])
    node = SimpleNamespace(id=selected['id'], code=selected['code'],
        metric=SimpleNamespace(value=selected['metric']['value']), evaluation_protocol=selected['evaluation_protocol'])
    cfg = SimpleNamespace(workspace_dir=args.workspace, start_cpu_id=0, cpu_number=4,
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)))
    runner = Interpreter(args.workspace, timeout=600, max_parallel_run=1, cfg=cfg)
    try:
        result = retry_legacy_main_finalization(cfg, node, runner, expected_status_sha256=args.expected_status_sha256)
    finally:
        runner.terminate_all_subprocesses()
    print(json.dumps({'status': result['status'], 'node_id': selected['id']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
