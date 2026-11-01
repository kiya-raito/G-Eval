"""Run real Ollama integration checks using this repository's Notebook code.

Prerequisite: Notebook setup cells (dependency installation, Ollama install/start,
model pull and structured-output preflight) have already succeeded.
No model downloads or mock responses are performed by this script.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import tempfile

os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'G-Eval_sample_format.ipynb')
    parser.add_argument('--sample', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'samples/sample.tsv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model', default='llama3.2')
    args = parser.parse_args()
    target = args.output_dir or Path(tempfile.mkdtemp(prefix='geval-live-'))
    target.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(args.notebook.read_text(encoding='utf-8'))
    ns = {'MODEL_NAME': args.model, 'OLLAMA_BASE_URL': 'http://127.0.0.1:11434'}

    def cell(title):
        return next(''.join(c['source']) for c in notebook['cells']
                    if ''.join(c['source']).startswith(title))

    def definitions(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, (
            ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, '<notebook definitions>', 'exec'), ns)

    exec(cell('#@title DeepEvalの通信'), ns)
    exec(cell('#@title DeepEvalにOllama'), ns)
    exec(cell('#@title 評価メトリック'), ns)
    definitions(cell('#@title 評価を実行して'))
    columns = ['input_column', 'actual_output_column', 'expected_output_column']
    import ollama
    from importlib.metadata import version
    client = ollama.Client(host=ns['OLLAMA_BASE_URL'], timeout=120, trust_env=False)
    client.show(args.model)
    report = {'model': args.model, 'output_dir': str(target),
              'versions': {name: version(name) for name in (
                  'deepeval', 'ollama', 'pandas', 'httpx', 'pydantic')}, 'checks': {}}
    for mode in ('fixed', 'criteria'):
        metric_source = cell('#@title 評価メトリック').replace(
            "evaluation_mode = 'fixed'", f"evaluation_mode = '{mode}'")
        exec(metric_source, ns)
        result = ns['evaluate_tsv'](args.sample, target / f'{mode}.scored.tsv',
                                    ns['correctness_metric'], columns)
        statuses = result.Status.tolist()
        assert statuses == ['success', 'success', 'invalid_input'], (mode, statuses)
        if mode == 'fixed':
            assert result.Score.iloc[0] > result.Score.iloc[1], result.Score.tolist()
        report['checks'][mode] = result.fillna('').to_dict(orient='records')
        report['checks'][mode + '_steps'] = ns['correctness_metric'].evaluation_steps
    exec(cell('#@title 評価メトリック'), ns)
    exec(cell('#@title TSVファイルを出力しない'), ns)
    assert 0 <= ns['score'] <= 1 and ns['reason'].strip()
    report['checks']['single'] = {'score': ns['score'], 'reason': ns['reason']}

    from deepeval.config.settings import get_settings
    get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = .001
    timed = ns['evaluate_tsv'](args.sample, target / 'timeout.scored.tsv',
                               ns['correctness_metric'], columns)
    assert timed.Status.tolist() == ['timeout', 'timeout', 'invalid_input']
    report['checks']['timeout'] = timed.Status.tolist()
    get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 60
    exec(cell('#@title 評価メトリック'), ns)
    recovered = ns['evaluate_tsv'](args.sample, target / 'recovery.scored.tsv',
                                   ns['correctness_metric'], columns)
    assert recovered.Status.tolist() == ['success', 'success', 'invalid_input']
    report['checks']['recovery'] = recovered.fillna('').to_dict(orient='records')
    # Exercise a real refused loopback connection without interrupting the server.
    from deepeval.models import OllamaModel
    import pandas as pd
    ns['judge'] = OllamaModel(model=args.model, base_url='http://127.0.0.1:1',
                               timeout=.5, trust_env=False)
    exec(cell('#@title 評価メトリック'), ns)
    connection_failure = None
    try:
        ns['evaluate_tsv'](args.sample, target / 'connection.scored.tsv',
                            ns['correctness_metric'], columns)
    except Exception as exc:
        connection_failure = exc
    assert connection_failure is not None
    partial = pd.read_csv(target / 'connection.scored.tsv', sep='\t', keep_default_na=False)
    assert partial.Status.tolist() == ['fatal_error', 'not_processed', 'not_processed']
    assert partial.Score.tolist() == ['', '', '']
    report['checks']['connection_error'] = partial.to_dict(orient='records')
    report['ollama_models'] = client.list().model_dump(mode='json')
    report['ollama_running'] = client.ps().model_dump(mode='json')
    (target / 'live-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('LIVE_CHECKS_PASS', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
