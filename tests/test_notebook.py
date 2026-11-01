"""Exercise the Notebook's actual code and DeepEval with HTTP-only Ollama mocks."""
import ast
import asyncio
import csv
import json
import os
from pathlib import Path
from unittest.mock import Mock

os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = '1'
os.environ['DEEPEVAL_RETRY_MAX_ATTEMPTS'] = '1'

import httpx
import nbformat
import ollama
import pandas as pd
import pytest
from IPython.core.inputtransformer2 import TransformerManager
from deepeval.config.settings import get_settings
from deepeval.models import OllamaModel

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(ROOT / 'G-Eval_sample_format.ipynb', as_version=4)
COLUMNS = ['input_column', 'actual_output_column', 'expected_output_column']


def cell(title):
    return next(c.source for c in NOTEBOOK.cells if c.source.startswith(title))


def definitions(source, namespace):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef,
    ))]
    exec(compile(tree, '<notebook definitions>', 'exec'), namespace)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        raise AssertionError('Unexpected network request')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


@pytest.fixture
def notebook():
    ns = {'MODEL_NAME': 'llama3.2', 'OLLAMA_BASE_URL': 'http://127.0.0.1:11434'}
    exec(cell('#@title DeepEvalの通信'), ns)
    exec(cell('#@title DeepEvalにOllama'), ns)
    exec(cell('#@title 評価メトリック'), ns)
    definitions(cell('#@title 評価を実行して'), ns)
    return ns


def wire_judge(monkeypatch, handler):
    """Keep real GEval, OllamaModel, Ollama SDK and JSON/schema parsing."""
    def load_model(self, async_mode=False):
        assert async_mode, 'This notebook must use the cancellable async path'
        return ollama.AsyncClient(host=self.base_url, timeout=1,
                                  transport=httpx.MockTransport(handler))
    monkeypatch.setattr(OllamaModel, 'load_model', load_model)


def response(payload):
    return httpx.Response(200, json={
        'model': 'llama3.2', 'message': {'role': 'assistant',
        'content': json.dumps(payload, ensure_ascii=False)}, 'done': True,
    })


def write_input(path, rows, columns=COLUMNS):
    with path.open('w', encoding='utf-8', newline='') as stream:
        w = csv.writer(stream, delimiter='\t')
        w.writerow(columns)
        w.writerows(rows)


def run_tsv(ns, tmp_path, rows, columns=COLUMNS):
    source, target = tmp_path / 'input.tsv', tmp_path / 'output.tsv'
    write_input(source, rows, columns)
    result = ns['evaluate_tsv'](source, target, ns['correctness_metric'], COLUMNS)
    saved = pd.read_csv(target, sep='\t', dtype=str, keep_default_na=False,
                        skip_blank_lines=False)
    return result, saved


def test_notebook_structure_and_all_cells_compile():
    nbformat.validate(NOTEBOOK)
    assert len(NOTEBOOK.cells) == 14
    transformer = TransformerManager()
    for c in NOTEBOOK.cells:
        if c.cell_type == 'code':
            compile(transformer.transform_cell(c.source), '<notebook>', 'exec')
            assert not c.outputs and c.execution_count is None
    assert NOTEBOOK.cells[-1].source.startswith('#@title ランタイムの切断')


def test_success_invalid_response_and_empty_rows(notebook, monkeypatch, tmp_path):
    payloads = iter([{'score': 8, 'reason': '青色。\t参照と一致\n次の行'},
                     {'reason': 'score is missing'}, {'score': 5, 'reason': '部分一致'}])
    async def handler(request):
        body = json.loads(request.content)
        assert body['model'] == 'llama3.2'
        assert isinstance(body['format'], dict)
        assert body['options']['temperature'] == 0
        return response(next(payloads))
    wire_judge(monkeypatch, handler)
    rows = [['架空の町\tリノ\n質問', '"青色"', '青色'], ['q', 'a', 'b'],
            ['q', '', 'b'], ['q', '  ', 'b'], ['q', 'a', 'b']]
    data, saved = run_tsv(notebook, tmp_path, rows)
    assert saved.Status.tolist() == ['success', 'invalid_response', 'invalid_input',
                                     'invalid_input', 'success']
    assert data.Score.iloc[0] == .8 and data.Score.iloc[-1] == .5
    assert saved.Score.tolist()[1:4] == ['', '', '']
    assert saved.Reason.iloc[1] == '' and saved.Error.iloc[1]
    assert saved[COLUMNS].values.tolist() == rows
    assert saved.Reason.iloc[0] == '青色。\t参照と一致\n次の行'
    assert notebook['correctness_metric'].model is notebook['judge']
    assert notebook['judge'].base_url == 'http://127.0.0.1:11434'


def test_criteria_generated_once_and_reused(notebook, monkeypatch, tmp_path, capsys):
    exec(cell('#@title 評価メトリック').replace("evaluation_mode = 'fixed'", "evaluation_mode = 'criteria'"), notebook)
    calls = []
    async def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            assert 'steps' in body['format']['properties']
            return response({'steps': ['Compare reference and actual output.']})
        return response({'score': 9, 'reason': 'ok'})
    wire_judge(monkeypatch, handler)
    _, saved = run_tsv(notebook, tmp_path, [['q', 'a', 'b']] * 2)
    assert saved.Status.tolist() == ['success', 'success']
    assert len(calls) == 3  # one steps generation + two scores
    assert notebook['correctness_metric'].evaluation_steps == ['Compare reference and actual output.']
    assert 'Compare reference and actual output.' in capsys.readouterr().out


def test_official_timeout_cancels_then_next_row_succeeds(notebook, monkeypatch, tmp_path):
    get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = .05
    calls = 0
    cancelled = []
    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)
        return response({'score': 7, 'reason': 'after timeout'})
    wire_judge(monkeypatch, handler)
    # A running event loop resembles Colab/IPython's execution context.
    async def execute():
        return run_tsv(notebook, tmp_path, [['q', 'a', 'b']] * 2)
    _, saved = asyncio.run(execute())
    assert cancelled == [True]
    assert calls == 2
    assert saved.Status.tolist() == ['timeout', 'success']
    assert saved.Score.tolist() == ['', '0.7']
    assert saved.Reason.iloc[0] == ''


@pytest.mark.parametrize('failure', ['connection', 'http401', 'http404', 'http500', 'unknown'])
def test_fatal_errors_save_all_rows_and_reraise(notebook, monkeypatch, tmp_path, failure):
    calls = 0
    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response({'score': 10, 'reason': 'first'})
        if failure == 'connection':
            raise httpx.ConnectError('offline', request=request)
        if failure.startswith('http'):
            return httpx.Response(int(failure[4:]), json={'error': 'service failure'})
        raise RuntimeError('unexpected implementation failure')
    wire_judge(monkeypatch, handler)
    with pytest.raises(Exception):
        run_tsv(notebook, tmp_path, [['q', 'a', 'b']] * 3)
    saved = pd.read_csv(tmp_path / 'output.tsv', sep='\t', keep_default_na=False)
    assert saved.Status.tolist() == ['success', 'fatal_error', 'not_processed']
    assert saved.Score.tolist() == ['1.0', '', '']
    assert saved.Reason.tolist() == ['first', '', '']
    assert saved.Error.iloc[1]
    assert calls == 2


@pytest.mark.parametrize('payload', [{'score': 11, 'reason': 'bad'},
                                     {'score': -1, 'reason': 'bad'},
                                     {'score': float('nan'), 'reason': 'bad'},
                                     {'score': 5, 'reason': ''}])
def test_invalid_scores_and_reason_are_not_saved(notebook, monkeypatch, tmp_path, payload):
    async def handler(request):
        return response(payload)
    wire_judge(monkeypatch, handler)
    _, saved = run_tsv(notebook, tmp_path, [['q', 'a', 'b']])
    assert saved.Status.tolist() == ['invalid_response']
    assert saved.Score.iloc[0] == saved.Reason.iloc[0] == ''


@pytest.mark.parametrize('columns', [COLUMNS[:2], COLUMNS + ['Score'],
                                     [COLUMNS[0], COLUMNS[0], COLUMNS[2]]])
def test_header_errors_before_evaluation(notebook, tmp_path, columns):
    metric = Mock()
    source, target = tmp_path / 'input.tsv', tmp_path / 'output.tsv'
    write_input(source, [['q'] * len(columns)], columns)
    with pytest.raises(ValueError):
        notebook['evaluate_tsv'](source, target, metric, COLUMNS)
    metric.measure.assert_not_called()
    assert not target.exists()


def test_existing_output_and_same_path_are_preserved(notebook, tmp_path):
    source = tmp_path / 'input.tsv'
    write_input(source, [['q', 'a', 'b']])
    original = source.read_bytes()
    with pytest.raises(ValueError):
        notebook['evaluate_tsv'](source, source, Mock(), COLUMNS)
    target = tmp_path / 'output.tsv'
    target.write_text('existing result')
    with pytest.raises(FileExistsError):
        notebook['evaluate_tsv'](source, target, Mock(), COLUMNS)
    assert target.read_text() == 'existing result'
    assert source.read_bytes() == original


def test_blank_line_and_literal_na_preserved(notebook, monkeypatch, tmp_path):
    async def handler(request):
        return response({'score': 8, 'reason': 'ok'})
    wire_judge(monkeypatch, handler)
    source, target = tmp_path / 'input.tsv', tmp_path / 'output.tsv'
    source.write_text('\t'.join(COLUMNS) + '\nNA\ta\tb\n\nq\ta\tb\n', encoding='utf-8')
    result = notebook['evaluate_tsv'](source, target, notebook['correctness_metric'], COLUMNS)
    assert result.Status.tolist() == ['success', 'invalid_input', 'success']
    assert result.input_column.iloc[0] == 'NA'


def test_server_wait_and_owned_process_shutdown(monkeypatch):
    ns = {'OLLAMA_LOG': Path('/tmp/ollama.log')}
    definitions(cell('#@title 確認した公式'), ns)
    process = Mock()
    process.poll.return_value = None
    ns['ollama_process'] = process
    ready = Mock()
    ready.__enter__ = Mock(return_value=Mock(status=200))
    ready.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(ns['urllib'].request, 'build_opener', Mock(return_value=Mock(open=Mock(return_value=ready))))
    ns['wait_for_ollama'](process, 'http://127.0.0.1:11434')
    with pytest.raises(TimeoutError):
        ns['wait_for_ollama'](process, 'http://127.0.0.1:11434', timeout=0)
    process.poll.return_value = 1
    with pytest.raises(RuntimeError):
        ns['wait_for_ollama'](process, 'http://127.0.0.1:11434')
    ns['stop_ollama_server']()
    process.terminate.assert_not_called()
    process.poll.return_value = None
    ns['stop_ollama_server']()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)


def test_single_example_runs_without_tsv(notebook, monkeypatch):
    async def handler(request):
        return response({'score': 10, 'reason': 'matches'})
    wire_judge(monkeypatch, handler)
    exec(cell('#@title TSVファイルを出力しない'), notebook)
    assert notebook['correctness_metric'].score == 1


def test_extra_fields_never_silently_shift_columns(notebook, tmp_path):
    source, target = tmp_path / 'input.tsv', tmp_path / 'output.tsv'
    write_input(source, [['q', 'a', 'b', 'extra']])
    metric = Mock()
    with pytest.raises(ValueError, match='列数'):
        notebook['evaluate_tsv'](source, target, metric, COLUMNS)
    metric.measure.assert_not_called()
    assert not target.exists()


def test_missing_trailing_field_and_empty_dataset(notebook, monkeypatch, tmp_path):
    async def handler(request):
        return response({'score': 8, 'reason': 'ok'})
    wire_judge(monkeypatch, handler)
    _, saved = run_tsv(notebook, tmp_path, [['q', 'a']])
    assert saved.Status.tolist() == ['invalid_input']
    assert saved.expected_output_column.iloc[0] == ''
    other = tmp_path / 'empty'
    other.mkdir()
    _, saved = run_tsv(notebook, other, [])
    assert len(saved) == 0
    assert saved.columns.tolist() == COLUMNS + ['Score', 'Reason', 'Status', 'Error']


@pytest.mark.skipif(not (ROOT / 'documents/qiita.md').exists(),
                    reason='Qiita manuscript is intentionally local-only')
def test_article_code_matches_notebook_in_execution_order():
    import re
    article = (ROOT / 'documents/qiita.md').read_text()
    snippets = re.findall(r'```python\n(.*?)```', article, flags=re.S)
    cells = [c.source for c in NOTEBOOK.cells if c.cell_type == 'code']
    assert snippets == cells


def test_server_retries_readiness_and_kills_only_owned_process(monkeypatch):
    import subprocess
    import urllib.error
    ns = {'OLLAMA_LOG': Path('/tmp/ollama.log')}
    definitions(cell('#@title 確認した公式'), ns)
    process = Mock()
    process.poll.return_value = None
    ns['ollama_process'] = process
    ready = Mock()
    ready.__enter__ = Mock(return_value=Mock(status=200))
    ready.__exit__ = Mock(return_value=False)
    probe = Mock(side_effect=[urllib.error.URLError('starting'), ready])
    monkeypatch.setattr(ns['urllib'].request, 'build_opener', Mock(return_value=Mock(open=probe)))
    monkeypatch.setattr(ns['time'], 'sleep', Mock())
    ns['wait_for_ollama'](process, 'http://127.0.0.1:11434')
    assert probe.call_count == 2
    process.wait.side_effect = [subprocess.TimeoutExpired('ollama', 5), None]
    ns['stop_ollama_server']()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


@pytest.mark.parametrize('generated_steps', [[], [''], ['   ']])
def test_empty_generated_steps_cannot_be_saved_or_reused(notebook, monkeypatch, tmp_path,
                                                       generated_steps):
    exec(cell('#@title 評価メトリック').replace("evaluation_mode = 'fixed'", "evaluation_mode = 'criteria'"), notebook)
    calls = []
    async def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if 'steps' in body['format']['properties']:
            return response({'steps': generated_steps if len(calls) == 1
                              else ['Compare the actual output with the reference.']})
        return response({'score': 8, 'reason': 'ok'})
    wire_judge(monkeypatch, handler)
    _, saved = run_tsv(notebook, tmp_path, [['q', 'a', 'b']] * 2)
    assert saved.Status.tolist() == ['invalid_response', 'success']
    assert saved.Score.tolist() == ['', '0.8']
    assert saved.Reason.iloc[0] == ''
    assert len(calls) == 4  # invalid steps and score discarded, then regenerate and score


def test_parallel_output_reservation_refuses_second_run(notebook, monkeypatch, tmp_path):
    source, target = tmp_path / 'input.tsv', tmp_path / 'output.tsv'
    write_input(source, [['q', 'a', 'b']])
    calls = []
    async def handler(request):
        calls.append(request)
        reserved = pd.read_csv(target, sep='\t')
        assert reserved.Status.tolist() == ['not_processed']
        with pytest.raises(FileExistsError):
            notebook['evaluate_tsv'](source, target, notebook['correctness_metric'], COLUMNS)
        return response({'score': 9, 'reason': 'first writer'})
    wire_judge(monkeypatch, handler)
    notebook['evaluate_tsv'](source, target, notebook['correctness_metric'], COLUMNS)
    saved = pd.read_csv(target, sep='\t')
    assert saved.Score.tolist() == [.9]
    assert saved.Reason.tolist() == ['first writer']
    assert len(calls) == 1


def test_interrupt_during_server_startup_stops_owned_process(monkeypatch, tmp_path):
    import atexit
    import socket
    import subprocess
    import urllib.request
    source = cell('#@title 確認した公式').replace("Path('/content/ollama.log')", repr(tmp_path / 'ollama.log'))
    process = Mock()
    process.poll.return_value = None
    process_factory = Mock(return_value=process)
    monkeypatch.setattr(subprocess, 'run', Mock(return_value=Mock(stdout='', stderr='')))
    monkeypatch.setattr(subprocess, 'Popen', process_factory)
    monkeypatch.setattr(socket.socket, 'connect_ex', Mock(return_value=1))
    monkeypatch.setattr(urllib.request, 'build_opener', Mock(return_value=Mock(open=Mock(side_effect=KeyboardInterrupt))))
    ns = {'installer': tmp_path / 'install.sh', 'PosixPath': type(tmp_path)}
    try:
        with pytest.raises(KeyboardInterrupt):
            exec(source, ns)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        assert process_factory.call_args.kwargs['env']['OLLAMA_HOST'] == '127.0.0.1:11434'
    finally:
        if 'stop_ollama_server' in ns:
            atexit.unregister(ns['stop_ollama_server'])


def test_local_clients_ignore_proxy_environment(notebook):
    assert notebook['judge'].kwargs['trust_env'] is False
    assert 'trust_env=False' in cell('#@title モデルの存在')


@pytest.mark.parametrize('payload', [{'score': 15, 'reason': 'bad'},
                                     {'score': 8, 'reason': ''}])
def test_single_example_rejects_invalid_scores(notebook, monkeypatch, payload):
    async def handler(request):
        return response(payload)
    wire_judge(monkeypatch, handler)
    with pytest.raises(notebook['InvalidEvaluationResponse']):
        exec(cell('#@title TSVファイルを出力しない'), notebook)


def test_setup_installs_into_separate_dependency_environment(monkeypatch):
    import subprocess
    from importlib.metadata import distribution
    site = str(distribution('deepeval').locate_file(''))
    run = Mock()
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'check_output', Mock(return_value=site + '\n'))
    ns = {}
    exec(cell('#@title 必要なライブラリ'), ns)
    pip_commands = [call.args[0] for call in run.call_args_list
                    if 'pip' in call.args[0]]
    assert pip_commands
    assert all('--python' in command and '/content/geval-venv/bin/python' in command
               for command in pip_commands)
    assert pip_commands[-1][-1] == 'check'
    assert 'pandas==2.2.3' in pip_commands[-2]


def test_setup_refuses_already_imported_external_packages(monkeypatch, tmp_path):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', Mock(return_value=Mock(stdout='', stderr='')))
    monkeypatch.setattr(subprocess, 'check_output', Mock(return_value=str(tmp_path) + '\n'))
    with pytest.raises(RuntimeError, match='再起動'):
        exec(cell('#@title 必要なライブラリ'), {})


def test_missing_zstd_is_installed_without_driver_changes(monkeypatch):
    import subprocess
    import shutil
    run = Mock(return_value=Mock(stdout='', stderr=''))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(shutil, 'which', Mock(return_value=None))
    monkeypatch.setattr(Path, 'read_text', Mock(return_value='#!/bin/sh\n'))
    exec(cell('#@title Ollamaをインストール'), {})
    commands = [call.args[0] for call in run.call_args_list]
    assert ['apt-get', 'install', '-y', 'zstd'] in commands
    assert not any('cuda-drivers' in command for command in commands)


def test_setup_rejects_conflicting_loaded_transitive_dependency(monkeypatch):
    import subprocess
    import sys
    from importlib import metadata
    from types import SimpleNamespace
    site = str(metadata.distribution('deepeval').locate_file(''))
    original = metadata.distributions
    def distributions(**kwargs):
        found = list(original(**kwargs))
        if kwargs.get('path') != [site]:
            found = [d for d in found if d.metadata['Name'].lower() != 'requests']
            found.append(SimpleNamespace(metadata={'Name': 'requests'}, version='0.1'))
        return found
    monkeypatch.setattr(subprocess, 'run', Mock(return_value=Mock(stdout='', stderr='')))
    monkeypatch.setattr(subprocess, 'check_output', Mock(return_value=site + '\n'))
    monkeypatch.setattr(metadata, 'distributions', distributions)
    monkeypatch.setattr(metadata, 'packages_distributions', Mock(return_value={'requests': ['requests']}))
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(__file__='/external/requests/__init__.py'))
    with pytest.raises(RuntimeError, match='競合'):
        exec(cell('#@title 必要なライブラリ'), {})
