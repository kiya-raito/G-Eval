# OllamaとDeepEvalを使ったLLM評価

Google ColabのGPUランタイム内でOllamaの`llama3.2`を動かし、DeepEvalの`GEval`で生成文を評価する学習用サンプルです。TSV一括評価と単一例の評価を利用できます。「ローカル」はColabランタイム内を指し、利用者のPC上での推論を意味しません。

## 動作環境

- Google ColabのGPUランタイム（Python 3.13系）を使用し、このNotebook専用の新しいランタイムで、最初に依存環境のインストールセルを実行します。
- DeepEval 4.3.2、Ollama Python SDK 0.6.3、pandas 2.2.3、httpx 0.28.1、pydantic 2.14.2を使用します。依存関係は`requirements.txt`に記載しています。
- Colab既存のCUDAドライバーと`LD_LIBRARY_PATH`を利用し、ドライバーを再インストールしません。

関連する[Qiita記事](https://qiita.com/light_1729/items/a83b72dbb805e7569efb)の初公開日は2025年1月18日です。本リポジトリのコードは2026年11月1日にDeepEval 4.3.2向けに改訂し、ColabのT4 GPU、Python 3.13.16、Ollama 0.40.13で実推論を確認しました。

改訂版の動作確認では、モデルに`llama3.2:latest`（3.2B、Q4_K_M）を使用しました。サンプルTSVは正答1.0・誤答0.0・空欄は入力エラーとなりました。固定手順とcriteriaモード、単一例、フォーム変更、タイムアウト後の再評価、接続障害時の途中保存も確認しています。これはサンプルの動作確認であり、任意の文章に対する評価精度を示すものではありません。

## 評価の流れ

1. `G-Eval_sample_format.ipynb`をColabで開き、GPUランタイムを選択します。
2. 評価用の依存関係を`/content/geval-venv`へインストールします。Colab本体のパッケージを置換せず、このNotebookから評価用の依存を読み込みます。Pythonプロセスは共通なので、他のPython作業との併用は避けます。その後、必要ならDriveをマウントします。以前の評価用ライブラリをimport済みなら、ランタイムを再起動してインストールセルを最初に実行してください。
3. `https://ollama.com/install.sh`を取得し、表示された内容を確認してから次のセルで実行します。展開コマンドzstdがなければ、そのコマンドだけを追加します。スクリプトはColab Linux内のシステムを変更します。ホストPC上でそのまま実行するためのセルではありません。
4. `127.0.0.1:11434`限定でOllamaを起動し、APIの準備完了を待ちます。既存の別サーバーがポートを使用している場合は、起動セルがエラーになります。ログは`/content/ollama.log`です。
5. `llama3.2`を取得し、存在と最小の構造化応答を確認します。評価結果を比較する場合は、Ollamaのバージョンとモデル名を記録してください。厳密なモデルの識別には、一覧に表示されるdigestも利用できます。モデルタグと公式インストーラーは更新されるため、Python依存の固定だけでは環境全体を固定できません。
6. 通信・タイムアウト設定後、`OllamaModel(model="llama3.2", base_url="http://127.0.0.1:11434")`を`GEval`へ渡します。古い`set-local-model`コマンドや暗黙のモデル選択には依存しません。
7. フォームで評価方法・入力パス・列名を設定し、TSVまたは単一例を評価します。TSVを使わない場合はTSVセルをスキップできます。
8. 結果を保存・取得してから、最後のセルで自分が起動したサーバーを停止し、ランタイムを切断します。

推論先はColab内のOllamaです。ngrokによる外部公開や外部モデルAPIを使わず、DeepEvalテレメトリーは初回import前に無効化し、モデル通信は環境変数のHTTPプロキシを使いません。依存・モデルのダウンロードにはネットワークを使い、Driveを選んだ場合はGoogle Driveでデータを扱います。

## 評価基準のカスタマイズ

`evaluation_mode="fixed"`が標準です。明示した`evaluation_steps`で、入力と参照回答に照らした正確性を評価します。`evaluation_mode="criteria"`へ切り替えると、フォームの`criteria`を編集して評価基準を変更できます。

`GEval`へは`evaluation_steps`と`criteria`のどちらか一方を渡します。criteriaモードで生成された手順は同じメトリック内で再利用され、表示されます。空の手順が生成された行はスコアを保存せず、次の行で再生成します。基準を変更したら少数の正答・誤答で生成手順とスコアを確認してください。比較実験ではその手順を保存し、固定モードへ転記してください。設定を変えたら設定セル以降を再実行します。`#@param`は関数の引数ではなく、独立した変数への代入行に付けています。

`expected_output`との整合性評価は、検索等による外部の事実検証ではありません。結果は参照回答の品質、評価基準、評価モデルに依存し、温度0でも完全な決定性は保証されません。

## TSV入出力

`samples/sample.tsv`は架空の町についての3行です。正答例、誤答例、空欄の行を含みます。Colabへ`/content/sample.tsv`としてアップロードするか、Drive上のパスを指定してください。

| 設定 | 初期値 |
| --- | --- |
| 入力 | `/content/sample.tsv` |
| 出力 | `/content/sample.scored.tsv` |
| 入力列 | `input_column` |
| 実出力列 | `actual_output_column` |
| 参照回答列 | `expected_output_column` |

UTF-8・タブ区切りで読み書きし、元の列と行順を維持します。BOM付きUTF-8入力にも対応します。文字列`NA`等を欠損値へ自動変換しません。必須列不足、重複した列名、既存の結果列、入出力の同一路径は開始前にエラーとします。既存の出力ファイルは上書きしません。評価開始前に出力先を排他的に予約し、同じパスを使う並行実行を拒否します。予約中のファイルには全行が`not_processed`で保存され、評価終了・中断時に結果へ更新されます。再実行時は別の出力名を指定してください。空文字・空白だけの必須値は行単位のエラーです。

| 追加列 | 内容 |
| --- | --- |
| `Score` | 成功時の0〜1の有限スコア。失敗・未処理時は空欄 |
| `Reason` | 成功時の理由。失敗・未処理時は空欄 |
| `Status` | `success`、`invalid_input`、`invalid_response`、`timeout`、`fatal_error`、`not_processed` |
| `Error` | エラーの型と内容（入力不備は説明） |

標準60秒の公式タイムアウトを使い、行ごとに逐次評価します。タイムアウトと不正なモデル応答は記録して次へ進みます。接続、モデル未取得、HTTPエラー、未知の例外は途中結果を保存して再送出します。後続行は`not_processed`です。エラーメッセージに含まれる単語だけで接続障害を分類しません。

タイムアウトではクライアント側の非同期処理をキャンセルします。Ollamaサーバー側の推論停止は保証しません。残存処理の影響で次のリクエストが待たされる可能性があります。時間を変更する場合は、通信設定セルとその後のモデル・メトリック定義を再実行してください。SIGALRMは使いません。

## 原論文との関係とスコア

これは[原著者の実装](https://github.com/nlpyang/geval)そのものではなく、[DeepEvalのGEval](https://deepeval.com/docs/metrics-llm-evals)を使うサンプルです。[Liu et al. (2023)](https://aclanthology.org/2023.emnlp-main.153/)はGPT-4等で検証し、出力スコアの確率による重み付き和を提案しています。当時GPT-4はトークン確率を返さないため、論文では20回のサンプリングから推定しています。公開コードと論文には温度設定の差もあり、本サンプルと同一条件ではありません。

原著者の要約用プロンプトではcoherence・consistency・relevanceが1〜5、fluencyが1〜3です。本構成のDeepEval 4.3.2はOllamaの構造化応答から通常スコアを取り、標準内部尺度0〜10を0〜1に正規化します。モデル一覧上のlog probabilities対応フラグにかかわらず、Ollamaアダプターに生応答取得メソッドがなく通常採点へフォールバックします。確率による重み付けや20回サンプリングをこのNotebookが実施するわけではありません。原論文と同じ性能や人手評価との相関は保証しません。

## 出典・ライセンス

- 原論文: Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. 2023. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP, pp. 2511–2522. [論文・BibTeX](https://aclanthology.org/2023.emnlp-main.153/)
- [原著者実装](https://github.com/nlpyang/geval)（MIT License）
- [DeepEval](https://github.com/confident-ai/deepeval)（Apache License 2.0）
- [Ollama](https://github.com/ollama/ollama)・[Ollama Python SDK](https://github.com/ollama/ollama-python)（MIT License）

Built with Llama.

使用モデルには[Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)と[Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)が適用されます。利用・再配布の際はこれらの条件に従ってください。モデル重みは本リポジトリには同梱せず、実行時にOllamaで取得します。

pandasとhttpxはBSD-3-Clause、pydanticはMITで提供されています。各ライブラリは依存関係として取得し、本リポジトリには同梱していません。上記のライセンスは各参照先・依存ライブラリ・モデルのものであり、本リポジトリ独自のコード全体に適用するライセンスを示すものではありません。

## 開発・テスト

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Notebookの構文、評価処理、TSV入出力、エラー処理の回帰テストを実行できます。

実Ollamaとの結合テストは、Notebookの依存環境・サーバー起動・モデル取得・構造化応答確認を終えてから実行します。結果TSVと環境情報を新しいディレクトリへ保存します。

```sh
/content/geval-venv/bin/python tests/run_live_check.py --output-dir /content/geval-live-results
```
