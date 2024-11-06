import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import uuid
from audio_data import AudioData, db_to_amp
from pathlib import Path
import _plotly_patch  # noqa: F401
from enum import StrEnum, auto
from plotly_resampler import FigureResampler
from plotly.subplots import make_subplots
import plotly
import time
import gc
from dataclasses import dataclass, asdict
import json


app = dash.Dash(__name__)


@dataclass
class SessionData:
    audio: AudioData
    fig: FigureResampler


_session_data: dict[str, SessionData] = {}
_play_timestamp: dict[str, float] = {}
COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS
COOKIE_NAME = "record_normalizer"
DEFAULT_COOKIE = json.dumps({"src_dir": "", "dst_dir": ""})


@dataclass
class CookieData:
    src_dir: str
    dst_dir: str

    @staticmethod
    def get_cookie() -> "CookieData":
        return CookieData(**json.loads(dash.callback_context.cookies.get(COOKIE_NAME, DEFAULT_COOKIE)))

    def set_cookie(self) -> None:
        dash.callback_context.response.set_cookie(COOKIE_NAME, json.dumps(asdict(self)), max_age=None)


class UiId(StrEnum):
    STORE_SESSION_ID = auto()
    STORE_PLAY_TIMESTAMP = auto()
    DRP_INPUT_FILE = auto()
    INPUT_SRC_DIR = auto()
    BTN_SRC_DIR = auto()
    INPUT_DST_DIR = auto()
    INPUT_DST_NAME = auto()
    BTN_SAVE = auto()
    DOWNLOAD = auto()
    DIV_GRAPH = auto()
    GRAPH = auto()
    CHK_Y_FIXED = auto()
    CHK_SYNC = auto()
    SLIDER_L_COMPRESS = auto()
    SLIDER_L_LIMIT = auto()
    SLIDER_R_COMPRESS = auto()
    SLIDER_R_LIMIT = auto()
    INPUT_START = auto()
    INPUT_END = auto()


def create_layout():
    # UI定義
    session_id = str(uuid.uuid4())
    db_slider = {"min": -15, "max": 0, "step": 0.5, "value": -6, "marks": {v: str(v) for v in range(-15, 1)}}
    return html.Div(
        [
            dcc.Store(id=UiId.STORE_SESSION_ID, data=session_id),
            # ファイル読み込みボタン
            html.Div(
                [
                    dcc.Dropdown(id=UiId.DRP_INPUT_FILE, options=[], style={"flexGrow": 4}),
                    html.Button("設定", id=UiId.BTN_SRC_DIR, style={"flexGrow": 1}),
                    dcc.Input(id=UiId.INPUT_SRC_DIR, value="", style={"flexGrow": 10}),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "columnGap": 10,
                    "width": 800,
                },
            ),
            # 出力書き込みボタン
            html.Div(
                [
                    dcc.Input(id=UiId.INPUT_DST_NAME, value="", style={"flexGrow": 4}),
                    html.Button("保存", id=UiId.BTN_SAVE, style={"flexGrow": 1}),
                    dcc.Input(id=UiId.INPUT_DST_DIR, value="", style={"flexGrow": 10}),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "columnGap": 10,
                    "width": 800,
                },
            ),
            # 波形プロット
            dcc.Loading(
                [html.Div([dcc.Graph(id=UiId.GRAPH)], id=UiId.DIV_GRAPH)],
                target_components={UiId.DIV_GRAPH: "children"},
            ),
            # 制御
            html.Div(
                [
                    # y軸固定
                    dcc.Checklist(id=UiId.CHK_Y_FIXED, options=["Y fixed"], value=["Y fixed"]),
                    # LR同期チェックボックス
                    dcc.Checklist(id=UiId.CHK_SYNC, options=["LR sync"], value=["LR sync"]),
                    # クロップ範囲
                    html.Label("Start"),
                    dcc.Input(id=UiId.INPUT_START, value=0, min=0, type="number", style={"width": 50}),
                    html.Label("End"),
                    dcc.Input(id=UiId.INPUT_END, value=0, max=0, type="number", style={"width": 50}),
                ],
                style={"display": "flex", "flexDirection": "row", "alignItems": "center", "columnGap": 10},
            ),
            # Lチャンネル閾値
            html.Div(
                [
                    # Lチャンネルコンプレッサー
                    html.Div(
                        [
                            html.Label("L Compressor"),
                            dcc.Slider(id=UiId.SLIDER_L_COMPRESS, **db_slider),
                        ],
                        style={"width": "50%"},
                    ),
                    # Lチャンネルリミッター
                    html.Div(
                        [
                            html.Label("L Limitter"),
                            dcc.Slider(id=UiId.SLIDER_L_LIMIT, **db_slider),
                        ],
                        style={"width": "50%"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "row", "columnGap": 10},
            ),
            # Rチャンネル閾値
            html.Div(
                [
                    # Rチャンネルコンプレッサー
                    html.Div(
                        [
                            html.Label("R Compressor"),
                            dcc.Slider(id=UiId.SLIDER_R_COMPRESS, **db_slider),
                        ],
                        style={"width": "50%"},
                    ),
                    # Rチャンネルリミッター
                    html.Div(
                        [
                            html.Label("R Limitter"),
                            dcc.Slider(id=UiId.SLIDER_R_LIMIT, **db_slider),
                        ],
                        style={"width": "50%"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "row", "columnGap": 10},
            ),
        ]
    )


app.layout = create_layout


def create_hline(name: str, y: float, len_signal: float, color: str, axis: int) -> dict:
    """水平ライン作成"""
    return {
        "mode": "lines",
        "name": name,
        "x": [0, len_signal, len_signal, 0],
        "y": [y, y, -y, -y],
        "type": "scatter",
        "xaxis": f"x{axis}",
        "yaxis": f"y{axis}",
        "line": {"color": color, "width": 1},
    }


def create_vline(name: str, x: float, color: str, axis: int) -> dict:
    """垂直ライン作成"""
    return {
        "mode": "lines",
        "name": name,
        "x": [x, x],
        "y": [1.0, -1],
        "type": "scatter",
        "xaxis": f"x{axis}",
        "yaxis": f"y{axis}",
        "line": {"color": color, "width": 1},
    }


# 初期化
@app.callback(
    [
        Output(UiId.INPUT_SRC_DIR, "value"),
        Output(UiId.INPUT_DST_DIR, "value"),
        Output(UiId.DRP_INPUT_FILE, "options"),
    ],
    [Input(UiId.STORE_SESSION_ID, "data")],
)
def init_dir(session_id: str) -> None:
    cookie_data = CookieData.get_cookie()
    audio_names = [
        p.name for p in sorted(list(Path(cookie_data.src_dir).glob("*"))) if p.suffix.lower() in (".mp3", ".flac")
    ]
    return [cookie_data.src_dir, cookie_data.dst_dir, audio_names]


# 描画範囲更新
@app.callback(
    Output(UiId.GRAPH, "figure", allow_duplicate=True),
    Input(UiId.GRAPH, "relayoutData"),
    State(UiId.STORE_SESSION_ID, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, session_id: str):
    data = _session_data.get(session_id, None)
    if data is None:
        return no_update
    return data.fig.construct_update_data_patch(relayoutdata)


# ファイル読み込み
@app.callback(
    [
        Output(UiId.DIV_GRAPH, "children"),
        Output(UiId.INPUT_DST_NAME, "value"),
        Output(UiId.INPUT_START, "value"),
        Output(UiId.INPUT_END, "value"),
    ],
    [
        Input(UiId.DRP_INPUT_FILE, "value"),
        State(UiId.STORE_SESSION_ID, "data"),
        State(UiId.INPUT_SRC_DIR, "value"),
        State(UiId.CHK_Y_FIXED, "value"),
    ],
)
def load_file(filename: str, session_id: str, src_dir: str, y_fixed: list[str]):
    if filename is None:
        return no_update

    global _session_data
    data = _session_data.get(session_id)
    if data is not None:
        # 前のデータは削除
        data.audio.stop()
        _session_data[session_id] = None
        del data
        gc.collect()

    print("filename:", filename)
    audio = AudioData(Path(src_dir) / filename)

    # 波形プロット
    # x = pd.to_datetime(audio.times, unit="s")
    figure = FigureResampler(make_subplots(rows=2, cols=1, shared_xaxes=True))
    figure.add_trace(
        go.Scattergl(
            mode="lines",
            name="L",
            line=dict(width=1, color=COLORS[0]),
        ),
        hf_x=audio.times,
        hf_y=audio.signal[:, 0].astype(np.float16),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            mode="lines",
            name="R",
            line=dict(width=1, color=COLORS[0]),
        ),
        hf_x=audio.times,
        hf_y=audio.signal[:, 1].astype(np.float16),
        row=2,
        col=1,
    )
    figure.update_layout(height=500, margin=dict(l=10, t=10, r=10, b=10))
    figure.update_yaxes(fixedrange=len(y_fixed) > 0)
    children = [
        dcc.Graph(id=UiId.GRAPH, figure=figure),
    ]

    _session_data[session_id] = SessionData(audio, figure)

    return [children, Path(filename).stem, 0, 0]


# 入力フォルダ設定
@app.callback(
    Output(UiId.DRP_INPUT_FILE, "options", allow_duplicate=True),
    Input(UiId.BTN_SRC_DIR, "n_clicks"),
    State(UiId.INPUT_SRC_DIR, "value"),
    prevent_initial_call=True,
)
def change_src_dir(n_clicks: int, src_dir: str):
    if n_clicks is None:
        return no_update

    cookie_data = CookieData.get_cookie()
    cookie_data.src_dir = src_dir
    cookie_data.set_cookie()
    audio_names = [p.name for p in sorted(list(Path(src_dir).glob("*"))) if p.suffix.lower() in (".mp3", ".flac")]
    return audio_names


# 保存
@app.callback(
    Output(UiId.INPUT_DST_DIR, "value", allow_duplicate=True),  # OutputがないとCookieが保存されない
    Input(UiId.BTN_SAVE, "n_clicks"),
    State(UiId.STORE_SESSION_ID, "data"),
    State(UiId.INPUT_DST_DIR, "value"),
    State(UiId.INPUT_DST_NAME, "value"),
    State(UiId.INPUT_START, "value"),
    State(UiId.INPUT_END, "value"),
    State(UiId.SLIDER_L_COMPRESS, "value"),
    State(UiId.SLIDER_L_LIMIT, "value"),
    State(UiId.SLIDER_R_COMPRESS, "value"),
    State(UiId.SLIDER_R_LIMIT, "value"),
    prevent_initial_call=True,
)
def save(
    n_clicks: int,
    session_id: str,
    dst_dir: str,
    dst_name: str,
    start: float,
    end: float,
    l_compress: float,
    l_limit: float,
    r_compress: float,
    r_limit: float,
):
    if n_clicks is None:
        return no_update

    cookie_data = CookieData.get_cookie()
    cookie_data.dst_dir = dst_dir
    cookie_data.set_cookie()

    data = _session_data.get(session_id, None)
    if data is None:
        return dst_dir

    data.audio.save(Path(dst_dir) / f"{dst_name}.mp3", start, end, l_compress, l_limit, r_compress, r_limit)
    print("saved")

    return dst_dir


# 再生
@app.callback(Input(UiId.GRAPH, "clickData"), State(UiId.STORE_SESSION_ID, "data"))
def play(clickData: dict, session_id: str):
    if not clickData:
        return no_update

    data = _session_data.get(session_id, None)
    if data is None:
        return no_update

    data.audio.play(clickData["points"][0]["x"])

    # 再生したタイミングを記録
    global _play_timestamp
    _play_timestamp[session_id] = time.time()


# 停止
@app.callback(Input(UiId.DIV_GRAPH, "n_clicks"), State(UiId.STORE_SESSION_ID, "data"))
def stop(n_clicks: int, session_id: str):
    if n_clicks == 0:
        return no_update

    data = _session_data.get(session_id, None)
    if data is None:
        return no_update

    # 再生直後に停止しないように、再生したタイミングをチェック
    global _play_timestamp
    if time.time() - _play_timestamp.get(session_id, 0) > 0.1:
        data.audio.stop()


# グラフ更新
@app.callback(
    [Output(UiId.GRAPH, "figure")],
    [
        Input(UiId.INPUT_START, "value"),
        Input(UiId.INPUT_END, "value"),
        Input(UiId.SLIDER_L_COMPRESS, "value"),
        Input(UiId.SLIDER_L_LIMIT, "value"),
        Input(UiId.SLIDER_R_COMPRESS, "value"),
        Input(UiId.SLIDER_R_LIMIT, "value"),
        State(UiId.STORE_SESSION_ID, "data"),
        State(UiId.CHK_SYNC, "value"),
        State(UiId.GRAPH, "figure"),
    ],
)
def change_value(
    input_start: int,
    input_end: int,
    l_compress: float,
    l_limit: float,
    r_compress: float,
    r_limit: float,
    session_id: str,
    sync: list[str],
    fig: dict,
):
    if l_compress is None:
        return no_update

    data = _session_data.get(session_id, None)
    if data is None:
        return no_update

    fig_data = fig["data"]
    for _ in range(len(fig_data), 10):
        fig_data.append({})
    len_signal = len(data.audio.signal) / data.audio.sr
    fig_data[2] = create_hline("l-compress", db_to_amp(l_compress), len_signal, COLORS[1], 1)
    fig_data[3] = create_hline("l-limit", db_to_amp(l_limit), len_signal, COLORS[2], 1)
    fig_data[4] = create_vline("start", input_start, COLORS[3], 1)
    fig_data[5] = create_vline("end", data.audio.end_time + input_end, COLORS[3], 1)
    fig_data[6] = create_hline("r-compress", db_to_amp(r_compress), len_signal, COLORS[1], 2)
    fig_data[7] = create_hline("r-limit", db_to_amp(r_limit), len_signal, COLORS[2], 2)
    fig_data[8] = create_vline("start", input_start, COLORS[3], 2)
    fig_data[9] = create_vline("end", data.audio.end_time + input_end, COLORS[3], 2)

    if len(sync):
        r_compress = l_compress
        r_limit = l_limit
    return [fig]


# Y軸固定
@app.callback(
    [Output(UiId.GRAPH, "figure", allow_duplicate=True)],
    [Input(UiId.CHK_Y_FIXED, "value"), State(UiId.GRAPH, "figure")],
    prevent_initial_call=True,
)
def change_y_fixed(y_fixed: list[str], fig: dict):
    if y_fixed is None:
        return no_update
    fig["layout"]["yaxis"]["fixedrange"] = len(y_fixed) > 0
    fig["layout"]["yaxis2"]["fixedrange"] = len(y_fixed) > 0
    return [fig]


# L Compressor
@app.callback(
    [
        Output(UiId.SLIDER_L_LIMIT, "value", allow_duplicate=True),
        Output(UiId.SLIDER_R_COMPRESS, "value", allow_duplicate=True),
    ],
    [
        Input(UiId.SLIDER_L_COMPRESS, "value"),
        State(UiId.SLIDER_L_LIMIT, "value"),
        State(UiId.SLIDER_R_COMPRESS, "value"),
        State(UiId.CHK_SYNC, "value"),
    ],
    prevent_initial_call=True,
)
def change_l_compress(
    l_compress: float,
    l_limit: float,
    r_compress: float,
    sync: list[str],
):
    if l_limit < l_compress:
        l_limit = l_compress

    if len(sync) > 0:
        r_compress = l_compress
    return [l_limit, r_compress]


# L Limit
@app.callback(
    [
        Output(UiId.SLIDER_L_COMPRESS, "value", allow_duplicate=True),
        Output(UiId.SLIDER_R_LIMIT, "value", allow_duplicate=True),
    ],
    [
        Input(UiId.SLIDER_L_LIMIT, "value"),
        State(UiId.SLIDER_L_COMPRESS, "value"),
        State(UiId.SLIDER_R_LIMIT, "value"),
        State(UiId.CHK_SYNC, "value"),
    ],
    prevent_initial_call=True,
)
def change_l_limit(
    l_limit: float,
    l_compress: float,
    r_limit: float,
    sync: list[str],
):
    if l_compress > l_limit:
        l_compress = l_limit

    if len(sync) > 0:
        r_limit = l_limit
    return [l_compress, r_limit]


# R Compressor
@app.callback(
    [Output(UiId.SLIDER_R_LIMIT, "value"), Output(UiId.SLIDER_L_COMPRESS, "value")],
    [
        Input(UiId.SLIDER_R_COMPRESS, "value"),
        State(UiId.SLIDER_R_LIMIT, "value"),
        State(UiId.SLIDER_L_COMPRESS, "value"),
        State(UiId.CHK_SYNC, "value"),
    ],
    prevent_initial_call=True,
)
def change_r_compress(
    r_compress: float,
    r_limit: float,
    l_compress: float,
    sync: list[str],
):
    if r_limit < r_compress:
        r_limit = r_compress

    if len(sync) > 0:
        l_compress = r_compress
    return [r_limit, l_compress]


# L Limit
@app.callback(
    [Output(UiId.SLIDER_R_COMPRESS, "value"), Output(UiId.SLIDER_L_LIMIT, "value")],
    [
        Input(UiId.SLIDER_R_LIMIT, "value"),
        State(UiId.SLIDER_R_COMPRESS, "value"),
        State(UiId.SLIDER_L_LIMIT, "value"),
        State(UiId.CHK_SYNC, "value"),
    ],
    prevent_initial_call=True,
)
def change_r_limit(
    r_limit: float,
    r_compress: float,
    l_limit: float,
    sync: list[str],
):
    if r_compress > r_limit:
        r_compress = r_limit

    if len(sync) > 0:
        l_limit = r_limit
    return [r_compress, l_limit]


# メイン関数
if __name__ == "__main__":
    # register_plotly_resampler("auto", default_n_shown_samples=10000)
    app.run_server(debug=False, port=5000, threaded=True)
