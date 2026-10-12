from datetime import datetime, time, timedelta, timezone
import json
from uuid import uuid4
import httpx
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8000"
SHANGHAI = timezone(timedelta(hours=8))
STATUS = {"queued": "等待处理", "running": "处理中", "succeeded": "已完成", "failed": "失败"}
st.set_page_config(page_title="MindTrack · 纵向追踪", page_icon="🌿", layout="wide")
st.markdown("""<style>
    .block-container {max-width:1150px;padding-top:2rem;}
    h1 {letter-spacing:-1px;} [data-testid="stMetric"] {background:white;padding:16px;border-radius:10px;}
    </style>""", unsafe_allow_html=True)
st.caption("MINDTRACK  /  大创项目学习原型  /  v0.1")
st.title("纵向追踪工作台")
st.write("从一次采集到一段趋势，查看数据处理进度与历史结果。")
st.info("本版本的风险分数由演示场景指定，仅用于系统联调。公开样例是不同来源的模拟组合。")


def request(method, path, **kwargs):
    response = httpx.request(method, API + path, timeout=30, **kwargs)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"请求失败（{response.status_code}）：{detail}")
    return response


try:
    config = request("GET", "/api/config").json()
except (httpx.HTTPError, RuntimeError) as exc:
    st.error("后端暂时不可用，请先在项目目录运行 python run.py。")
    st.caption(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("观察对象")
    subject = st.selectbox("匿名编号", [f"demo_{i:02d}" for i in range(1, 6)])
    st.caption("编号用于关联记录，本地学习版没有登录与权限管理。")
    today = datetime.now(SHANGHAI).date()
    start_day = st.date_input("起始日期", today - timedelta(days=30))
    end_day = st.date_input("结束日期", today)
    st.divider()
    st.caption("文本处理：" + ("本地MiniLM语义向量" if config["text_mode"] == "minilm" else "简单统计占位"))
    st.markdown(f"[查看接口文档]({API}/docs)")

if start_day > end_day:
    st.error("起始日期不能晚于结束日期。")
    st.stop()

params = {"subject_id": subject,
          "start": datetime.combine(start_day, time.min, SHANGHAI).isoformat(),
          "end": datetime.combine(end_day + timedelta(days=1), time.min, SHANGHAI).isoformat(),
          "limit": 10000}
submit_tab, history_tab, learn_tab = st.tabs(["提交与进度", "历史趋势", "学习与替换"])

with submit_tab:
    with st.form("submit"):
        left, right = st.columns(2)
        with left:
            mode = st.radio("数据来源", ["公开样例组合", "上传自己的文件"])
            collected_day = st.date_input("采集日期", today)
            collected_time = st.time_input("采集时间（北京时间）", datetime.now(SHANGHAI).time().replace(microsecond=0))
        with right:
            scenario = st.selectbox("模拟评分场景", config["scenarios"], index=2)
            text = st.text_area("文本描述", "今天学习了一会儿，晚上打算早点休息。", max_chars=4000)
        with st.expander("上传文件（选择“上传自己的文件”时使用，可只上传部分模态）"):
            eeg = st.file_uploader("脑电 EDF", type=["edf"])
            face = st.file_uploader("人脸图片", type=["png", "jpg", "jpeg"])
            audio = st.file_uploader("语音", type=["wav", "ogg", "flac"])
        submitted = st.form_submit_button("提交处理", type="primary", use_container_width=True)
    if submitted:
        payload = {"subject_id": subject, "collection_id": "web_" + uuid4().hex[:16],
                   "collected_at": datetime.combine(collected_day, collected_time, SHANGHAI).isoformat(),
                   "text": text, "scenario": scenario}
        try:
            if mode == "公开样例组合":
                data = request("POST", "/api/demo", json=payload).json()
            else:
                files = {kind: (f.name, f.getvalue(), f.type) for kind, f in
                         {"eeg": eeg, "face": face, "audio": audio}.items() if f is not None}
                data = request("POST", "/api/tasks", data={"metadata": json.dumps(payload)}, files=files).json()
            st.session_state.last_task = data["task_id"]
            st.success("任务已提交。下方进度每两秒刷新，完成后可到“历史趋势”查看。")
        except (httpx.HTTPError, RuntimeError) as exc:
            st.error(str(exc))

    @st.fragment(run_every="2s")
    def progress():
        task_id = st.session_state.get("last_task")
        if not task_id:
            st.caption("提交后在这里查看任务状态。首次处理语音或加载模型可能稍慢。")
            return
        try:
            task = request("GET", f"/api/tasks/{task_id}").json()
            st.write(f"**最近任务 · {STATUS[task['status']]}**")
            st.caption(f"{task['subject_id']} · {task_id}")
            if task["status"] == "failed":
                st.error(task["error"])
                if st.button("重试此任务", key="retry_latest"):
                    request("POST", f"/api/tasks/{task_id}/retry")
                    st.rerun(scope="fragment")
            elif task["result"]:
                result = task["result"]
                a, b, c = st.columns(3)
                a.metric("模拟分数" if result["is_simulated"] else "模型分数", f"{result['score']:.1f}")
                b.metric("演示等级" if result["is_simulated"] else "模型等级", result["risk_label"])
                c.metric("处理模态", len(result["features"]["modalities"]))
                st.caption(result["explanation"])
                if result["risk_level"] >= 3:
                    st.warning("高等级提示已触发。当前为模拟结果，用于验证提示功能。" if result["is_simulated"]
                               else "模型结果达到配置的提示阈值，请按项目约定复核。")
                with st.expander("查看主线一输出（特征、维度、来源说明）"):
                    st.json(result["features"])
            else:
                st.caption("后台正在处理，页面查询不会等待整次推理完成。")
        except (httpx.HTTPError, RuntimeError) as exc:
            st.error(str(exc))
    progress()

with history_tab:
    st.subheader("个人记录与趋势")
    st.caption("按采集时间排序；时间空档不代表连续测量。切换编号或日期可筛选，完成新任务后点击刷新。")
    st.button("刷新历史", key="refresh")
    try:
        records = request("GET", "/api/records", params=params).json()
        completed = [r for r in records if r["result"]]
        a, b, c = st.columns(3)
        a.metric("记录数", len(records))
        b.metric("完成数", len(completed))
        c.metric("失败数", sum(r["status"] == "failed" for r in records))
        if completed:
            frame = pd.DataFrame([{"采集时间": pd.to_datetime(r["collected_at"]).tz_convert("Asia/Shanghai"),
                                   "分数": r["result"]["score"],
                                   "系列": r["result"]["model_version"] + (" · 模拟" if r["result"]["is_simulated"] else " · 模型")}
                                  for r in completed]).sort_values("采集时间")
            st.line_chart(frame, x="采集时间", y="分数", color="系列")
        else:
            st.info("当前范围没有已完成记录。可以先提交一次公开样例组合。")
        if records:
            table = [{"采集时间": pd.to_datetime(r["collected_at"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"),
                      "状态": STATUS[r["status"]], "分数": (r["result"] or {}).get("score"),
                      "等级": (r["result"] or {}).get("risk_label", ""),
                      "模拟": (r["result"] or {}).get("is_simulated"), "任务编号": r["id"]} for r in records]
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
            export_params = {k: v for k, v in params.items() if k != "limit"}
            data = request("GET", "/api/export", params=export_params).content
            st.download_button("导出当前筛选结果 CSV", data, file_name=f"{subject}_records.csv", mime="text/csv")
    except (httpx.HTTPError, RuntimeError) as exc:
        st.error(str(exc))

with learn_tab:
    st.subheader("跟着一次请求读代码")
    st.markdown("""
    1. `dashboard.py` 提交数据，`app/main.py` 保存文件并创建任务。
    2. `app/pipeline.py` 在线程里执行处理，`app/db.py` 保存状态和结果。
    3. `app/adapters/features.py` 接主线一：输入采集样本，输出各模态特征。
    4. `app/adapters/predictor.py` 接主线二：输入特征和历史，输出统一的预测结果。
    5. 页面查询结果、展示趋势，`/api/export` 导出CSV。

    队友交付后，新增自己的适配类并修改 `config.toml` 的插件路径。
    字段约定与迁移步骤见项目内的 `docs/接口与替换.md`。
    """)
    st.json({"特征模块": config["extractor"], "预测模块": config["predictor"]})
