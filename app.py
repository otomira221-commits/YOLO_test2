import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="マルチ物体カウンター", layout="wide")
st.title("マルチ物体カウンター")

@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

class_names_dict = model.names
class_names_list = list(class_names_dict.values())

with st.sidebar:
    st.header("⚙️ 設定パネル")
    st.write("verson9 - Plotlyインタラクティブグラフ版")
    
    selected_class_names = st.multiselect(
        "カウントする対象（複数選択可）", 
        class_names_list, 
        default=["person", "car"]
    )
    
    uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if not selected_class_names:
    st.warning("⚠️ サイドバーから少なくとも1つの対象を選んでください！")
else:
    selected_class_ids = [list(class_names_dict.keys())[list(class_names_dict.values()).index(name)] for name in selected_class_names]

    if uploaded_file is not None:
        st.write(f"⏳ {', '.join(selected_class_names)} を解析中...")
        
        metrics_container = st.empty()
        
        tab1, tab2 = st.tabs(["🎥 リアルタイム映像", "📊 データ推移グラフ"])
        
        with tab1:
            stframe = st.empty() 
            
        with tab2:
            chart_placeholder = st.empty() 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        temp_path = ""
        save_path = ""
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
            
            save_path = temp_path.replace(".mp4", "_output.mp4")
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                 st.error("エラー：動画ファイルの読み込みに失敗しました。")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 0fpsエラーを防ぐための安全対策
                safe_fps = max(1, fps)
                # グラフを更新する頻度（0.5秒に1回）
                chart_update_freq = max(1, int(safe_fps / 2)) 
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                out = cv2.VideoWriter(save_path, fourcc, safe_fps, (width, height))

                counts_history = {name: [] for name in selected_class_names}
                frame_count = 0
                current_counts = {name: 0 for name in selected_class_names}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    results = model(frame)
                    
                    if results[0].boxes is not None:
                        classes = results[0].boxes.cls.cpu().numpy()
                        for name, cls_id in zip(selected_class_names, selected_class_ids):
                            current_counts[name] = int((classes == cls_id).sum())
                    else:
                        for name in selected_class_names:
                            current_counts[name] = 0

                    for name in selected_class_names:
                        counts_history[name].append(current_counts[name])
                    
                    progress_ratio = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress_ratio)
                    status_text.text(f"処理中... {frame_count} / {total_frames} フレーム完了")

                    # --- 【新機能】Plotlyを使ったリッチなグラフの描画 ---
                    # 毎フレーム更新すると重いので、一定間隔 または 最後のフレームだけ更新する
                    if frame_count % chart_update_freq == 0 or frame_count == total_frames:
                        df = pd.DataFrame(counts_history)
                        # Plotly Expressで折れ線グラフを作成
                        fig = px.line(
                            df, 
                            title="リアルタイム検知数の推移",
                            labels={"index": "フレーム数", "value": "検知された数", "variable": "対象の種類"}
                        )
                        # グラフの余白を調整してスッキリ見せる
                        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        
                        # st.line_chart の代わりに st.plotly_chart を使う
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    # -----------------------------------------------------
                    
                    with metrics_container.container():
                        cols = st.columns(len(selected_class_names))
                        for i, name in enumerate(selected_class_names):
                            cols[i].metric(label=f"{name} の数", value=current_counts[name])
                    
                    res_frame = results[0].plot()
                    display_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                    
                    stframe.image(display_frame, channels="RGB", use_container_width=True)
                    out.write(res_frame)
                        
                cap.release()
                out.release()
                
                progress_bar.progress(1.0)
                status_text.text("処理が完了しました！")
                st.success("解析完了！")

                with open(save_path, "rb") as f:
                    video_bytes = f.read()
                    
                st.download_button(
                    label="解析済み動画をダウンロード", 
                    data=video_bytes, 
                    file_name="analyzed_plotly_multi.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"解析中にエラーが発生しました。\n詳細: {e}")

        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
