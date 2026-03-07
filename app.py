import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title("物体カウンター")
st.write("verson5 - 超高速化＆グラフ分析機能つき")

# モデルを一度だけ読み込んで使い回す
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# --- カウントする対象を選ぶUI ---
class_names_dict = model.names
class_names_list = list(class_names_dict.values())
selected_class_name = st.selectbox("カウントする対象を選んでください", class_names_list, index=0)
selected_class_id = list(class_names_dict.keys())[list(class_names_dict.values()).index(selected_class_name)]

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write(f"⏳ 「{selected_class_name}」を解析中...")
    
    temp_path = ""
    save_path = ""
    
    try:
        # 一時ファイルの作成
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        save_path = temp_path.replace(".mp4", "_output.mp4")
        
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
             st.error("エラー：動画ファイルの読み込みに失敗しました。")
        else:
            # 動画の情報を取得
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            # --- UIの空箱を準備 ---
            metric_placeholder = st.empty() 
            chart_placeholder = st.empty()  
            stframe = st.empty()            
            
            # --- 解析用の変数 ---
            max_target = 0
            counts_history = [] 
            frame_count = 0
            last_count = 0 # 最後に検知した数を覚えておく変数

            # 動画のフレームを1枚ずつ読み込むループ
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                res_frame = frame.copy() # 描画用のフレームを準備
                
                # 【高速化の要】1秒に1回（または最初のフレーム）だけAIを動かす
                if frame_count % fps == 0 or frame_count == 1:
                    results = model(frame)
                    
                    if results[0].boxes is not None:
                        classes = results[0].boxes.cls.cpu().numpy()
                        last_count = (classes == selected_class_id).sum()
                    else:
                        last_count = 0
                    
                    if last_count > max_target:
                        max_target = last_count

                    # AIが動いたタイミングでグラフと数値を更新
                    counts_history.append(last_count)
                    chart_placeholder.line_chart(counts_history)
                    metric_placeholder.metric(label=f"現在の {selected_class_name} 検知数", value=f"{last_count}")
                    
                    # バウンディングボックス（緑の枠）を描画
                    res_frame = results[0].plot()
                    
                    # 画面の映像も1秒に1回だけ更新する（ブラウザが重くなるのを防ぐため）
                    display_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(display_frame, channels="RGB", use_container_width=True)

                # 左上の文字は全フレームに書き込む（動画がカクカクしないように）
                cv2.putText(res_frame, f"{selected_class_name}: {last_count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # 出力用動画には全フレーム書き込む
                out.write(res_frame)
                    
            cap.release()
            out.release()
            
            st.success(f"解析完了！ この動画での最大同時検知数（{selected_class_name}）は {max_target} でした。")

            with open(save_path, "rb") as f:
                video_bytes = f.read()
                
            st.download_button(
                label="解析済み動画をダウンロード", 
                data=video_bytes, 
                file_name=f"analyzed_{selected_class_name}.mp4",
                mime="video/mp4"
            )

    except Exception as e:
        st.error(f"解析中にエラーが発生しました。\n詳細: {e}")

    finally:
        # お片付け処理
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
