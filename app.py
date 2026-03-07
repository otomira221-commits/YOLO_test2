import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title("物体カウンター") # タイトルも少し汎用的に変更
st.write("verson3 - クラス選択機能つき")

# モデルを一度だけ読み込んで使い回す
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# --- 【新機能】カウントする対象を選ぶUIを追加 ---
# YOLOモデルが知っている80種類の名前リストを取得
class_names_dict = model.names  # {0: 'person', 1: 'bicycle', ...} という辞書
class_names_list = list(class_names_dict.values()) # ['person', 'bicycle', ...] というリストに変換

# プルダウンメニューを表示（デフォルトは index=0 つまり 'person' に設定）
selected_class_name = st.selectbox("カウントする対象を選んでください", class_names_list, index=0)

# 選ばれた名前から、クラスIDを逆引きして取得する
# 例：'car' が選ばれたら、IDの 2 を取得する
selected_class_id = list(class_names_dict.keys())[list(class_names_dict.values()).index(selected_class_name)]
# ----------------------------------------------

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write(f"⏳ 「{selected_class_name}」を解析中...")
    
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            metric_placeholder = st.empty()
            stframe = st.empty()
            max_target = 0 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model(frame)
                
                # --- 【変更点】選んだクラスIDを数えるようにする ---
                target_count = 0
                if results[0].boxes is not None:
                    classes = results[0].boxes.cls.cpu().numpy()
                    # 前は == 0 だった部分を、選んだID (selected_class_id) に変更
                    target_count = (classes == selected_class_id).sum()
                
                if target_count > max_target:
                    max_target = target_count

                res_frame = results[0].plot()
                
                # 画面に書き込む文字も「Count」に変更
                cv2.putText(res_frame, f"{selected_class_name}: {target_count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                out.write(res_frame)
                metric_placeholder.metric(label=f"現在の {selected_class_name} 検知数", value=f"{target_count}")
                
                display_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                stframe.image(display_frame, channels="RGB", use_container_width=True)
                    
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
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
