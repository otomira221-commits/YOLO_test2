import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# 【変更点1】 以下の2行は削除（Streamlitのキャッシュ機能に任せてモデルの読み込みを高速化）
# st.cache_data.clear()
# st.cache_resource.clear()

st.title("人カウンター")
st.write("verson2")

# モデルを一度だけ読み込んで使い回す
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write("⏳ 解析中...")
    
    # 【変更点2】 お片付け（finally）でエラーにならないよう、先に変数を用意しておく
    temp_path = ""
    save_path = ""
    
    # ここから「エラーが起きるかもしれない処理」を try で囲む
    try:
        # アップロードされた動画を一時保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        save_path = temp_path.replace(".mp4", "_output.mp4")
        
        cap = cv2.VideoCapture(temp_path)
        
        # 動画が正しく読み込めたかチェック
        if not cap.isOpened():
             st.error("エラー：動画ファイルの読み込みに失敗しました。ファイルが破損している可能性があります。")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            # UIの準備
            metric_placeholder = st.empty()
            stframe = st.empty()
            max_people = 0 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # AI推論
                results = model(frame)
                
                person_count = 0
                if results[0].boxes is not None:
                    classes = results[0].boxes.cls.cpu().numpy()
                    person_count = (classes == 0).sum()
                
                if person_count > max_people:
                    max_people = person_count

                res_frame = results[0].plot()
                
                cv2.putText(res_frame, f"Count: {person_count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                out.write(res_frame)
                metric_placeholder.metric(label="現在の検知人数", value=f"{person_count} 人")
                
                display_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                stframe.image(display_frame, channels="RGB", use_container_width=True)
                    
            # 処理が終わったらOpenCVを閉じる
            cap.release()
            out.release()
            
            st.success(f"解析完了！ この動画での最大同時検知人数は {max_people} 人でした。")

            # ダウンロードボタンの表示（データをメモリに読み込んでから渡す）
            with open(save_path, "rb") as f:
                video_bytes = f.read()
                
            st.download_button(
                label="解析済み動画をダウンロード", 
                data=video_bytes, 
                file_name="analyzed_video.mp4",
                mime="video/mp4"
            )

    # 【変更点3】 予期せぬエラーが起きたらキャッチして画面に表示
    except Exception as e:
        st.error(f"解析中にエラーが発生しました。\n詳細: {e}")

    # 【変更点4】 成功しても失敗しても、最後に必ず実行される「お片付け」
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        
        # サーバーの容量圧迫を防ぐため、一時ファイルを確実に削除
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
