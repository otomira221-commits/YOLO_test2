import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title("人カウンター")
st.write("verson2")

model = YOLO('yolov8s.pt')
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write("⏳ 解析中...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    cap = cv2.VideoCapture(temp_path)
    
    # 保存用の設定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    save_path = temp_path.replace(".mp4", "_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # --- UIの準備 ---
    # 人数を表示するためのプレースホルダーを上に作る
    metric_placeholder = st.empty()
    stframe = st.empty()
    
    max_people = 0 # 動画内での最大人数を記録

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # AI推論
        results = model(frame)
        
        # --- 【新機能】人数をカウントする ---
        # results[0].boxes.cls に検出した物のIDが入っています。
        # 0番が「人(person)」です。
        person_count = 0
        if results[0].boxes is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            person_count = (classes == 0).sum() # クラスIDが0の数を合計
        
        # 最大人数を更新
        if person_count > max_people:
            max_people = person_count

        # 推論結果の画像を作成
        res_frame = results[0].plot()
        
        # 画像に現在の人数を直接書き込む（OpenCVを使用）
        cv2.putText(res_frame, f"Count: {person_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 保存と表示
        out.write(res_frame)
        metric_placeholder.metric(label="現在の検知人数", value=f"{person_count} 人")
        
        display_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB", use_container_width=True)
            
    cap.release()
    out.release()
    
    st.success(f"解析完了！ この動画での最大同時検知人数は {max_people} 人でした。")

    with open(save_path, "rb") as f:
        st.download_button("解析済み動画をダウンロード", f, "analyzed_video.mp4")
    

    os.remove(temp_path)
