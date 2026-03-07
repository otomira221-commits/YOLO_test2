import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title("物体カウンター")
st.write("verson4 - リアルタイムグラフ分析機能つき")

@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

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

            # --- UIの準備（空箱の配置） ---
            metric_placeholder = st.empty() # 数値用
            chart_placeholder = st.empty()  # 【新機能】グラフ用
            stframe = st.empty()            # 動画用
            
            max_target = 0
            # 【新機能】フレームごとの検知数を記録するリスト
            counts_history = [] 
            # 【追加1】フレームを数えるカウンターと、最新の描画結果を覚えておく箱を用意
            frame_count = 0
            current_res_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
               # フレームを1つ読み込むごとにカウントアップ
                frame_count += 1
                
                # 【追加2】「1秒ごと（fpsで割り切れるフレーム数）」の時だけAIを動かす
                if frame_count % fps == 0:
                    
                # --- ここから下のAI推論は、1秒に1回しか実行されない ---
                    results = model(frame)
                    
                    target_count = 0
                    if results[0].boxes is not None:
                        classes = results[0].boxes.cls.cpu().numpy()
                        target_count = (classes == selected_class_id).sum()
                    
                    if target_count > max_target:
                        max_target = target_count

                    counts_history.append(target_count)
                    
                    # 変数名を res_frame から current_res_frame に変更
                    current_res_frame = results[0].plot()
                    cv2.putText(current_res_frame, f"{selected_class_name}: {target_count}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                    # UIの更新も1秒に1回だけ行う
                    metric_placeholder.metric(label=f"現在の {selected_class_name} 検知数", value=f"{target_count}")
                    chart_placeholder.line_chart(counts_history)
                    
                    display_frame = cv2.cvtColor(current_res_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(display_frame, channels="RGB", use_container_width=True)
                    # --------------------------------------------------------

                # 【追加3】出力用動画の保存処理（これは毎フレーム必ず行う！）
                # もしすでにAIが描画した画像(current_res_frame)があれば、それを書き込む
                # なければ、AIがまだ動いていない最初の1秒間なので、元の動画のフレームをそのまま書き込む
                if current_res_frame is not None:
                    out.write(current_res_frame)
                else:
                    out.write(frame)
                
                # --- UIの更新 ---
                metric_placeholder.metric(label=f"現在の {selected_class_name} 検知数", value=f"{target_count}")
                
                # 【新機能】履歴リストを使って折れ線グラフを更新
                chart_placeholder.line_chart(counts_history)
                
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

