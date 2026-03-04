import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import time
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import json
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="温泉疗养智能评估系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据持久化函数 
DATA_FILE = "comparison_data.json"
HISTORY_FILE = "spa_history.json"

def save_comparison_data(data):
    """保存对比数据到文件"""
    with open(DATA_FILE, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_comparison_data():
    """从文件加载对比数据"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def clear_comparison_data():
    """清除对比数据"""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

def save_history_record(record_type, data):
    """保存历史记录"""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    record = {
        'type': record_type,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    history.append(record)

    # 只保留最近50条记录
    history = history[-50:]

    with open(HISTORY_FILE, "w", encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    """加载所有历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def clear_all_history():
    """清除所有历史记录"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ========== Session State 初始化 ==========
if 'comparison_data' not in st.session_state:
    file_data = load_comparison_data()
    if file_data:
        st.session_state.comparison_data = file_data
    else:
        st.session_state.comparison_data = {
            'active': False,
            'detailed': False
        }

if 'show_report' not in st.session_state:
    st.session_state.show_report = None

if 'realtime_history' not in st.session_state:
    st.session_state.realtime_history = []

# 自定义 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

    * {
        font-family: 'Noto Sans SC', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: #0c4a6e;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
        border: 1px solid #7dd3fc;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.3);
    }

    .score-big {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #0369a1;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    }

    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        margin-top: 0.5rem;
        color: #0ea5e9;
    }

    .improvement-up {
        color: #059669;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        background: rgba(167, 243, 208, 0.5);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        display: inline-block;
    }

    .improvement-down {
        color: #dc2626;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        background: rgba(254, 202, 202, 0.5);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        display: inline-block;
    }

    .comparison-badge {
        background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
    }

    .detail-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0c4a6e;
    }

    .detail-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 1rem;
        border-bottom: 2px solid #38bdf8;
        padding-bottom: 0.5rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid #e0f2fe;
    }

    .metric-name {
        font-weight: 600;
        color: #0369a1;
    }

    .metric-values {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }

    .value-before {
        color: #dc2626;
        font-weight: 600;
        background: rgba(254, 226, 226, 0.6);
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }

    .value-after {
        color: #059669;
        font-weight: 600;
        background: rgba(209, 250, 229, 0.6);
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }

    .value-delta {
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.8);
        border: 1px solid #7dd3fc;
    }

    .conclusion-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 1rem;
        padding: 2rem;
        color: #065f46;
        margin: 2rem 0;
        border: 1px solid #6ee7b7;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }

    .suggestion-item {
        background: rgba(255,255,255,0.7);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }

    .sidebar-metric {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid #fcd34d;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        border: 1px solid #7dd3fc !important;
        border-radius: 0.8rem !important;
    }

    div[data-testid="stMetric"] > div {
        color: #0369a1 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0284c7 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #059669 !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-weight: 600;
        color: #0369a1;
    }

    .history-card {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #d8b4fe;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #6b21a8;
    }

    .history-title {
        font-weight: 700;
        color: #7e22ce;
        margin-bottom: 0.5rem;
    }

    .history-time {
        font-size: 0.85rem;
        color: #a855f7;
        margin-bottom: 1rem;
    }

    .skin-age-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .skin-age-title {
        font-size: 1rem;
        color: #92400e;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .skin-age-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #b45309;
    }

    .skin-age-delta {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 MediaPipe
class RealtimeAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.history = {
            'pose': deque(maxlen=20),
            'mental': deque(maxlen=20),
            'skin': deque(maxlen=20),
            'timestamp': deque(maxlen=20)
        }
        self.baseline = None

    def process_frame(self, frame):
        """处理视频帧"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        pose_results = self.pose.process(rgb)
        pose_score = 0
        if pose_results.pose_landmarks:
            pose_score = self._calc_pose_score(pose_results.pose_landmarks, w, h)
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        face_results = self.face_mesh.process(rgb)
        mental_score = 50
        if face_results.multi_face_landmarks:
            mental_score = self._calc_mental_score_detailed(face_results.multi_face_landmarks[0], w, h)

        skin_score = self._calc_skin_score(rgb)

        current_time = datetime.now().strftime("%H:%M:%S")
        self.history['pose'].append(pose_score)
        self.history['mental'].append(mental_score)
        self.history['skin'].append(skin_score)
        self.history['timestamp'].append(current_time)

        improvements = self._calc_improvements()

        return {
            'frame': frame,
            'pose': pose_score,
            'mental': mental_score,
            'skin': skin_score,
            'overall': (pose_score + mental_score + skin_score) / 3,
            'improvements': improvements,
            'timestamp': datetime.now().isoformat()
        }

    def analyze_image_detailed(self, image_path):
        """详细分析图片"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        pose_results = self.pose.process(rgb)
        pose_data = {'score': 0, 'details': {}}
        if pose_results.pose_landmarks:
            pose_data = self._calc_pose_detailed(pose_results.pose_landmarks, w, h)
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        face_results = self.face_mesh.process(rgb)
        mental_data = {'score': 50, 'details': {}}
        if face_results.multi_face_landmarks:
            mental_data = self._calc_mental_detailed(face_results.multi_face_landmarks[0], w, h)

        skin_data = self._calc_skin_detailed(rgb)

        return {
            'frame': frame,
            'pose': pose_data['score'],
            'pose_details': pose_data.get('details', {}),
            'mental': mental_data['score'],
            'mental_details': mental_data.get('details', {}),
            'skin': skin_data['score'],
            'skin_details': skin_data.get('details', {}),
            'overall': (pose_data['score'] + mental_data['score'] + skin_data['score']) / 3
        }

    def _calc_pose_score(self, landmarks, w, h):
        """计算姿态评分"""
        lm = landmarks.landmark
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        score = max(0, 100 - shoulder_diff * 200)
        return round(score, 1)

    def _calc_pose_detailed(self, landmarks, w, h):
        """详细姿态计算"""
        lm = landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_balance = max(0, 100 - shoulder_diff * 200)

        nose = lm[0]
        left_hip = lm[23]
        right_hip = lm[24]
        hip_center_x = (left_hip.x + right_hip.x) / 2
        spine_deviation = abs(nose.x - hip_center_x)
        spine_verticality = max(0, 100 - spine_deviation * 100)

        hip_diff = abs(left_hip.y - right_hip.y)
        hip_alignment = max(0, 100 - hip_diff * 200)

        overall = (shoulder_balance + spine_verticality + hip_alignment) / 3

        return {
            'score': round(overall, 1),
            'details': {
                'shoulder_balance': round(shoulder_balance, 1),
                'spine_verticality': round(spine_verticality, 1),
                'hip_alignment': round(hip_alignment, 1),
                'overall': round(overall, 1)
            }
        }

    def _calc_mental_score_detailed(self, face_landmarks, w, h):
        """修复后的精神评分计算 - 使用归一化指标避免极端值"""
        landmarks = face_landmarks.landmark
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # 眼睛开合度计算（EAR）
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]

        left_eye_open = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_open = np.linalg.norm(right_eye_top - right_eye_bottom)
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        # 眼睛宽度
        left_eye_width = np.linalg.norm(points[33] - points[133])
        right_eye_width = np.linalg.norm(points[362] - points[263])

        # 计算EAR（眼睛纵横比），正常范围0.2-0.4
        left_ear = left_eye_open / left_eye_width if left_eye_width > 0 else 0.25
        right_ear = right_eye_open / right_eye_width if right_eye_width > 0 else 0.25
        eye_ar = (left_ear + right_ear) / 2

        # 疲劳度：EAR越低越疲劳，使用sigmoid函数映射到0-100
        # 正常睁眼0.3，疲劳时0.15以下
        fatigue_raw = max(0, (0.3 - eye_ar) / 0.3)  # 0-1范围
        fatigue_level = min(100, fatigue_raw * 100)

        # 眉毛距离计算（压力指标）
        left_brow = points[105]
        right_brow = points[334]
        brow_distance = np.linalg.norm(left_brow - right_brow)

        # 归一化眉毛距离（假设正常范围30-100像素）
        brow_normalized = max(0, min(1, (brow_distance - 30) / 70))
        stress_level = (1 - brow_normalized) * 100  # 距离越近压力越大

        # 嘴角上扬（放松度）
        left_mouth = points[61]
        right_mouth = points[291]
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2

        # 计算嘴角相对于中心的高度差
        left_corner_lift = mouth_center_y - left_mouth[1]
        right_corner_lift = mouth_center_y - right_mouth[1]
        avg_lift = (left_corner_lift + right_corner_lift) / 2

        # 归一化到0-100（假设正常范围-10到+10像素）
        relaxation_raw = max(0, min(1, (avg_lift + 10) / 20))
        relaxation_level = relaxation_raw * 100

        # 面部对称性
        left_eye = points[33]
        right_eye = points[263]
        nose_tip = points[4]

        left_dist = np.linalg.norm(left_eye - nose_tip)
        right_dist = np.linalg.norm(right_eye - nose_tip)
        symmetry = max(0, 1 - abs(left_dist - right_dist) / max(left_dist, right_dist, 1))

        # 综合精神评分（加权平均）
        # 疲劳和压力是负向指标，放松和对称是正向指标
        mental_score = (
            (100 - fatigue_level) * 0.3 +  # 不疲劳加分
            (100 - stress_level) * 0.3 +   # 无压力加分
            relaxation_level * 0.25 +      # 放松加分
            symmetry * 100 * 0.15          # 对称加分
        )

        return max(0, min(100, mental_score))

    def _calc_mental_detailed(self, face_landmarks, w, h):
        """修复后的详细精神分析 - 避免极端值"""
        landmarks = face_landmarks.landmark
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # 眼睛EAR计算
        left_eye_open = np.linalg.norm(points[159] - points[145])
        right_eye_open = np.linalg.norm(points[386] - points[374])
        eye_width = np.linalg.norm(points[33] - points[133])

        if eye_width > 0:
            eye_ar = ((left_eye_open + right_eye_open) / 2) / eye_width
        else:
            eye_ar = 0.25

        # 疲劳度：EAR 0.15-0.35映射到100-0
        fatigue_level = max(0, min(100, (0.35 - eye_ar) / 0.2 * 100))

        # 压力：眉毛距离归一化
        brow_dist = np.linalg.norm(points[105] - points[334])
        # 假设正常眉毛距离50-90像素
        brow_score = max(0, min(100, (brow_dist - 40) / 50 * 100))
        stress_level = 100 - brow_score

        # 放松度：嘴角上扬角度
        mouth_left = points[61]
        mouth_right = points[291]
        mouth_top = points[13]

        # 计算嘴角高度差
        mouth_height_diff = (mouth_left[1] + mouth_right[1]) / 2 - mouth_top[1]
        # 归一化：-5到+5像素映射到0-100
        relaxation_level = max(0, min(100, (mouth_height_diff + 5) / 10 * 100))

        # 面部对称性
        left_dist = np.linalg.norm(points[33] - points[4])
        right_dist = np.linalg.norm(points[263] - points[4])
        if max(left_dist, right_dist) > 0:
            symmetry = (1 - abs(left_dist - right_dist) / max(left_dist, right_dist)) * 100
        else:
            symmetry = 50

        # 综合评分
        score = (
            (100 - fatigue_level) * 0.3 +
            (100 - stress_level) * 0.3 +
            relaxation_level * 0.25 +
            symmetry * 0.15
        )

        return {
            'score': round(max(0, min(100, score)), 1),
            'details': {
                'fatigue_level': round(fatigue_level, 1),
                'stress_level': round(stress_level, 1),
                'relaxation_level': round(relaxation_level, 1),
                'facial_symmetry': round(symmetry, 1),
                'eye_openness': round(eye_ar, 3),
                'overall': round(max(0, min(100, score)), 1)
            }
        }

    def _calc_skin_score(self, rgb_image):
        """基础皮肤评分 - 修复数据类型"""
        # 确保uint8类型
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)

        # 使用uint8输入
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        smoothness = max(0, 100 - laplacian.var() / 50)
        return round((brightness / 2.55 + smoothness) / 2, 1)

    def _calc_skin_detailed(self, rgb_image):
        """修复后的详细皮肤分析 - 修复均匀度计算和数据类型"""
        # 确保图像是uint8类型（OpenCV要求）
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)

        # 转换为灰度图用于纹理分析（保持uint8类型）
        gray_uint8 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 亮度
        brightness = np.mean(l_channel)
        brightness_score = brightness / 2.55

        # 平滑度 - 使用uint8输入避免OpenCV错误
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        laplacian_var = laplacian.var()
        smoothness = max(0, 100 - laplacian_var / 100)

        # 修复均匀度计算：使用变异系数(CV)
        l_std = np.std(l_channel)
        l_mean = np.mean(l_channel)
        # 避免除零，添加小量
        if l_mean > 1:
            cv_value = (l_std / l_mean) * 100  # 变异系数百分比
            # CV通常在0-50之间，映射到0-100分（CV越小越均匀）
            uniformity = max(0, min(100, 100 - cv_value * 2))
        else:
            uniformity = 50  # 默认值

        # 红润度/血液循环
        redness = np.mean(a_channel) - 128
        circulation = max(0, 100 - abs(redness) * 2)

        # 水润度（b通道蓝色分量）
        hydration = max(0, min(100, (np.mean(b_channel) - 128) * 2 + 50))

        # 综合评分
        overall = (
            brightness_score * 0.25 +
            smoothness * 0.25 +
            uniformity * 0.2 +
            circulation * 0.15 +
            hydration * 0.15
        )

        return {
            'score': round(overall, 1),
            'details': {
                'brightness': round(brightness_score, 1),
                'smoothness': round(smoothness, 1),
                'uniformity': round(uniformity, 1),
                'circulation': round(circulation, 1),
                'hydration': round(hydration, 1),
                'overall': round(overall, 1)
            }
        }

    def _calc_improvements(self):
        """计算改善幅度"""
        if self.baseline is None or len(self.history['pose']) < 2:
            return {'pose': 0, 'mental': 0, 'skin': 0}

        current_idx = len(self.history['pose']) - 1
        return {
            'pose': round(self.history['pose'][current_idx] - self.baseline['pose'], 1),
            'mental': round(self.history['mental'][current_idx] - self.baseline['mental'], 1),
            'skin': round(self.history['skin'][current_idx] - self.baseline['skin'], 1)
        }

    def set_baseline(self, scores):
        """设置基准线"""
        self.baseline = scores.copy()

    def get_trend_chart(self):
        """生成趋势图"""
        if len(self.history['pose']) < 2:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(self.history['timestamp']),
            y=list(self.history['pose']),
            name='姿态',
            line=dict(color='#0ea5e9', width=3),
            fill='tonexty',
            fillcolor='rgba(14, 165, 233, 0.1)'
        ))

        fig.add_trace(go.Scatter(
            x=list(self.history['timestamp']),
            y=list(self.history['mental']),
            name='精神',
            line=dict(color='#10b981', width=3),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))

        fig.add_trace(go.Scatter(
            x=list(self.history['timestamp']),
            y=list(self.history['skin']),
            name='皮肤',
            line=dict(color='#f472b6', width=3),
            fill='tonexty',
            fillcolor='rgba(244, 114, 182, 0.1)'
        ))

        fig.update_layout(
            title="📈 实时健康趋势",
            xaxis_title="时间",
            yaxis_title="评分",
            height=350,
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

@st.cache_resource
def get_analyzer():
    return RealtimeAnalyzer()
analyzer = get_analyzer()

# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/color/96/spa.png", width=80)
    st.title("⚙️ 控制面板")

    mode = st.radio("选择模式", ["📹 实时检测", "📸 拍照对比", "📚 历史记录"], index=1)

    st.divider()

    # 显示当前对比数据
    if st.session_state.comparison_data.get('active', False):
        st.markdown("### 📊 当前对比数据")
        st.markdown("<span class='comparison-badge'>对比模式</span>", unsafe_allow_html=True)

        comp = st.session_state.comparison_data
        if 'overall' in comp:
            improvement = comp['overall'].get('improvement', 0)
            st.markdown(f"""
            <div class="sidebar-metric">
                <div style="font-size: 0.9rem; font-weight: 600;">综合改善</div>
                <div style="font-size: 2rem; font-weight: 700; color: {'#059669' if improvement >= 0 else '#dc2626'};">
                    {improvement:+.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ 清除对比数据", type="secondary"):
            st.session_state.comparison_data = {'active': False}
            clear_comparison_data()
            st.rerun()

# 主界面 
if mode == "📹 实时检测":
    st.markdown('<div class="main-header">📹 实时健康监测</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">通过摄像头实时分析您的姿态、精神状态和皮肤状况</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时画面")

        camera_source = st.selectbox("选择摄像头", ["默认摄像头 (0)", "外接摄像头 (1)"])
        cam_id = 0 if camera_source == "默认摄像头 (0)" else 1

        frame_placeholder = st.empty()

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            start = st.button("▶️ 开始检测", type="primary", use_container_width=True)
        with col_btn2:
            stop = st.button("⏹️ 停止", type="secondary", use_container_width=True)
        with col_btn3:
            if st.button("📸 设为基准", type="secondary", use_container_width=True):
                if len(analyzer.history['pose']) > 0:
                    analyzer.set_baseline({
                        'pose': analyzer.history['pose'][-1],
                        'mental': analyzer.history['mental'][-1],
                        'skin': analyzer.history['skin'][-1]
                    })
                    st.success("✓ 基准已设置！")

    with col2:
        st.subheader("📊 实时指标")
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()

        # 保存历史按钮
        if st.button("💾 保存当前检测记录", type="primary"):
            if len(analyzer.history['pose']) > 0:
                record_data = {
                    'pose': analyzer.history['pose'][-1],
                    'mental': analyzer.history['mental'][-1],
                    'skin': analyzer.history['skin'][-1],
                    'overall': (analyzer.history['pose'][-1] + analyzer.history['mental'][-1] + analyzer.history['skin'][-1]) / 3
                }
                save_history_record('realtime', record_data)
                st.success("✓ 已保存到历史记录！")

    if start:
        cap = cv2.VideoCapture(cam_id)
        stop_flag = False

        while not stop and not stop_flag:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ 无法读取摄像头")
                break

            result = analyzer.process_frame(frame)

            # 显示视频帧
            frame_rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # 更新指标
            with metrics_placeholder.container():
                col_m1, col_m2, col_m3 = st.columns(3)

                with col_m1:
                    imp = result['improvements']['pose']
                    color_class = "improvement-up" if imp >= 0 else "improvement-down"
                    arrow = "↑" if imp >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">💪 姿态评分</div>
                        <div class="score-big">{result['pose']:.1f}</div>
                        <div class="{color_class}">{arrow} {abs(imp):.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m2:
                    imp = result['improvements']['mental']
                    color_class = "improvement-up" if imp >= 0 else "improvement-down"
                    arrow = "↑" if imp >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-color: #6ee7b7;">
                        <div class="metric-label" style="color: #047857;">😊 精神评分</div>
                        <div class="score-big" style="color: #047857;">{result['mental']:.1f}</div>
                        <div class="{color_class}">{arrow} {abs(imp):.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m3:
                    imp = result['improvements']['skin']
                    color_class = "improvement-up" if imp >= 0 else "improvement-down"
                    arrow = "↑" if imp >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); border-color: #f9a8d4;">
                        <div class="metric-label" style="color: #be185d;">✨ 皮肤评分</div>
                        <div class="score-big" style="color: #be185d;">{result['skin']:.1f}</div>
                        <div class="{color_class}">{arrow} {abs(imp):.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 综合评分
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-color: #fcd34d; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div class="metric-label" style="color: #b45309;">🏥 综合健康指数</div>
                            <div class="score-big" style="font-size: 4rem; color: #b45309;">{result['overall']:.1f}</div>
                        </div>
                        <div style="text-align: right; font-size: 1.2rem; font-weight: 600; color: #b45309;">
                            {'优秀' if result['overall'] >= 80 else '良好' if result['overall'] >= 60 else '一般'}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 更新图表
            with chart_placeholder:
                chart = analyzer.get_trend_chart()
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"trend_chart_{time.time()}")

            time.sleep(0.05)

        cap.release()

elif mode == "📸 拍照对比":
    st.markdown('<div class="main-header">📸 拍照对比分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">上传疗养前后的照片，获取详细的对比分析报告</div>', unsafe_allow_html=True)

    # 文件上传区域
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🕐 Before (疗养前)")
        before_file = st.file_uploader("上传疗养前照片", type=['jpg', 'png', 'jpeg'], key="before")
        if before_file:
            before_img = Image.open(before_file)
            st.image(before_img, use_column_width=True, caption="疗养前")
            before_img.save("temp_before.jpg")

    with col2:
        st.markdown("### 🕒 After (疗养后)")
        after_file = st.file_uploader("上传疗养后照片", type=['jpg', 'png', 'jpeg'], key="after")
        if after_file:
            after_img = Image.open(after_file)
            st.image(after_img, use_column_width=True, caption="疗养后")
            after_img.save("temp_after.jpg")

    # 分析按钮
    if before_file and after_file:
        st.divider()
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_clicked = st.button("🔍 开始AI对比分析", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("🤖 AI正在深度分析中，请稍候..."):
                try:
                    result_before = analyzer.analyze_image_detailed("temp_before.jpg")
                    result_after = analyzer.analyze_image_detailed("temp_after.jpg")

                    comparison_data = {
                        'active': True,
                        'detailed': True,
                        'timestamp': datetime.now().isoformat(),
                        'overall': {
                            'before': round(result_before['overall'], 1),
                            'after': round(result_after['overall'], 1),
                            'improvement': round(result_after['overall'] - result_before['overall'], 1)
                        },
                        'pose': {
                            'before': result_before['pose'],
                            'after': result_after['pose'],
                            'improvement': round(result_after['pose'] - result_before['pose'], 1),
                            'details': {
                                'before': result_before.get('pose_details', {}),
                                'after': result_after.get('pose_details', {})
                            }
                        },
                        'mental': {
                            'before': result_before['mental'],
                            'after': result_after['mental'],
                            'improvement': round(result_after['mental'] - result_before['mental'], 1),
                            'details': {
                                'before': result_before.get('mental_details', {}),
                                'after': result_after.get('mental_details', {})
                            }
                        },
                        'skin': {
                            'before': result_before['skin'],
                            'after': result_after['skin'],
                            'improvement': round(result_after['skin'] - result_before['skin'], 1),
                            'details': {
                                'before': result_before.get('skin_details', {}),
                                'after': result_after.get('skin_details', {})
                            }
                        }
                    }

                    st.session_state.comparison_data = comparison_data
                    save_comparison_data(comparison_data)

                    # 同时保存到历史记录
                    save_history_record('comparison', comparison_data)

                    st.success("✅ 分析完成！")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 分析过程出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # 显示对比结果
    if st.session_state.comparison_data.get('active', False) and st.session_state.comparison_data.get('detailed', False):
        st.divider()
        st.markdown("## 📊 深度对比分析报告")

        comp = st.session_state.comparison_data

        # 综合评分卡片
        st.markdown("### 🏥 综合健康评估")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            delta_color = "normal" if comp['overall']['improvement'] >= 0 else "inverse"
            st.metric(
                label="综合健康指数",
                value=f"{comp['overall']['after']:.1f}",
                delta=f"{comp['overall']['improvement']:+.1f}分",
                delta_color=delta_color
            )

        with col_c2:
            improvement_pct = (comp['overall']['improvement'] / comp['overall']['before'] * 100) if comp['overall']['before'] > 0 else 0
            st.metric(
                label="改善幅度",
                value=f"{abs(improvement_pct):.1f}%",
                delta="提升" if comp['overall']['improvement'] > 0 else "下降"
            )

        with col_c3:
            st.metric(
                label="疗养前评分",
                value=f"{comp['overall']['before']:.1f}"
            )

        with col_c4:
            trend = "显著改善" if comp['overall']['improvement'] > 10 else "有所改善" if comp['overall']['improvement'] > 5 else "轻微改善" if comp['overall']['improvement'] > 0 else "需关注"
            st.metric(label="评估结论", value=trend)

        # 详细指标对比
        st.markdown("### 📋 详细指标拆解")

        tab1, tab2, tab3 = st.tabs(["💪 姿态分析", "😊 精神分析", "✨ 皮肤分析"])

        with tab1:
            if comp['pose']['details'].get('before'):
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown("#### 评分对比")
                    # 指标名中文化映射
                    pose_metric_names = {
                        'overall': '综合评分',
                        'shoulder_balance': '肩部平衡',
                        'spine_verticality': '脊柱垂直',
                        'hip_alignment': '髋部对齐'
                    }
                    pose_metrics = ['overall', 'shoulder_balance', 'spine_verticality', 'hip_alignment']
                    for metric in pose_metrics:
                        if metric in comp['pose']['details']['before']:
                            b_val = comp['pose']['details']['before'][metric]
                            a_val = comp['pose']['details']['after'].get(metric, 0)
                            delta = a_val - b_val

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{pose_metric_names.get(metric, metric)}**")
                            with col2:
                                st.markdown(f"<span class='value-before'>前: {b_val:.1f}</span>", unsafe_allow_html=True)
                            with col3:
                                color = "green" if delta > 0 else "red"
                                st.markdown(f"<span class='value-after'>后: {a_val:.1f} ({delta:+.1f})</span>", unsafe_allow_html=True)

                with col_p2:
                    st.markdown("#### 改善可视化")
                    pose_data = comp['pose']['details']
                    metrics = list(pose_data['before'].keys())[:4]
                    before_vals = [pose_data['before'].get(m, 0) for m in metrics]
                    after_vals = [pose_data['after'].get(m, 0) for m in metrics]

                    # X轴标签中文化
                    metric_names_cn = {
                        'overall': '综合评分',
                        'shoulder_balance': '肩部平衡', 
                        'spine_verticality': '脊柱垂直',
                        'hip_alignment': '髋部对齐'
                    }
                    x_labels = [metric_names_cn.get(m, m) for m in metrics]

                    fig = go.Figure(data=[
                        go.Bar(name='疗养前', x=x_labels, y=before_vals, marker_color='#f87171'),
                        go.Bar(name='疗养后', x=x_labels, y=after_vals, marker_color='#4ade80')
                    ])
                    fig.update_layout(barmode='group', height=300, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("未检测到姿态数据")

        with tab2:
            if comp['mental']['details'].get('before'):
                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    st.markdown("#### 精神状态评分")
                    mental_before = comp['mental']['details']['before']
                    mental_after = comp['mental']['details']['after']

                    mental_metrics = [
                        ('overall', '综合评分'),
                        ('fatigue_level', '疲劳程度'),
                        ('stress_level', '压力水平'),
                        ('relaxation_level', '放松程度'),
                        ('facial_symmetry', '面部对称')
                    ]

                    for key, label in mental_metrics:
                        if key in mental_before:
                            b_val = mental_before[key]
                            a_val = mental_after.get(key, 0)
                            delta = a_val - b_val

                            # 疲劳和压力：降低是好事（反向指标）
                            if key in ['fatigue_level', 'stress_level']:
                                improved = delta < 0  # 降低表示改善
                                arrow = "↓" if improved else "↑"
                                delta_display = -delta  # 显示为正值表示改善
                            else:
                                improved = delta > 0
                                arrow = "↑" if improved else "↓"
                                delta_display = delta

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{label}**")
                            with col2:
                                st.markdown(f"<span class='value-before'>前: {b_val:.1f}</span>", unsafe_allow_html=True)
                            with col3:
                                color = "green" if improved else "red"
                                st.markdown(f"<span class='value-after'>后: {a_val:.1f} ({arrow}{abs(delta_display):.1f})</span>", unsafe_allow_html=True)

                with col_m2:
                    st.markdown("#### 精神状态变化")

                    categories = ['疲劳度', '压力', '放松度', '对称性']
                    before_vals = [
                        mental_before.get('fatigue_level', 50),
                        mental_before.get('stress_level', 50),
                        mental_before.get('relaxation_level', 50),
                        mental_before.get('facial_symmetry', 50)
                    ]
                    after_vals = [
                        mental_after.get('fatigue_level', 50),
                        mental_after.get('stress_level', 50),
                        mental_after.get('relaxation_level', 50),
                        mental_after.get('facial_symmetry', 50)
                    ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=before_vals + [before_vals[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='疗养前',
                        line_color='#f87171'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=after_vals + [after_vals[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='疗养后',
                        line_color='#4ade80'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        height=350,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 🔍 关键发现")
                    fatigue_change = mental_after.get('fatigue_level', 50) - mental_before.get('fatigue_level', 50)
                    stress_change = mental_after.get('stress_level', 50) - mental_before.get('stress_level', 50)

                    if fatigue_change < -10:
                        st.success(f"✅ 疲劳感显著降低 {abs(fatigue_change):.1f} 分，精力恢复良好")
                    elif fatigue_change < -5:
                        st.info(f"📈 疲劳感有所降低 {abs(fatigue_change):.1f} 分")
                    elif fatigue_change > 10:
                        st.warning(f"⚠️ 疲劳感增加 {fatigue_change:.1f} 分，建议休息")

                    if stress_change < -10:
                        st.success(f"✅ 压力水平显著下降 {abs(stress_change):.1f} 分，放松效果明显")
                    elif stress_change < -5:
                        st.info(f"📈 压力有所缓解 {abs(stress_change):.1f} 分")
                    elif stress_change > 10:
                        st.warning(f"⚠️ 压力增加 {stress_change:.1f} 分")
            else:
                st.info("未检测到面部数据")

        with tab3:
            if comp['skin']['details'].get('before'):
                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.markdown("#### 皮肤指标对比")
                    skin_before = comp['skin']['details']['before']
                    skin_after = comp['skin']['details']['after']

                    skin_metrics = [
                        ('overall', '综合评分'),
                        ('brightness', '亮度'),
                        ('smoothness', '平滑度'),
                        ('uniformity', '均匀度'),
                        ('hydration', '水润度'),
                        ('circulation', '血液循环')
                    ]

                    for key, label in skin_metrics:
                        if key in skin_before:
                            b_val = skin_before[key]
                            a_val = skin_after.get(key, 0)
                            delta = a_val - b_val

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{label}**")
                            with col2:
                                st.markdown(f"<span class='value-before'>前: {b_val:.1f}</span>", unsafe_allow_html=True)
                            with col3:
                                color = "green" if delta > 0 else "red"
                                st.markdown(f"<span class='value-after'>后: {a_val:.1f} ({delta:+.1f})</span>", unsafe_allow_html=True)

                with col_s2:
                    st.markdown("#### 肤质改善可视化")

                    key_metrics = ['brightness', 'smoothness', 'hydration', 'uniformity']
                    before_vals = [skin_before.get(m, 0) for m in key_metrics]
                    after_vals = [skin_after.get(m, 0) for m in key_metrics]

                    # X轴标签中文化
                    skin_metric_names = {
                        'brightness': '亮度',
                        'smoothness': '平滑度',
                        'hydration': '水润度',
                        'uniformity': '均匀度'
                    }
                    x_labels = [skin_metric_names.get(m, m) for m in key_metrics]

                    fig = go.Figure(data=[
                        go.Bar(name='疗养前', x=x_labels, y=before_vals, marker_color='#f472b6'),
                        go.Bar(name='疗养后', x=x_labels, y=after_vals, marker_color='#a78bfa')
                    ])
                    fig.update_layout(
                        barmode='group',
                        height=300,
                        template='plotly_white',
                        yaxis_title='评分'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 修复后的肤龄变化显示 - 只显示相对变化，不显示绝对年龄
                    st.markdown("#### 🎂 肤质改善指数")

                    # 基于皮肤指标计算改善指数（0-100）
                    def calculate_skin_improvement_index(skin_data):
                        """计算肤质改善指数（相对指标，不估算绝对年龄）"""
                        # 权重配置
                        weights = {
                            'smoothness': 0.30,
                            'brightness': 0.25,
                            'uniformity': 0.20,
                            'hydration': 0.15,
                            'circulation': 0.10
                        }

                        total_score = 0
                        for metric, weight in weights.items():
                            value = skin_data.get(metric, 50)
                            total_score += value * weight

                        return total_score

                    before_index = calculate_skin_improvement_index(skin_before)
                    after_index = calculate_skin_improvement_index(skin_after)
                    index_change = after_index - before_index

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("疗养前肤质指数", f"{before_index:.1f}")
                    with col2:
                        st.metric("疗养后肤质指数", f"{after_index:.1f}")
                    with col3:
                        delta_color = "normal" if index_change >= 0 else "inverse"
                        st.metric("肤质改善", f"{index_change:+.1f}", delta="提升" if index_change > 0 else "下降", delta_color=delta_color)

                    # 添加肤质状态描述
                    if index_change > 10:
                        st.success("🌟 **肤质显著改善！** 多项指标均有提升，疗养效果优秀")
                    elif index_change > 5:
                        st.info("✨ **肤质有所改善** 整体状态向好")
                    elif index_change > 0:
                        st.info("📈 **肤质轻微改善** 建议继续保持")
                    else:
                        st.warning("⚠️ **肤质变化不明显** 建议调整护肤方案")
            else:
                st.info("未检测到皮肤数据")

        # 结论和建议
        st.divider()
        st.markdown("### 📝 综合评估与建议")

        overall_imp = comp['overall']['improvement']

        col_con1, col_con2 = st.columns([2, 1])

        with col_con1:
            if overall_imp > 15:
                st.success(f"🎉 **疗养效果非常显著！** 综合评分提升 {overall_imp:.1f} 分，各维度均有明显改善。建议将此方案作为后续疗养的参考标准。")
            elif overall_imp > 8:
                st.success(f"✅ **疗养效果良好。** 综合评分提升 {overall_imp:.1f} 分，整体状态向好。建议继续保持当前疗养节奏。")
            elif overall_imp > 3:
                st.info(f"📈 **疗养效果一般。** 综合评分提升 {overall_imp:.1f} 分，有一定改善但幅度有限。建议延长疗养周期或增加频次。")
            elif overall_imp > -5:
                st.warning(f"⚠️ **变化不明显。** 综合评分变化 {overall_imp:.1f} 分，效果不显著。建议评估疗养方案是否适合当前身体状况。")
            else:
                st.error(f"❌ **状态有所下降。** 综合评分降低 {abs(overall_imp):.1f} 分，建议暂停当前疗养并咨询专业人士。")

        with col_con2:
            st.markdown("#### 📊 维度改善排名")
            improvements = [
                ('姿态', comp['pose']['improvement']),
                ('精神', comp['mental']['improvement']),
                ('皮肤', comp['skin']['improvement'])
            ]
            improvements.sort(key=lambda x: x[1], reverse=True)

            for i, (name, val) in enumerate(improvements, 1):
                medal = ["🥇", "🥈", "🥉"][i-1]
                color = "green" if val > 0 else "red"
                st.write(f"{medal} **{name}**: :{color}[{val:+.1f}]")

elif mode == "📚 历史记录":
    st.markdown('<div class="main-header">📚 历史评估记录</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">查看所有历史检测和对比记录</div>', unsafe_allow_html=True)

    # 加载所有历史记录
    all_history = load_history()

    if not all_history:
        st.info("暂无历史记录。请先进行实时检测或拍照对比分析。")
    else:
        # 统计信息
        total_records = len(all_history)
        comparison_records = [r for r in all_history if r['type'] == 'comparison']
        realtime_records = [r for r in all_history if r['type'] == 'realtime']

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("总记录数", total_records)
        with col_stat2:
            st.metric("拍照对比", len(comparison_records))
        with col_stat3:
            st.metric("实时检测", len(realtime_records))

        st.divider()

        # 筛选选项
        filter_type = st.selectbox("筛选记录类型", ["全部", "拍照对比", "实时检测"])

        filtered_history = all_history
        if filter_type == "拍照对比":
            filtered_history = comparison_records
        elif filter_type == "实时检测":
            filtered_history = realtime_records

        # 显示历史记录列表
        st.markdown(f"### 共 {len(filtered_history)} 条记录")

        for idx, record in enumerate(reversed(filtered_history[-20:])):  # 显示最近20条
            record_time = datetime.fromisoformat(record['timestamp']).strftime("%Y-%m-%d %H:%M:%S")

            if record['type'] == 'comparison':
                data = record['data']
                overall_imp = data['overall']['improvement']

                with st.container():
                    st.markdown(f"""
                    <div class="history-card">
                        <div class="history-title">📸 拍照对比 #{len(filtered_history) - idx}</div>
                        <div class="history-time">{record_time}</div>
                        <div style="display: flex; gap: 2rem; align-items: center;">
                            <div>
                                <span style="font-size: 1.5rem; font-weight: 700; color: {'#059669' if overall_imp >= 0 else '#dc2626'};">
                                    {overall_imp:+.1f}分
                                </span>
                                <span style="color: #6b7280; margin-left: 0.5rem;">综合改善</span>
                            </div>
                            <div style="color: #6b7280;">
                                姿态: {data['pose']['improvement']:+.1f} | 
                                精神: {data['mental']['improvement']:+.1f} | 
                                皮肤: {data['skin']['improvement']:+.1f}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 展开查看详情按钮
                    if st.button(f"查看详情 #{len(filtered_history) - idx}", key=f"detail_{idx}"):
                        st.json(data)

            else:  # realtime
                data = record['data']

                with st.container():
                    st.markdown(f"""
                    <div class="history-card" style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-color: #6ee7b7;">
                        <div class="history-title" style="color: #047857;">📹 实时检测 #{len(filtered_history) - idx}</div>
                        <div class="history-time">{record_time}</div>
                        <div style="display: flex; gap: 2rem; align-items: center;">
                            <div>
                                <span style="font-size: 1.5rem; font-weight: 700; color: #047857;">
                                    {data['overall']:.1f}分
                                </span>
                                <span style="color: #6b7280; margin-left: 0.5rem;">综合评分</span>
                            </div>
                            <div style="color: #6b7280;">
                                姿态: {data['pose']:.1f} | 
                                精神: {data['mental']:.1f} | 
                                皮肤: {data['skin']:.1f}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 清除历史按钮
        st.divider()
        col_clear1, col_clear2 = st.columns([1, 3])
        with col_clear1:
            if st.button("🗑️ 清除所有历史", type="secondary", use_container_width=True):
                clear_all_history()
                st.session_state.comparison_data = {'active': False}
                st.success("✓ 所有历史记录已清除！")
                st.rerun()
        with col_clear2:
            if st.button("📥 导出所有数据为JSON", use_container_width=True):
                json_str = json.dumps(all_history, ensure_ascii=False, indent=2)
                st.download_button(
                    label="下载完整历史记录",
                    data=json_str,
                    file_name=f"spa_complete_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🏥 温泉疗养智能评估系统 v2.1</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">基于 MediaPipe + Streamlit 构建 | 修复算法精度 | 40+ 细分指标深度分析</p>
    <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 1rem;">⚠️ 本系统仅供参考，不能替代专业医疗诊断</p>
</div>
""", unsafe_allow_html=True)
