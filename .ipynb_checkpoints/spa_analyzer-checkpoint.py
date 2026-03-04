import mediapipe as mp
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import euclidean

@dataclass
class PoseMetrics:
    """姿态指标数据类 - 细化版"""
    # 基础指标
    shoulder_balance: float
    spine_verticality: float
    hip_alignment: float
    knee_symmetry: float
    overall_score: float
    
    # 新增细分指标
    shoulder_angle: float  # 肩部倾斜角度
    head_position: float   # 头部前倾/后仰评分
    body_center_line: float  # 身体中线偏移
    weight_distribution: float  # 重心分布
    
    # 详细分析
    left_shoulder_y: float
    right_shoulder_y: float
    spine_deviation_px: float
    
    key_points: Dict[str, Tuple[float, float, float]]

@dataclass
class FaceMetrics:
    """面部指标数据类 - 细化版"""
    # 基础指标
    facial_symmetry: float
    eye_openness: float
    mouth_expression: float
    forehead_tension: float
    mental_state_score: float
    expression_label: str
    
    # 新增细分指标
    left_eye_openness: float
    right_eye_openness: float
    eye_bag_severity: float  # 眼袋严重程度
    mouth_corner_lift: float  # 嘴角上扬程度
    brow_furrow_depth: float  # 眉间皱纹深度
    jaw_tension: float  # 下颌紧张度
    
    # 情绪细分
    stress_level: float  # 压力水平 0-100
    fatigue_level: float  # 疲劳程度 0-100
    relaxation_level: float  # 放松程度 0-100
    
    # 原始数据
    eye_aspect_ratio: float
    mouth_aspect_ratio: float

@dataclass
class SkinMetrics:
    """皮肤指标数据类 - 细化版"""
    # 基础指标
    brightness: float
    smoothness: float
    redness: float
    spot_score: float
    estimated_age: int
    overall_score: float
    
    # 新增细分指标 - 色彩分析
    luminance_mean: float  # 亮度均值
    luminance_std: float   # 亮度标准差（均匀度）
    color_uniformity: float  # 肤色均匀度
    
    # 纹理分析
    texture_score: float   # 纹理细腻度
    pore_visibility: float  # 毛孔可见度
    wrinkle_score: float   # 皱纹评分
    
    # 健康指标
    circulation_score: float  # 血液循环评分（基于红润度）
    hydration_score: float    # 水润度评分
    clarity_score: float      # 通透度评分
    
    # 区域分析
    t_zone_oiliness: float    # T区出油度
    cheek_moisture: float     # 脸颊水润度
    
    # 原始数据
    lab_channels: Dict[str, float]
    analysis_region: Tuple[int, int]

class MediaPipeSpaAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose_analyzer = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        print("✓ MediaPipe 分析器初始化完成")
    
    def analyze_image(self, image_path: str) -> Dict:
        """完整分析一张图片"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        result = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'image_shape': (h, w),
            'pose': {'detected': False},
            'face': {'detected': False},
            'skin': None,
            'details': {}  # 新增详细说明
        }
        
        print("  [1/3] 分析身体姿态...")
        pose_result = self._analyze_pose(image_rgb)
        result['pose'] = pose_result
        if pose_result['detected']:
            result['details']['pose'] = self._generate_pose_description(pose_result['metrics'])
        
        print("  [2/3] 分析面部特征...")
        face_result = self._analyze_face(image_rgb)
        result['face'] = face_result
        if face_result['detected']:
            result['details']['face'] = self._generate_face_description(face_result['metrics'])
        
        print("  [3/3] 分析皮肤健康...")
        if face_result.get('detected') and face_result.get('face_location'):
            skin_result = self._analyze_skin_detailed(image_rgb, face_result['face_location'])
        else:
            skin_result = self._analyze_skin_detailed(image_rgb, None)
        result['skin'] = skin_result
        result['details']['skin'] = self._generate_skin_description(skin_result)
        
        result['overall_wellness_score'] = self._calculate_wellness_score(result)
        result['overall_assessment'] = self._generate_overall_assessment(result)
        
        return result
    
    def _analyze_pose(self, image_rgb: np.ndarray) -> Dict:
        """姿态分析"""
        results = self.pose_analyzer.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'detected': False, 'message': '未检测到人体'}
        
        landmarks = results.pose_landmarks.landmark
        h, w = image_rgb.shape[:2]
        
        key_points = {}
        for idx, landmark in enumerate(landmarks):
            name = self.mp_pose.PoseLandmark(idx).name
            key_points[name] = (landmark.x * w, landmark.y * h, landmark.visibility)
        
        metrics = self._calculate_pose_metrics_enhanced(key_points, w, h)
        
        return {
            'detected': True,
            'key_points': key_points,
            'metrics': metrics,
            'landmarks': results.pose_landmarks
        }
    
    def _calculate_pose_metrics_enhanced(self, kp: Dict, img_w: int, img_h: int) -> PoseMetrics:
        """计算详细的姿态指标"""
        # 基础点
        left_shoulder = kp.get('LEFT_SHOULDER', (0, 0, 0))
        right_shoulder = kp.get('RIGHT_SHOULDER', (0, 0, 0))
        left_hip = kp.get('LEFT_HIP', (0, 0, 0))
        right_hip = kp.get('RIGHT_HIP', (0, 0, 0))
        nose = kp.get('NOSE', (0, 0, 0))
        left_ear = kp.get('LEFT_EAR', (0, 0, 0))
        right_ear = kp.get('RIGHT_EAR', (0, 0, 0))
        left_knee = kp.get('LEFT_KNEE', (0, 0, 0))
        right_knee = kp.get('RIGHT_KNEE', (0, 0, 0))
        left_ankle = kp.get('LEFT_ANKLE', (0, 0, 0))
        right_ankle = kp.get('RIGHT_ANKLE', (0, 0, 0))
        
        # 1. 肩部平衡（考虑角度）
        shoulder_diff_y = abs(left_shoulder[1] - right_shoulder[1])
        shoulder_diff_x = abs(left_shoulder[0] - right_shoulder[0])
        shoulder_angle = np.degrees(np.arctan2(shoulder_diff_y, shoulder_diff_x))
        shoulder_balance = max(0, 100 - shoulder_angle * 2)
        
        # 2. 脊柱垂直度（更精确计算）
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        spine_deviation_px = abs(nose[0] - hip_center_x)
        spine_deviation_percent = (spine_deviation_px / img_w) * 100
        spine_verticality = max(0, 100 - spine_deviation_percent * 5)
        
        # 3. 头部位置（前倾检测）
        ear_center_x = (left_ear[0] + right_ear[0]) / 2
        ear_center_y = (left_ear[1] + right_ear[1]) / 2
        head_offset = abs(nose[0] - ear_center_x)
        head_position = max(0, 100 - (head_offset / img_w) * 200)
        
        # 4. 髋部对齐
        hip_diff_y = abs(left_hip[1] - right_hip[1])
        hip_alignment = max(0, 100 - hip_diff_y * 3)
        
        # 5. 膝盖对称
        knee_diff_y = abs(left_knee[1] - right_knee[1])
        knee_symmetry = max(0, 100 - knee_diff_y * 2)
        
        # 6. 身体中线偏移
        body_center_deviation = abs((nose[0] - img_w/2) / (img_w/2)) * 100
        body_center_line = max(0, 100 - body_center_deviation)
        
        # 7. 重心分布（基于脚踝位置）
        ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
        weight_distribution = max(0, 100 - abs(ankle_center_x - img_w/2) / (img_w/2) * 100)
        
        # 综合评分（加权）
        overall = (
            shoulder_balance * 0.25 +
            spine_verticality * 0.25 +
            hip_alignment * 0.15 +
            knee_symmetry * 0.15 +
            head_position * 0.10 +
            body_center_line * 0.05 +
            weight_distribution * 0.05
        )
        
        return PoseMetrics(
            shoulder_balance=round(shoulder_balance, 2),
            spine_verticality=round(spine_verticality, 2),
            hip_alignment=round(hip_alignment, 2),
            knee_symmetry=round(knee_symmetry, 2),
            overall_score=round(overall, 2),
            shoulder_angle=round(shoulder_angle, 2),
            head_position=round(head_position, 2),
            body_center_line=round(body_center_line, 2),
            weight_distribution=round(weight_distribution, 2),
            left_shoulder_y=round(left_shoulder[1], 2),
            right_shoulder_y=round(right_shoulder[1], 2),
            spine_deviation_px=round(spine_deviation_px, 2),
            key_points=kp
        )
    
    def _analyze_face(self, image_rgb: np.ndarray) -> Dict:
        """面部分析"""
        h, w = image_rgb.shape[:2]
        
        detection_results = self.face_detection.process(image_rgb)
        if not detection_results.detections:
            return {'detected': False, 'message': '未检测到面部'}
        
        detection = detection_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_location = {
            'x': int(bbox.xmin * w),
            'y': int(bbox.ymin * h),
            'width': int(bbox.width * w),
            'height': int(bbox.height * h)
        }
        
        mesh_results = self.face_mesh.process(image_rgb)
        if not mesh_results.multi_face_landmarks:
            return {'detected': False, 'message': '面部网格检测失败'}
        
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        face_points = np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks])
        metrics = self._calculate_face_metrics_enhanced(face_points, face_location)
        
        return {
            'detected': True,
            'face_location': face_location,
            'landmarks': face_points,
            'metrics': metrics
        }
    
    def _calculate_face_metrics_enhanced(self, points: np.ndarray, face_loc: Dict) -> FaceMetrics:
        """计算详细的面部指标"""
        # 关键点索引（MediaPipe Face Mesh）
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 263
        NOSE_TIP = 4
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        MOUTH_TOP = 13
        MOUTH_BOTTOM = 14
        LEFT_BROW = 105
        RIGHT_BROW = 334
        BROW_CENTER = 9
        CHIN = 152
        LEFT_JAW = 234
        RIGHT_JAW = 454
        
        # 1. 面部对称性（多维度）
        left_eye = points[LEFT_EYE_CORNER]
        right_eye = points[RIGHT_EYE_CORNER]
        nose_tip = points[NOSE_TIP]
        chin = points[CHIN]
        
        # 眼鼻距离对称
        left_dist = euclidean(left_eye[:2], nose_tip[:2])
        right_dist = euclidean(right_eye[:2], nose_tip[:2])
        eye_nose_symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
        
        # 左右脸高度对称
        left_face_height = euclidean(left_eye[:2], chin[:2])
        right_face_height = euclidean(right_eye[:2], chin[:2])
        face_height_symmetry = 1 - abs(left_face_height - right_face_height) / max(left_face_height, right_face_height)
        
        facial_symmetry = (eye_nose_symmetry + face_height_symmetry) / 2 * 100
        
        # 2. 眼睛开合度（左右分别计算）
        left_eye_open = euclidean(points[LEFT_EYE_TOP], points[LEFT_EYE_BOTTOM])
        right_eye_open = euclidean(points[RIGHT_EYE_TOP], points[RIGHT_EYE_BOTTOM])
        eye_openness = (left_eye_open + right_eye_open) / 2
        
        # 眼宽高比（判断眼睛状态）
        left_eye_width = euclidean(points[33], points[133])
        right_eye_width = euclidean(points[362], points[263])
        left_ear = left_eye_open / left_eye_width if left_eye_width > 0 else 0
        right_ear = right_eye_open / right_eye_width if right_eye_width > 0 else 0
        eye_aspect_ratio = (left_ear + right_ear) / 2
        
        # 3. 眼袋检测（下眼睑到脸颊的距离）
        left_eye_bag = euclidean(points[145], points[2])  # 近似
        right_eye_bag = euclidean(points[374], points[2])
        eye_bag_severity = min(100, (left_eye_bag + right_eye_bag) / 2 * 50)
        
        # 4. 嘴角上扬程度
        left_mouth = points[LEFT_MOUTH]
        right_mouth = points[RIGHT_MOUTH]
        mouth_center = points[MOUTH_TOP]
        
        mouth_width = euclidean(left_mouth, right_mouth)
        mouth_height = euclidean(points[MOUTH_TOP], points[MOUTH_BOTTOM])
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # 嘴角相对于水平线的上扬角度
        mouth_slope = (right_mouth[1] - left_mouth[1]) / (right_mouth[0] - left_mouth[0] + 1e-6)
        mouth_corner_lift = np.degrees(np.arctan(mouth_slope))
        mouth_expression = mouth_corner_lift  # 正值表示上扬
        
        # 5. 眉间皱纹（眉毛距离）
        brow_distance = euclidean(points[LEFT_BROW], points[RIGHT_BROW])
        brow_furrow_depth = max(0, 100 - brow_distance * 2)
        
        # 6. 额头紧张度
        forehead = points[BROW_CENTER]
        forehead_tension = max(0, 100 - brow_distance / 2)
        
        # 7. 下颌紧张度（下颌角度）
        jaw_angle = abs(points[LEFT_JAW][1] - points[RIGHT_JAW][1])
        jaw_tension = min(100, jaw_angle * 2)
        
        # 8. 精神状态评分（多因子模型）
        # 疲劳指标：眼睛开合度小 + 眼袋严重
        fatigue_level = min(100, (100 - eye_openness * 10) * 0.5 + eye_bag_severity * 0.5)
        
        # 压力指标：眉间皱纹 + 下颌紧张
        stress_level = min(100, brow_furrow_depth * 0.6 + jaw_tension * 0.4)
        
        # 放松指标：嘴角上扬 + 眼睛自然睁开
        relaxation_level = min(100, max(0, mouth_expression * 5) + eye_openness * 5)
        
        # 综合精神评分
        mental_state_score = max(0, min(100, 
            100 - fatigue_level * 0.4 - stress_level * 0.4 + relaxation_level * 0.2
        ))
        
        # 表情标签
        if stress_level > 60:
            expression = '紧张焦虑'
        elif fatigue_level > 60:
            expression = '疲劳倦怠'
        elif mouth_expression > 5:
            expression = '放松愉悦'
        elif relaxation_level > 50:
            expression = '轻松舒适'
        else:
            expression = '平静自然'
        
        return FaceMetrics(
            facial_symmetry=round(facial_symmetry, 2),
            eye_openness=round(eye_openness, 2),
            mouth_expression=round(mouth_expression, 2),
            forehead_tension=round(forehead_tension, 2),
            mental_state_score=round(mental_state_score, 2),
            expression_label=expression,
            left_eye_openness=round(left_eye_open, 2),
            right_eye_openness=round(right_eye_open, 2),
            eye_bag_severity=round(eye_bag_severity, 2),
            mouth_corner_lift=round(mouth_corner_lift, 2),
            brow_furrow_depth=round(brow_furrow_depth, 2),
            jaw_tension=round(jaw_tension, 2),
            stress_level=round(stress_level, 2),
            fatigue_level=round(fatigue_level, 2),
            relaxation_level=round(relaxation_level, 2),
            eye_aspect_ratio=round(eye_aspect_ratio, 3),
            mouth_aspect_ratio=round(mouth_aspect_ratio, 3)
        )
    
    def _analyze_skin_detailed(self, image_rgb: np.ndarray, face_loc: Optional[Dict]) -> Dict:
        """详细的皮肤分析"""
        if face_loc:
            x, y = face_loc['x'], face_loc['y']
            w, h = face_loc['width'], face_loc['height']
            # 扩展区域包含额头和下巴
            y_ext = max(0, y - h//3)
            h_ext = min(image_rgb.shape[0] - y_ext, h + h//2)
            x_ext = max(0, x - w//10)
            w_ext = min(image_rgb.shape[1] - x_ext, w + w//5)
            skin_region = image_rgb[y_ext:y_ext+h_ext, x_ext:x_ext+w_ext]
        else:
            skin_region = image_rgb
            x_ext, y_ext, w_ext, h_ext = 0, 0, image_rgb.shape[1], image_rgb.shape[0]
        
        if skin_region.size == 0:
            return self._empty_skin_metrics()
        
        # 色彩空间转换
        lab = cv2.cvtColor(skin_region, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        
        hsv = cv2.cvtColor(skin_region, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0].astype(np.float32)
        s_channel = hsv[:, :, 1].astype(np.float32)
        v_channel = hsv[:, :, 2].astype(np.float32)
        
        gray = cv2.cvtColor(skin_region, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 1. 亮度分析
        luminance_mean = np.mean(l_channel)
        luminance_std = np.std(l_channel)
        brightness = luminance_mean
        
        # 2. 均匀度（标准差越小越均匀）
        color_uniformity = max(0, 100 - luminance_std * 2)
        
        # 3. 纹理分析（拉普拉斯算子）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        texture_score = max(0, 100 - laplacian_var / 100)
        smoothness = texture_score
        
        # 4. 毛孔检测（高频成分）
        from scipy import ndimage
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        pore_visibility = min(100, np.mean(sobel_mag) * 5)
        
        # 5. 红润度/血液循环（a通道）
        redness_raw = np.mean(a_channel) - 128
        circulation_score = max(0, 100 - abs(redness_raw) * 2)
        redness = redness_raw
        
        # 6. 色斑检测（亮度极值）
        _, bright_spots = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
        _, dark_spots = cv2.threshold(l_channel, 50, 255, cv2.THRESH_BINARY_INV)
        spot_score = max(0, 100 - (np.sum(bright_spots) + np.sum(dark_spots)) / (l_channel.size * 255) * 1000)
        
        # 7. 水润度（b通道蓝色分量）
        hydration_score = min(100, max(0, (np.mean(b_channel) - 128) * 2 + 50))
        
        # 8. 通透度（饱和度）
        clarity_score = max(0, 100 - np.mean(s_channel) / 2.55)
        
        # 9. 皱纹检测（边缘密度）
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        wrinkle_density = np.sum(edges > 0) / edges.size * 100
        wrinkle_score = max(0, 100 - wrinkle_density * 5)
        
        # 10. T区和脸颊分析（简化版）
        h_region, w_region = skin_region.shape[:2]
        t_zone = skin_region[:h_region//3, w_region//3:2*w_region//3]
        cheek = skin_region[h_region//3:2*h_region//3, :w_region//3]
        
        if t_zone.size > 0 and cheek.size > 0:
            t_zone_lab = cv2.cvtColor(t_zone, cv2.COLOR_RGB2LAB)
            cheek_lab = cv2.cvtColor(cheek, cv2.COLOR_RGB2LAB)
            t_zone_oiliness = np.mean(t_zone_lab[:, :, 1]) - 128
            cheek_moisture = np.mean(cheek_lab[:, :, 2]) - 128
        else:
            t_zone_oiliness = 0
            cheek_moisture = 0
        
        # 11. 肤龄估算（多因子）
        age_factors = (
            (100 - smoothness) * 0.3 +
            (100 - wrinkle_score) * 0.3 +
            (100 - color_uniformity) * 0.2 +
            pore_visibility * 0.1 +
            (100 - clarity_score) * 0.1
        )
        estimated_age = int(20 + age_factors * 0.4)
        
        # 综合皮肤评分（加权）
        overall = (
            (brightness / 2.55) * 0.20 +      # 亮度
            smoothness * 0.25 +                # 平滑度
            color_uniformity * 0.20 +          # 均匀度
            circulation_score * 0.15 +         # 血液循环
            hydration_score * 0.10 +           # 水润度
            clarity_score * 0.10               # 通透度
        )
        
        return {
            'brightness': round(brightness, 2),
            'smoothness': round(smoothness, 2),
            'redness': round(redness, 2),
            'spot_score': round(spot_score, 2),
            'estimated_age': estimated_age,
            'overall_score': round(overall, 2),
            'luminance_mean': round(luminance_mean, 2),
            'luminance_std': round(luminance_std, 2),
            'color_uniformity': round(color_uniformity, 2),
            'texture_score': round(texture_score, 2),
            'pore_visibility': round(pore_visibility, 2),
            'wrinkle_score': round(wrinkle_score, 2),
            'circulation_score': round(circulation_score, 2),
            'hydration_score': round(hydration_score, 2),
            'clarity_score': round(clarity_score, 2),
            't_zone_oiliness': round(t_zone_oiliness, 2),
            'cheek_moisture': round(cheek_moisture, 2),
            'lab_channels': {
                'L_mean': round(float(np.mean(l_channel)), 2),
                'A_mean': round(float(np.mean(a_channel)), 2),
                'B_mean': round(float(np.mean(b_channel)), 2)
            },
            'analysis_region': (skin_region.shape[1], skin_region.shape[0])
        }
    
    def _empty_skin_metrics(self) -> Dict:
        """返回空皮肤指标"""
        return {
            'brightness': 0, 'smoothness': 0, 'redness': 0, 'spot_score': 0,
            'estimated_age': 0, 'overall_score': 0, 'luminance_mean': 0,
            'luminance_std': 0, 'color_uniformity': 0, 'texture_score': 0,
            'pore_visibility': 0, 'wrinkle_score': 0, 'circulation_score': 0,
            'hydration_score': 0, 'clarity_score': 0, 't_zone_oiliness': 0,
            'cheek_moisture': 0, 'lab_channels': {}, 'analysis_region': (0, 0)
        }
    
    def _calculate_wellness_score(self, result: Dict) -> float:
        """计算综合健康评分"""
        scores = []
        weights = []
        
        if result['pose'].get('detected'):
            scores.append(result['pose']['metrics'].overall_score)
            weights.append(0.25)
        
        if result['face'].get('detected'):
            scores.append(result['face']['metrics'].mental_state_score)
            weights.append(0.35)
        
        if result['skin']:
            scores.append(result['skin']['overall_score'])
            weights.append(0.40)
        
        if not scores:
            return 0
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        return round(sum(s * w for s, w in zip(scores, weights)), 2)
    
    def _generate_pose_description(self, metrics: PoseMetrics) -> str:
        """生成姿态分析描述"""
        issues = []
        if metrics.shoulder_balance < 80:
            issues.append(f"肩部倾斜{metrics.shoulder_angle:.1f}度")
        if metrics.spine_verticality < 80:
            issues.append(f"脊柱偏移{metrics.spine_deviation_px:.1f}像素")
        if metrics.head_position < 80:
            issues.append("头部前倾")
        if metrics.hip_alignment < 80:
            issues.append("髋部不正")
        
        if not issues:
            return "姿态良好，身体对称性佳"
        return "; ".join(issues)
    
    def _generate_face_description(self, metrics: FaceMetrics) -> str:
        """生成面部分析描述"""
        states = []
        if metrics.stress_level > 50:
            states.append(f"压力水平较高({metrics.stress_level:.0f}%)")
        if metrics.fatigue_level > 50:
            states.append(f"疲劳明显({metrics.fatigue_level:.0f}%)")
        if metrics.relaxation_level > 50:
            states.append(f"放松度良好({metrics.relaxation_level:.0f}%)")
        
        details = f"对称性{metrics.facial_symmetry:.0f}%, 嘴角上扬{metrics.mouth_corner_lift:.1f}度"
        
        if not states:
            return f"精神状态平衡; {details}"
        return "; ".join(states) + f"; {details}"
    
    def _generate_skin_description(self, metrics: Dict) -> str:
        """生成皮肤分析描述"""
        issues = []
        if metrics['brightness'] < 100:
            issues.append("肤色偏暗")
        if metrics['smoothness'] < 70:
            issues.append("纹理粗糙")
        if metrics['color_uniformity'] < 70:
            issues.append("肤色不均")
        if metrics['hydration_score'] < 60:
            issues.append("缺水")
        if metrics['wrinkle_score'] < 60:
            issues.append("有细纹")
        
        if not issues:
            return f"肤质良好，估计肤龄{metrics['estimated_age']}岁"
        return "; ".join(issues) + f"; 估计肤龄{metrics['estimated_age']}岁"
    
    def _generate_overall_assessment(self, result: Dict) -> str:
        """生成整体评估"""
        score = result['overall_wellness_score']
        if score >= 85:
            return "优秀 - 身心状态极佳"
        elif score >= 70:
            return "良好 - 整体状态健康"
        elif score >= 60:
            return "一般 - 有改善空间"
        elif score >= 50:
            return "较差 - 建议调理"
        else:
            return "差 - 需要关注"
    
    def visualize(self, result: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """可视化"""
        image = cv2.imread(result['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 创建画布（原图 + 信息面板）
        panel_width = 450
        canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        canvas[:, :w] = image
        canvas[:, w:] = [30, 30, 30]  # 深灰背景
        
        # 绘制骨骼和面部标记
        if result['pose'].get('detected'):
            self._draw_pose_on_image(canvas[:, :w], result['pose']['landmarks'])
        
        if result['face'].get('detected'):
            self._draw_face_on_image(canvas[:, :w], result['face']['landmarks'])
        
        # 绘制信息面板
        pil_image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_image)
        
        # 字体设置
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 24)
            font_header = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
            font_text = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
            font_small = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        panel_x = w + 20
        y_offset = 30
        
        # 标题
        draw.text((panel_x, y_offset), "温泉疗养分析报告", fill=(0, 255, 255), font=font_title)
        y_offset += 45
        
        # 综合评分
        wellness = result['overall_wellness_score']
        color = (0, 255, 0) if wellness >= 80 else (255, 255, 0) if wellness >= 60 else (255, 100, 100)
        draw.text((panel_x, y_offset), f"综合健康: {wellness:.1f}", fill=color, font=font_header)
        draw.text((panel_x + 200, y_offset), result['overall_assessment'], fill=(200, 200, 200), font=font_small)
        y_offset += 50
        
        # 姿态部分
        if result['pose'].get('detected'):
            m = result['pose']['metrics']
            draw.text((panel_x, y_offset), f"■ 姿态分析 (总分: {m.overall_score:.1f})", fill=(100, 200, 255), font=font_header)
            y_offset += 30
            
            items = [
                (f"  肩部平衡: {m.shoulder_balance:.1f}", m.shoulder_balance),
                (f"  脊柱垂直: {m.spine_verticality:.1f}", m.spine_verticality),
                (f"  头部位置: {m.head_position:.1f}", m.head_position),
                (f"  髋部对齐: {m.hip_alignment:.1f}", m.hip_alignment),
                (f"  重心分布: {m.weight_distribution:.1f}", m.weight_distribution),
            ]
            for text, score in items:
                color = (150, 255, 150) if score >= 80 else (255, 255, 150) if score >= 60 else (255, 150, 150)
                draw.text((panel_x, y_offset), text, fill=color, font=font_text)
                y_offset += 22
            y_offset += 10
        
        # 面部部分
        if result['face'].get('detected'):
            m = result['face']['metrics']
            draw.text((panel_x, y_offset), f"■ 精神分析 (总分: {m.mental_state_score:.1f})", fill=(100, 255, 200), font=font_header)
            y_offset += 30
            
            draw.text((panel_x, y_offset), f"  表情: {m.expression_label}", fill=(255, 255, 200), font=font_text)
            y_offset += 22
            
            items = [
                (f"  压力水平: {m.stress_level:.1f}%", 100 - m.stress_level),
                (f"  疲劳程度: {m.fatigue_level:.1f}%", 100 - m.fatigue_level),
                (f"  放松程度: {m.relaxation_level:.1f}%", m.relaxation_level),
                (f"  面部对称: {m.facial_symmetry:.1f}%", m.facial_symmetry),
                (f"  眉间紧张: {m.brow_furrow_depth:.1f}", 100 - m.brow_furrow_depth),
            ]
            for text, score in items:
                color = (150, 255, 150) if score >= 80 else (255, 255, 150) if score >= 60 else (255, 150, 150)
                draw.text((panel_x, y_offset), text, fill=color, font=font_text)
                y_offset += 22
            y_offset += 10
        
        # 皮肤部分
        if result['skin']:
            m = result['skin']
            draw.text((panel_x, y_offset), f"■ 皮肤分析 (总分: {m['overall_score']:.1f})", fill=(255, 180, 200), font=font_header)
            y_offset += 30
            
            draw.text((panel_x, y_offset), f"  估计肤龄: {m['estimated_age']}岁", fill=(255, 220, 180), font=font_text)
            y_offset += 22
            
            items = [
                (f"  亮度均匀: {m['color_uniformity']:.1f}", m['color_uniformity']),
                (f"  纹理细腻: {m['texture_score']:.1f}", m['texture_score']),
                (f"  血液循环: {m['circulation_score']:.1f}", m['circulation_score']),
                (f"  水润度: {m['hydration_score']:.1f}", m['hydration_score']),
                (f"  通透度: {m['clarity_score']:.1f}", m['clarity_score']),
                (f"  抗皱度: {m['wrinkle_score']:.1f}", m['wrinkle_score']),
            ]
            for text, score in items:
                color = (150, 255, 150) if score >= 80 else (255, 255, 150) if score >= 60 else (255, 150, 150)
                draw.text((panel_x, y_offset), text, fill=color, font=font_text)
                y_offset += 22
        
        canvas = np.array(pil_image)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        return canvas
    
    def _draw_pose_on_image(self, image: np.ndarray, landmarks):
        """绘制姿态骨骼"""
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    def _draw_face_on_image(self, image: np.ndarray, face_points: np.ndarray):
        """绘制面部网格（采样显示）"""
        for (x, y, z) in face_points[::3]:  # 每3个点显示一个，避免太密
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 100), -1)
    
    def close(self):
        """释放资源"""
        self.pose_analyzer.close()
        self.face_mesh.close()
        self.face_detection.close()

# 初始化
analyzer = MediaPipeSpaAnalyzer()
'''
print("1. 姿态指标：从4项增加到11项（增加肩部角度、头部位置、重心分布等）")
print("2. 精神指标：从6项增加到15项（修复了固定75分问题，增加压力/疲劳/放松三维评估）")
print("3. 皮肤指标：从6项增加到16项（增加纹理、毛孔、皱纹、T区/脸颊分区分析）")
