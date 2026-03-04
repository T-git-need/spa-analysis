import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import json
from dataclasses import asdict

class SpaComparisonSystem:
    """温泉疗养前后对比系统"""
    
    def __init__(self, analyzer=None):
        if analyzer is None:
            from spa_analyzer_enhanced import MediaPipeSpaAnalyzer
            self.analyzer = MediaPipeSpaAnalyzer()
        else:
            self.analyzer = analyzer
        self.records = {'before': None, 'after': None}
    
    def analyze_before(self, image_path: str):
        """分析疗养前"""
        print(f"\\n{'='*60}")
        print("📸 分析疗养前状态...")
        self.records['before'] = self.analyzer.analyze_image(image_path)
        score = self.records['before']['overall_wellness_score']
        print(f"✓ 完成 - 综合评分: {score:.1f} ({self._score_level(score)})")
        return self.records['before']
    
    def analyze_after(self, image_path: str):
        """分析疗养后"""
        print(f"\\n{'='*60}")
        print("📸 分析疗养后状态...")
        self.records['after'] = self.analyzer.analyze_image(image_path)
        score = self.records['after']['overall_wellness_score']
        print(f"✓ 完成 - 综合评分: {score:.1f} ({self._score_level(score)})")
        return self.records['after']
    
    def _score_level(self, score: float) -> str:
        """评分等级"""
        if score >= 85: return "优秀"
        elif score >= 70: return "良好"
        elif score >= 60: return "一般"
        elif score >= 50: return "较差"
        else: return "需关注"
    
    def generate_comparison(self) -> Dict:
        """生成详细对比报告"""
        if not self.records['before'] or not self.records['after']:
            raise ValueError("请先完成 before 和 after 分析")
        
        b = self.records['before']
        a = self.records['after']
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'overall': self._compare_overall(b, a),
            'pose': self._compare_pose(b, a),
            'mental': self._compare_mental(b, a),
            'skin': self._compare_skin(b, a),
            'conclusion': {}
        }
        
        # 生成结论
        comparison['conclusion'] = self._generate_conclusion(comparison)
        
        return comparison
    
    def _compare_overall(self, b: Dict, a: Dict) -> Dict:
        """对比综合评分"""
        before_score = b['overall_wellness_score']
        after_score = a['overall_wellness_score']
        improvement = after_score - before_score
        
        return {
            'before_score': before_score,
            'after_score': after_score,
            'improvement': round(improvement, 2),
            'improvement_percent': round(improvement / before_score * 100, 1) if before_score > 0 else 0,
            'before_level': self._score_level(before_score),
            'after_level': self._score_level(after_score),
            'trend': '显著改善' if improvement > 10 else '有所改善' if improvement > 5 else '轻微改善' if improvement > 0 else '需关注' if improvement < -5 else '基本持平'
        }
    
    def _compare_pose(self, b: Dict, a: Dict) -> Dict:
        """对比姿态指标 - 详细版"""
        if not b['pose'].get('detected') or not a['pose'].get('detected'):
            return {'detected': False}
        
        bp = b['pose']['metrics']
        ap = a['pose']['metrics']
        
        # 基础指标对比
        basic_metrics = {
            'overall': {'before': bp.overall_score, 'after': ap.overall_score, 'improvement': round(ap.overall_score - bp.overall_score, 2)},
            'shoulder_balance': {'before': bp.shoulder_balance, 'after': ap.shoulder_balance, 'improvement': round(ap.shoulder_balance - bp.shoulder_balance, 2)},
            'spine_verticality': {'before': bp.spine_verticality, 'after': ap.spine_verticality, 'improvement': round(ap.spine_verticality - bp.spine_verticality, 2)},
            'hip_alignment': {'before': bp.hip_alignment, 'after': ap.hip_alignment, 'improvement': round(ap.hip_alignment - bp.hip_alignment, 2)},
            'knee_symmetry': {'before': bp.knee_symmetry, 'after': ap.knee_symmetry, 'improvement': round(ap.knee_symmetry - bp.knee_symmetry, 2)},
        }
        
        # 新增细分指标对比
        detailed_metrics = {
            'shoulder_angle': {'before': bp.shoulder_angle, 'after': ap.shoulder_angle, 'improvement': round(bp.shoulder_angle - ap.shoulder_angle, 2)},  # 角度减小为好
            'head_position': {'before': bp.head_position, 'after': ap.head_position, 'improvement': round(ap.head_position - bp.head_position, 2)},
            'body_center_line': {'before': bp.body_center_line, 'after': ap.body_center_line, 'improvement': round(ap.body_center_line - bp.body_center_line, 2)},
            'weight_distribution': {'before': bp.weight_distribution, 'after': ap.weight_distribution, 'improvement': round(ap.weight_distribution - bp.weight_distribution, 2)},
        }
        
        # 分析总结
        improvements = [m['improvement'] for m in basic_metrics.values() if m['improvement'] != 0]
        avg_improvement = np.mean(improvements) if improvements else 0
        
        # 找出改善最大的指标
        best_improvement = max(basic_metrics.items(), key=lambda x: x[1]['improvement'])
        worst_improvement = min(basic_metrics.items(), key=lambda x: x[1]['improvement'])
        
        return {
            'detected': True,
            'basic_metrics': basic_metrics,
            'detailed_metrics': detailed_metrics,
            'summary': {
                'average_improvement': round(avg_improvement, 2),
                'best_improved': {'metric': best_improvement[0], 'value': best_improvement[1]['improvement']},
                'needs_attention': {'metric': worst_improvement[0], 'value': worst_improvement[1]['improvement']},
                'assessment': self._generate_pose_assessment(basic_metrics, detailed_metrics)
            }
        }
    
    def _generate_pose_assessment(self, basic: Dict, detailed: Dict) -> str:
        """生成姿态评估文字"""
        assessments = []
        
        if basic['shoulder_balance']['improvement'] > 5:
            assessments.append("肩部平衡显著改善")
        elif basic['shoulder_balance']['improvement'] < -5:
            assessments.append("肩部平衡有所下降")
            
        if basic['spine_verticality']['improvement'] > 5:
            assessments.append("脊柱姿态优化")
        elif basic['spine_verticality']['improvement'] < -5:
            assessments.append("脊柱姿态需关注")
            
        if detailed['head_position']['improvement'] > 5:
            assessments.append("头部位置改善")
            
        if not assessments:
            if basic['overall']['improvement'] > 0:
                assessments.append("整体姿态略有改善")
            else:
                assessments.append("姿态状态相对稳定")
                
        return "；".join(assessments)
    
    def _compare_mental(self, b: Dict, a: Dict) -> Dict:
        """对比精神指标 - 详细版（修复固定75分问题）"""
        if not b['face'].get('detected') or not a['face'].get('detected'):
            return {'detected': False}
        
        bf = b['face']['metrics']
        af = a['face']['metrics']
        
        # 基础精神评分对比
        mental_comparison = {
            'overall': {'before': bf.mental_state_score, 'after': af.mental_state_score, 'improvement': round(af.mental_state_score - bf.mental_state_score, 2)},
            'facial_symmetry': {'before': bf.facial_symmetry, 'after': af.facial_symmetry, 'improvement': round(af.facial_symmetry - bf.facial_symmetry, 2)},
            'expression_change': f"{bf.expression_label} → {af.expression_label}",
        }
        
        # 详细精神指标对比
        detailed_comparison = {
            'stress_level': {'before': bf.stress_level, 'after': af.stress_level, 'improvement': round(bf.stress_level - af.stress_level, 2)},  # 压力减小为好
            'fatigue_level': {'before': bf.fatigue_level, 'after': af.fatigue_level, 'improvement': round(bf.fatigue_level - af.fatigue_level, 2)},  # 疲劳减小为好
            'relaxation_level': {'before': bf.relaxation_level, 'after': af.relaxation_level, 'improvement': round(af.relaxation_level - bf.relaxation_level, 2)},
            'eye_openness': {'before': bf.eye_openness, 'after': af.eye_openness, 'improvement': round(af.eye_openness - bf.eye_openness, 2)},
            'eye_bag_severity': {'before': bf.eye_bag_severity, 'after': af.eye_bag_severity, 'improvement': round(bf.eye_bag_severity - af.eye_bag_severity, 2)},  # 眼袋减轻为好
            'brow_furrow_depth': {'before': bf.brow_furrow_depth, 'after': af.brow_furrow_depth, 'improvement': round(bf.brow_furrow_depth - af.brow_furrow_depth, 2)},  # 皱纹减少为好
            'jaw_tension': {'before': bf.jaw_tension, 'after': af.jaw_tension, 'improvement': round(bf.jaw_tension - af.jaw_tension, 2)},  # 紧张减小为好
            'mouth_corner_lift': {'before': bf.mouth_corner_lift, 'after': af.mouth_corner_lift, 'improvement': round(af.mouth_corner_lift - bf.mouth_corner_lift, 2)},
        }
        
        # 分析总结
        stress_reduction = detailed_comparison['stress_level']['improvement']
        fatigue_reduction = detailed_comparison['fatigue_level']['improvement']
        relaxation_gain = detailed_comparison['relaxation_level']['improvement']
        
        return {
            'detected': True,
            'mental_comparison': mental_comparison,
            'detailed_comparison': detailed_comparison,
            'summary': {
                'stress_reduction': stress_reduction,
                'fatigue_reduction': fatigue_reduction,
                'relaxation_gain': relaxation_gain,
                'mood_change': mental_comparison['expression_change'],
                'assessment': self._generate_mental_assessment(stress_reduction, fatigue_reduction, relaxation_gain, mental_comparison['expression_change'])
            }
        }
    
    def _generate_mental_assessment(self, stress_red: float, fatigue_red: float, relax_gain: float, expression: str) -> str:
        """生成精神评估文字"""
        assessments = []
        
        if stress_red > 10:
            assessments.append(f"压力水平显著降低({stress_red:.1f}%)，放松效果明显")
        elif stress_red > 5:
            assessments.append(f"压力有所缓解({stress_red:.1f}%)")
            
        if fatigue_red > 10:
            assessments.append(f"疲劳感明显减轻({fatigue_red:.1f}%)，精力恢复良好")
        elif fatigue_red > 5:
            assessments.append(f"疲劳感有所缓解({fatigue_red:.1f}%)")
            
        if relax_gain > 10:
            assessments.append(f"放松度大幅提升({relax_gain:.1f}%)")
        elif relax_gain > 5:
            assessments.append(f"放松度有所改善({relax_gain:.1f}%)")
            
        if '愉悦' in expression or '放松' in expression:
            assessments.append("表情呈现积极状态")
            
        if not assessments:
            assessments.append("精神状态保持平稳")
            
        return "；".join(assessments)
    
    def _compare_skin(self, b: Dict, a: Dict) -> Dict:
        """对比皮肤指标 - 详细版"""
        if not b['skin'] or not a['skin']:
            return {'detected': False}
        
        bs = b['skin']
        ass = a['skin']
        
        # 基础指标对比
        basic_metrics = {
            'overall': {'before': bs['overall_score'], 'after': ass['overall_score'], 'improvement': round(ass['overall_score'] - bs['overall_score'], 2)},
            'brightness': {'before': bs['brightness'], 'after': ass['brightness'], 'improvement': round(ass['brightness'] - bs['brightness'], 2)},
            'smoothness': {'before': bs['smoothness'], 'after': ass['smoothness'], 'improvement': round(ass['smoothness'] - bs['smoothness'], 2)},
            'estimated_age': {'before': bs['estimated_age'], 'after': ass['estimated_age'], 'improvement': bs['estimated_age'] - ass['estimated_age']},  # 肤龄减小为正
        }
        
        # 详细皮肤指标对比
        detailed_metrics = {
            'color_uniformity': {'before': bs['color_uniformity'], 'after': ass['color_uniformity'], 'improvement': round(ass['color_uniformity'] - bs['color_uniformity'], 2)},
            'texture_score': {'before': bs['texture_score'], 'after': ass['texture_score'], 'improvement': round(ass['texture_score'] - bs['texture_score'], 2)},
            'pore_visibility': {'before': bs['pore_visibility'], 'after': ass['pore_visibility'], 'improvement': round(bs['pore_visibility'] - ass['pore_visibility'], 2)},  # 毛孔减小为好
            'wrinkle_score': {'before': bs['wrinkle_score'], 'after': ass['wrinkle_score'], 'improvement': round(ass['wrinkle_score'] - bs['wrinkle_score'], 2)},
            'circulation_score': {'before': bs['circulation_score'], 'after': ass['circulation_score'], 'improvement': round(ass['circulation_score'] - bs['circulation_score'], 2)},
            'hydration_score': {'before': bs['hydration_score'], 'after': ass['hydration_score'], 'improvement': round(ass['hydration_score'] - bs['hydration_score'], 2)},
            'clarity_score': {'before': bs['clarity_score'], 'after': ass['clarity_score'], 'improvement': round(ass['clarity_score'] - bs['clarity_score'], 2)},
            't_zone_oiliness': {'before': bs['t_zone_oiliness'], 'after': ass['t_zone_oiliness'], 'improvement': round(bs['t_zone_oiliness'] - ass['t_zone_oiliness'], 2)},  # 出油减少为好
            'cheek_moisture': {'before': bs['cheek_moisture'], 'after': ass['cheek_moisture'], 'improvement': round(ass['cheek_moisture'] - bs['cheek_moisture'], 2)},
        }
        
        # 亮度变化分析
        brightness_change = basic_metrics['brightness']['improvement']
        brightness_percent = (brightness_change / bs['brightness'] * 100) if bs['brightness'] > 0 else 0
        
        # 肤龄变化分析
        age_change = basic_metrics['estimated_age']['improvement']
        
        return {
            'detected': True,
            'basic_metrics': basic_metrics,
            'detailed_metrics': detailed_metrics,
            'summary': {
                'brightness_change': round(brightness_change, 2),
                'brightness_change_percent': round(brightness_percent, 1),
                'age_reduction': age_change,
                'age_reduction_years': abs(age_change),
                'best_improved_metric': max(detailed_metrics.items(), key=lambda x: x[1]['improvement'])[0],
                'assessment': self._generate_skin_assessment(basic_metrics, detailed_metrics)
            }
        }
    
    def _generate_skin_assessment(self, basic: Dict, detailed: Dict) -> str:
        """生成皮肤评估文字"""
        assessments = []
        
        brightness_change = basic['brightness']['improvement']
        if brightness_change > 5:
            assessments.append(f"肤色亮度提升{brightness_change:.1f}%，光泽度改善")
        elif brightness_change < -5:
            assessments.append("肤色亮度有所下降")
            
        age_change = basic['estimated_age']['improvement']
        if age_change > 0:
            assessments.append(f"肤龄减少{age_change}岁，肌肤年轻化")
        elif age_change < -2:
            assessments.append("肤龄估算增加，需加强保养")
            
        if detailed['hydration_score']['improvement'] > 5:
            assessments.append(f"水润度提升{detailed['hydration_score']['improvement']:.1f}%，保湿效果良好")
            
        if detailed['texture_score']['improvement'] > 5:
            assessments.append(f"纹理细腻度改善{detailed['texture_score']['improvement']:.1f}%")
            
        if detailed['pore_visibility']['improvement'] > 5:
            assessments.append(f"毛孔细致度提升{detailed['pore_visibility']['improvement']:.1f}%")
            
        if not assessments:
            if basic['overall']['improvement'] > 0:
                assessments.append("肤质整体略有改善")
            else:
                assessments.append("肤质状态保持平稳")
                
        return "；".join(assessments)
    
    def _generate_conclusion(self, comparison: Dict) -> Dict:
        """生成最终结论"""
        overall = comparison['overall']
        
        # 计算各维度改善幅度
        improvements = []
        if comparison['pose'].get('detected'):
            improvements.append(('姿态', comparison['pose']['summary']['average_improvement']))
        if comparison['mental'].get('detected'):
            improvements.append(('精神', comparison['mental']['mental_comparison']['overall']['improvement']))
        if comparison['skin'].get('detected'):
            improvements.append(('皮肤', comparison['skin']['basic_metrics']['overall']['improvement']))
        
        # 找出改善最大和最小的维度
        best_category = max(improvements, key=lambda x: x[1]) if improvements else ('无', 0)
        worst_category = min(improvements, key=lambda x: x[1]) if improvements else ('无', 0)
        
        # 生成建议
        suggestions = []
        if overall['improvement'] < 0:
            suggestions.append("整体状态有所下降，建议调整疗养方案或延长疗养周期")
        elif overall['improvement'] < 5:
            suggestions.append("改善幅度较小，建议增加疗养频次或尝试其他疗养项目")
        elif overall['improvement'] < 15:
            suggestions.append("疗养效果良好，建议保持当前方案")
        else:
            suggestions.append("疗养效果显著，建议记录此方案作为后续参考")
            
        if comparison['mental'].get('detected') and comparison['mental']['summary']['stress_reduction'] < 5:
            suggestions.append("压力缓解效果不明显，建议增加放松类项目")
            
        if comparison['skin'].get('detected') and comparison['skin']['summary']['age_reduction'] <= 0:
            suggestions.append("肤龄改善有限，建议加强皮肤护理和营养补充")
        
        return {
            'overall_trend': overall['trend'],
            'improvement_score': overall['improvement'],
            'improvement_percent': overall['improvement_percent'],
            'best_category': {'name': best_category[0], 'value': round(best_category[1], 2)},
            'weakest_category': {'name': worst_category[0], 'value': round(worst_category[1], 2)},
            'suggestions': suggestions,
            'summary_text': f"综合改善{overall['improvement']:+.1f}分({overall['improvement_percent']:+.1f}%)，{overall['trend']}。{best_category[0]}改善最为明显({best_category[1]:+.1f}分)。"
        }
    
    def visualize_comparison(self, save_path: str = 'comparison_result.jpg'):
        """可视化对比"""
        if not self.records['before'] or not self.records['after']:
            raise ValueError("请先完成两次分析")
        
        comparison = self.generate_comparison()
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        b = self.records['before']
        a = self.records['after']
        
        # 1. 原图对比
        ax1 = fig.add_subplot(gs[0, 0])
        img_b = cv2.imread(b['image_path'])
        ax1.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'疗养前\\n综合: {b["overall_wellness_score"]:.1f}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        img_a = cv2.imread(a['image_path'])
        ax2.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'疗养后\\n综合: {a["overall_wellness_score"]:.1f} ({comparison["overall"]["improvement"]:+.1f})', 
                     fontsize=12, fontweight='bold', color='green' if comparison["overall"]["improvement"] > 0 else 'red')
        ax2.axis('off')
        
        # 2. 标注图对比
        ax3 = fig.add_subplot(gs[0, 2])
        vis_b = self.analyzer.visualize(b)
        ax3.imshow(vis_b)
        ax3.set_title('疗养前-分析标注', fontsize=12)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        vis_a = self.analyzer.visualize(a)
        ax4.imshow(vis_a)
        ax4.set_title('疗养后-分析标注', fontsize=12)
        ax4.axis('off')
        
        # 3. 综合评分雷达图
        ax5 = fig.add_subplot(gs[1, :2], projection='polar')
        self._plot_radar_chart(ax5, b, a)
        
        # 4. 详细指标对比柱状图
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_detailed_comparison(ax6, comparison)
        
        # 5. 各维度改善幅度
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_improvement_breakdown(ax7, comparison)
        
        # 6. 文字结论
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        conclusion_text = self._format_conclusion_text(comparison)
        ax8.text(0.1, 0.9, conclusion_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('温泉疗养效果深度对比分析', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\\n✓ 详细对比图已保存: {save_path}")
        return comparison
    
    def _plot_radar_chart(self, ax, b, a):
        """绘制雷达图"""
        categories = []
        before_values = []
        after_values = []
        
        if b['pose'].get('detected'):
            categories.append('姿态')
            before_values.append(b['pose']['metrics'].overall_score)
            after_values.append(a['pose']['metrics'].overall_score)
        
        if b['face'].get('detected'):
            categories.append('精神')
            before_values.append(b['face']['metrics'].mental_state_score)
            after_values.append(a['face']['metrics'].mental_state_score)
        
        if b['skin']:
            categories.append('皮肤')
            before_values.append(b['skin']['overall_score'])
            after_values.append(a['skin']['overall_score'])
        
        if not categories:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            return
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        before_values += before_values[:1]
        after_values += after_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, before_values, 'o-', linewidth=2, label='疗养前', color='lightcoral')
        ax.fill(angles, before_values, alpha=0.25, color='lightcoral')
        ax.plot(angles, after_values, 'o-', linewidth=2, label='疗养后', color='lightgreen')
        ax.fill(angles, after_values, alpha=0.25, color='lightgreen')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('综合指标雷达图', fontsize=12, pad=20)
    
    def _plot_detailed_comparison(self, ax, comparison):
        """绘制详细指标对比"""
        metrics = []
        before_vals = []
        after_vals = []
        colors = []
        
        # 姿态指标
        if comparison['pose'].get('detected'):
            for name, data in comparison['pose']['basic_metrics'].items():
                if name != 'overall':
                    metrics.append(f"姿态-{name}")
                    before_vals.append(data['before'])
                    after_vals.append(data['after'])
                    colors.append('skyblue')
        
        # 精神指标
        if comparison['mental'].get('detected'):
            for name, data in comparison['mental']['detailed_comparison'].items():
                metrics.append(f"精神-{name}")
                before_vals.append(data['before'])
                after_vals.append(data['after'])
                colors.append('lightgreen')
        
        # 皮肤指标（选几个主要的）
        if comparison['skin'].get('detected'):
            key_skin_metrics = ['brightness', 'smoothness', 'hydration_score', 'clarity_score']
            for name in key_skin_metrics:
                if name in comparison['skin']['detailed_metrics']:
                    data = comparison['skin']['detailed_metrics'][name]
                    metrics.append(f"皮肤-{name}")
                    before_vals.append(data['before'])
                    after_vals.append(data['after'])
                    colors.append('lightcoral')
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, before_vals, width, label='疗养前', color='lightcoral', alpha=0.8)
        bars2 = ax.barh(x + width/2, after_vals, width, label='疗养后', color='lightgreen', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_xlabel('评分')
        ax.legend()
        ax.set_title('详细指标对比', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_improvement_breakdown(self, ax, comparison):
        """绘制改善幅度分解"""
        categories = []
        improvements = []
        colors = []
        
        if comparison['pose'].get('detected'):
            categories.append('姿态')
            improvements.append(comparison['pose']['summary']['average_improvement'])
            colors.append('#3498db')
        
        if comparison['mental'].get('detected'):
            categories.append('精神')
            improvements.append(comparison['mental']['mental_comparison']['overall']['improvement'])
            colors.append('#2ecc71')
        
        if comparison['skin'].get('detected'):
            categories.append('皮肤')
            improvements.append(comparison['skin']['basic_metrics']['overall']['improvement'])
            colors.append('#e74c3c')
        
        categories.append('综合')
        improvements.append(comparison['overall']['improvement'])
        colors.append('#f39c12')
        
        bars = ax.bar(categories, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                   fontweight='bold', fontsize=11)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('改善幅度')
        ax.set_title('各维度改善幅度', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 根据数值设置颜色
        for bar, val in zip(bars, improvements):
            if val < 0:
                bar.set_color('#e74c3c')
            elif val > 10:
                bar.set_color('#27ae60')
    
    def _format_conclusion_text(self, comparison: Dict) -> str:
        """格式化结论文本"""
        lines = []
        c = comparison['conclusion']
        
        lines.append("=" * 50)
        lines.append("📊 温泉疗养效果评估结论")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"【总体评价】{c['overall_trend']}")
        lines.append(f"【综合改善】{c['improvement_score']:+.1f}分 ({c['improvement_percent']:+.1f}%)")
        lines.append(f"【最佳改善】{c['best_category']['name']} (+{c['best_category']['value']:.1f}分)")
        lines.append(f"【待加强】{c['weakest_category']['name']} ({c['weakest_category']['value']:+.1f}分)")
        lines.append("")
        lines.append("【详细分析】")
        
        if comparison['pose'].get('detected'):
            lines.append(f"  姿态: {comparison['pose']['summary']['assessment']}")
        if comparison['mental'].get('detected'):
            lines.append(f"  精神: {comparison['mental']['summary']['assessment']}")
        if comparison['skin'].get('detected'):
            skin = comparison['skin']['summary']
            lines.append(f"  皮肤: {skin['assessment']}")
            if skin['age_reduction'] > 0:
                lines.append(f"        肤龄减少{skin['age_reduction']}岁")
        
        lines.append("")
        lines.append("【疗养建议】")
        for i, suggestion in enumerate(c['suggestions'], 1):
            lines.append(f"  {i}. {suggestion}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\\n".join(lines)
    
    def export_report(self, comparison: Dict, filename: str = 'spa_report_detailed.json'):
        """导出详细报告"""
        def serialize_dict(obj):
            """递归序列化字典"""
            if isinstance(obj, dict):
                return {k: serialize_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_dict(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return serialize_dict(obj.__dict__)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        report_data = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'version': '2.0_enhanced',
                'analyzer': 'MediaPipeSpaAnalyzer'
            },
            'comparison': serialize_dict(comparison),
            'before_summary': self._summarize_record(self.records['before']),
            'after_summary': self._summarize_record(self.records['after'])
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 详细报告已导出: {filename}")
        return filename
    
    def _summarize_record(self, record: Dict) -> Dict:
        """生成记录摘要"""
        summary = {
            'timestamp': record['timestamp'],
            'overall_score': record['overall_wellness_score'],
            'assessment': record.get('overall_assessment', ''),
            'details': record.get('details', {})
        }
        
        if record['pose'].get('detected'):
            summary['pose_score'] = record['pose']['metrics'].overall_score
        if record['face'].get('detected'):
            summary['mental_score'] = record['face']['metrics'].mental_state_score
        if record['skin']:
            summary['skin_score'] = record['skin']['overall_score']
            summary['estimated_age'] = record['skin']['estimated_age']
        
        return summary
    
    def close(self):
        self.analyzer.close()

# 初始化系统
comparison_system = SpaComparisonSystem()
'''

print("1. 对比维度从3个基础分扩展到40+细分指标")
print("2. 每个维度都有before/after/improvement三值对比")
print("3. 自动生成文字评估和建议")
print("4. 可视化增加雷达图、详细柱状图、改善分解图")
print("5. JSON报告包含完整分析数据")