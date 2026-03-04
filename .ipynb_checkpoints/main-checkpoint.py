from spa_analyzer_enhanced import MediaPipeSpaAnalyzer
from comparison_system_enhanced import SpaComparisonSystem
import matplotlib.pyplot as plt
import os

def main():
    """完整使用示例"""
    
    print("="*70)
    print("🏥 温泉疗养效果智能评估系统")
    print("基于 MediaPipe - 无需额外模型")
    print("增强：40+细分指标 | 修复精神评分 | 深度对比分析")
    print("="*70)
    
    spa = SpaComparisonSystem()
    
    try:
        # 设置图片路径 (修改为您的实际路径)
        before_image = r"photos/before.jpg"  # 疗养前照片
        after_image = r"photos/after.jpg"    # 疗养后照片
        
        # 检查文件是否存在
        if not os.path.exists(before_image):
            print(f"⚠️  未找到图片: {before_image}")
            print("请修改代码中的图片路径后重试")
            return
        
        if not os.path.exists(after_image):
            print(f"⚠️  未找到图片: {after_image}")
            print("请修改代码中的图片路径后重试")
            return
        
        # 3. 分析疗养前
        print("\\n" + "="*70)
        print("📸 步骤 1/4: 分析疗养前状态...")
        print("="*70)
        result_before = spa.analyze_before(before_image)
        
        # 打印详细指标
        print("\\n【疗养前详细指标】")
        if result_before['pose'].get('detected'):
            m = result_before['pose']['metrics']
            print(f"  姿态评分: {m.overall_score:.1f} (肩平衡:{m.shoulder_balance:.1f} 脊柱:{m.spine_verticality:.1f} 头部:{m.head_position:.1f})")
        
        if result_before['face'].get('detected'):
            m = result_before['face']['metrics']
            print(f"  精神评分: {m.mental_state_score:.1f} (疲劳:{m.fatigue_level:.1f}% 压力:{m.stress_level:.1f}% 放松:{m.relaxation_level:.1f}%)")
            print(f"  表情状态: {m.expression_label}")
        
        if result_before['skin']:
            m = result_before['skin']
            print(f"  皮肤评分: {m['overall_score']:.1f} (亮度:{m['brightness']:.1f} 平滑:{m['smoothness']:.1f} 水润:{m['hydration_score']:.1f})")
            print(f"  估计肤龄: {m['estimated_age']}岁")
        
        # 可视化单次结果
        print("\\n  生成可视化...")
        vis_before = spa.analyzer.visualize(result_before, "result_before.jpg")
        
        # 4. 分析疗养后
        print("\\n" + "="*70)
        print("📸 步骤 2/4: 分析疗养后状态...")
        print("="*70)
        result_after = spa.analyze_after(after_image)
        
        # 打印详细指标
        print("\\n【疗养后详细指标】")
        if result_after['pose'].get('detected'):
            m = result_after['pose']['metrics']
            print(f"  姿态评分: {m.overall_score:.1f} (肩平衡:{m.shoulder_balance:.1f} 脊柱:{m.spine_verticality:.1f} 头部:{m.head_position:.1f})")
        
        if result_after['face'].get('detected'):
            m = result_after['face']['metrics']
            print(f"  精神评分: {m.mental_state_score:.1f} (疲劳:{m.fatigue_level:.1f}% 压力:{m.stress_level:.1f}% 放松:{m.relaxation_level:.1f}%)")
            print(f"  表情状态: {m.expression_label}")
        
        if result_after['skin']:
            m = result_after['skin']
            print(f"  皮肤评分: {m['overall_score']:.1f} (亮度:{m['brightness']:.1f} 平滑:{m['smoothness']:.1f} 水润:{m['hydration_score']:.1f})")
            print(f"  估计肤龄: {m['estimated_age']}岁")
        
        vis_after = spa.analyzer.visualize(result_after, "result_after.jpg")
        
        # 5. 生成对比
        print("\\n" + "="*70)
        print("📊 步骤 3/4: 生成深度对比报告...")
        print("="*70)
        comparison = spa.generate_comparison()
        
        # 打印对比摘要
        print("\\n【对比结果摘要】")
        overall = comparison['overall']
        print(f"  综合改善: {overall['improvement']:+.2f}分 ({overall['improvement_percent']:+.1f}%) - {overall['trend']}")
        print(f"  评分变化: {overall['before_score']:.1f} → {overall['after_score']:.1f} ({overall['before_level']} → {overall['after_level']})")
        
        # 姿态对比
        if comparison['pose'].get('detected'):
            p = comparison['pose']
            print(f"\\n  【姿态改善】平均提升: {p['summary']['average_improvement']:+.2f}分")
            print(f"    最佳改善: {p['summary']['best_improved']['metric']} ({p['summary']['best_improved']['value']:+.2f})")
            print(f"    评估: {p['summary']['assessment']}")
        
        # 精神对比
        if comparison['mental'].get('detected'):
            m = comparison['mental']
            print(f"\\n  【精神改善】评分变化: {m['mental_comparison']['overall']['improvement']:+.2f}分")
            print(f"    压力缓解: {m['summary']['stress_reduction']:+.1f}%")
            print(f"    疲劳减轻: {m['summary']['fatigue_reduction']:+.1f}%")
            print(f"    放松提升: {m['summary']['relaxation_gain']:+.1f}%")
            print(f"    表情变化: {m['summary']['mood_change']}")
            print(f"    评估: {m['summary']['assessment']}")
        
        # 皮肤对比
        if comparison['skin'].get('detected'):
            s = comparison['skin']
            print(f"\\n  【皮肤改善】评分变化: {s['basic_metrics']['overall']['improvement']:+.2f}分")
            print(f"    亮度变化: {s['summary']['brightness_change']:+.2f} ({s['summary']['brightness_change_percent']:+.1f}%)")
            if s['summary']['age_reduction'] > 0:
                print(f"    肤龄减少: {s['summary']['age_reduction']}岁")
            print(f"    评估: {s['summary']['assessment']}")
        
        # 6. 可视化对比
        print("\\n" + "="*70)
        print("🎨 步骤 4/4: 生成可视化对比图表...")
        print("="*70)
        spa.visualize_comparison("comparison_full.jpg")
        
        # 7. 导出报告
        print("\\n" + "="*70)
        print("📄 导出详细报告...")
        print("="*70)
        spa.export_report(comparison, "spa_analysis_report_detailed.json")
        
        # 最终结论
        print("\\n" + "="*70)
        print("📝 最终结论")
        print("="*70)
        conclusion = comparison['conclusion']
        print(f"  {conclusion['summary_text']}")
        print(f"\\n  【疗养建议】")
        for i, suggestion in enumerate(conclusion['suggestions'], 1):
            print(f"    {i}. {suggestion}")
        
        print(f"\\n" + "="*70)
        print("✅ 分析完成！生成文件:")
        print("  📷 result_before.jpg - 疗养前分析图")
        print("  📷 result_after.jpg - 疗养后分析图")
        print("  📊 comparison_full.jpg - 详细对比图")
        print("  📋 spa_analysis_report_detailed.json - 完整数据报告")
        print("="*70)
        
    except Exception as e:
        print(f"\\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spa.close()

if __name__ == "__main__":
    main()
