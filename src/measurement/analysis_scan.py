import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
# ========================
# 配置参数
# ========================

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 扫描结果目录（默认指向示例数据，可自行修改）
SCAN_RESULT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "global_scan")
UM_PER_PIXEL = 0.397

OUTPUT_DIR = os.path.join(SCAN_RESULT_DIR, "morphology_statistics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# 字体设置：统一 Arial（期刊标准）
# ========================

# 清除之前的中文字体设置，统一用 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = True  # 正常显示负号
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# 字号设置（期刊标准）
FONT_SIZE = {
    'tick': 8,      # 刻度标签
    'label': 9,     # 轴标签
    'title': 10,    # 子图标题 (a)(b)(c)(d)
    'legend': 8,    # 图例
    'text': 9,      # 其他文字
}

# ========================
# 期刊尺寸参数
# ========================

COLUMN_WIDTH_CM = 17.4      # 双栏宽度
COLUMN_WIDTH_INCH = COLUMN_WIDTH_CM / 2.54
DPI = 1000                  # 线条图标准

# 2×2 布局，计算高度（保持比例美观）
# 宽度 17.4cm，每张图约 8cm 宽，高度按 4:3 比例约 6cm
# 2×2 加间隙，总高度约 13-14cm
FIG_HEIGHT_CM = 13.0
FIG_HEIGHT_INCH = FIG_HEIGHT_CM / 2.54

# ========================
# 辅助函数（简化，统一 Arial）
# ========================

def set_tick_font(ax, fontsize=FONT_SIZE['tick']):
    """设置刻度标签字体"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)


def add_subplot_label(ax, label, fontsize=FONT_SIZE['title']):
    """
    子图标签放在x轴下方居中
    """
    ax.text(0.5, -0.27, label,  # ← 用 label 参数，不是 cfg
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight='bold',
            fontname='Arial',
            color='black',
            ha='center', va='top',
            zorder=100)

# ========================
# 形态参数计算函数（不变）
# ========================

def calculate_morphology(mask_path, um_per_px):
    if not os.path.exists(mask_path):
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    inclusion_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    contours, _ = cv2.findContours(inclusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(contour)
    area_um2 = area_px * (um_per_px ** 2)
    perimeter_px = cv2.arcLength(contour, True)
    perimeter_um = perimeter_px * um_per_px
    equiv_diameter = 2 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else 0
    
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else np.nan
    
    if perimeter_um > 0:
        circularity = 4 * np.pi * area_um2 / (perimeter_um ** 2)
        circularity = max(0, min(1, circularity))
    else:
        circularity = np.nan
    
    return {
        '面积_um2': round(area_um2, 2),
        '周长_um': round(perimeter_um, 2),
        '等效直径_um': round(equiv_diameter, 2),
        '长宽比': round(aspect_ratio, 3),
        '圆度': round(circularity, 4),
    }


# ========================
# 主分析流程（不变）
# ========================

def main():
    csv_files = [f for f in os.listdir(SCAN_RESULT_DIR) 
                 if f.startswith('all_inclusions_summary') and f.endswith('.csv')]
    if not csv_files:
        print("❌ 未找到汇总CSV文件")
        return
    
    csv_path = os.path.join(SCAN_RESULT_DIR, csv_files[0])
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"📂 读取到 {len(df)} 个包裹体数据")
    
    morph_results = []
    mask_dir = os.path.join(SCAN_RESULT_DIR, 'roi_masks')
    
    for idx, row in df.iterrows():
        inclusion_name = row['包裹体名称']
        mask_path = os.path.join(mask_dir, f"{inclusion_name}_mask.png")
        
        morph = calculate_morphology(mask_path, UM_PER_PIXEL)
        if morph:
            morph_results.append(morph)
        else:
            morph_results.append({
                '面积_um2': np.nan, '周长_um': np.nan, '等效直径_um': np.nan,
                '长宽比': np.nan, '圆度': np.nan,
            })
        
        if (idx + 1) % 50 == 0:
            print(f"   已处理 {idx + 1}/{len(df)}...")
    
    morph_df = pd.DataFrame(morph_results)
    df_combined = pd.concat([df.reset_index(drop=True), morph_df], axis=1)
    
    combined_csv = os.path.join(OUTPUT_DIR, "inclusions_with_morphology.csv")
    df_combined.to_csv(combined_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 完整数据已保存: {combined_csv}")
    
    stats_cols = ['等效直径_um', '长宽比', '圆度', '气液比(G/L)']
    stats_df = df_combined[stats_cols].copy()
    stats_df = stats_df.replace([np.inf, -np.inf], np.nan)
    
    summary = pd.DataFrame({
        '参数': stats_cols,
        '最大值': stats_df.max().round(3).values,
        '最小值': stats_df.min().round(3).values,
        '平均值': stats_df.mean().round(3).values,
        '标准差': stats_df.std().round(3).values,
        '样本数': stats_df.count().values
    })
    
    summary_csv = os.path.join(OUTPUT_DIR, "morphology_summary.csv")
    summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 统计汇总表已保存: {summary_csv}")
    
    print("\n" + "="*70)
    print("📊 薄片流体包裹体形态参数统计结果")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70)
    
    plot_distributions(df_combined, OUTPUT_DIR)
    
    print(f"\n🎉 全部完成！结果保存在: {OUTPUT_DIR}")


# ========================
# 绘制分布图（期刊标准版）
# ========================

def plot_distributions(df, out_dir):
    df_plot = df.replace([np.inf, -np.inf], np.nan)
    
    fig, axes = plt.subplots(2, 2, 
                             figsize=(COLUMN_WIDTH_INCH, FIG_HEIGHT_INCH),
                             dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # ========== 增大四周间距 ==========
    plt.subplots_adjust(left=0.12, right=0.96,
                    bottom=0.22, top=0.92,
                    wspace=0.30, hspace=0.45)
    
    BAR_COLOR = '#5B9BD5'
    MEAN_COLOR = '#C00000'
    SPINE_WIDTH = 1.0
    
    plot_configs = [
        {
            'col': '等效直径_um',
            'label': '(a)',
            'xlabel': 'Equivalent diameter ($\\mu$m)',
            'ylabel': 'Number of inclusions',
            'ax': axes[0, 0],
            'bin_step': 5,
            'xlim': (-3, None),
            'xticks': np.arange(0, 81, 10),
            'xtick_fmt': '{:.0f}',
        },
        {
            'col': '长宽比',
            'label': '(b)',
            'xlabel': 'Aspect ratio',
            'ylabel': 'Number of inclusions',
            'ax': axes[0, 1],
            'bin_step': 0.2,
            'xlim': (0.8, None),
            'xticks': np.arange(1.0, 4.1, 0.5),
            'xtick_fmt': '{:.1f}',
        },
        {
            'col': '气液比(G/L)',
            'label': '(c)',
            'xlabel': 'Gas-liquid ratio (G/L)',
            'ylabel': 'Number of inclusions',
            'ax': axes[1, 0],
            'bin_step': 0.05,
            'xlim': (-0.03, None),
            'xticks': np.arange(0.0, 0.61, 0.1),
            'xtick_fmt': '{:.1f}',
        },
        {
            'col': '圆度',
            'label': '(d)',
            'xlabel': 'Circularity',
            'ylabel': 'Number of inclusions',
            'ax': axes[1, 1],
            'bin_step': 0.05,
            'xlim': (0.3, 1.0),
            'xticks': np.arange(0.3, 1.01, 0.1),
            'xtick_fmt': '{:.1f}',
            'legend_loc': 'upper left',
        },
    ]
    
    for cfg in plot_configs:
        col = cfg['col']
        ax = cfg['ax']
        data = df_plot[col].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue
        
        data_min, data_max = data.min(), data.max()
        bin_step = cfg['bin_step']
        bin_start = np.floor(data_min / bin_step) * bin_step
        bin_end = np.ceil(data_max / bin_step) * bin_step + bin_step
        bin_edges = np.arange(bin_start, bin_end, bin_step)
        
        # 绘制直方图
        n, bins, patches = ax.hist(
            data, bins=bin_edges,
            color=BAR_COLOR,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            zorder=2
        )
        
        # 边框
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
            spine.set_color('black')
            spine.set_zorder(10)
        
        # 均值线
        mean_val = data.mean()
        ax.axvline(
            mean_val,
            color=MEAN_COLOR,
            linestyle='--',
            linewidth=1.5,
            label=f'Mean = {mean_val:.2f}',
            zorder=10
        )
        
        # x轴范围
        xlim = cfg['xlim']
        if xlim:
            left = xlim[0] if xlim[0] is not None else data_min - bin_step
            right = xlim[1] if xlim[1] is not None else data_max + bin_step
            ax.set_xlim(left, right)
        else:
            margin = bin_step
            ax.set_xlim(data_min - margin, data_max + margin)
        
        # 整齐刻度
        xticks = cfg['xticks']
        x_left, x_right = ax.get_xlim()
        xticks = xticks[(xticks >= x_left) & (xticks <= x_right)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([cfg['xtick_fmt'].format(x) for x in xticks], 
                           fontsize=FONT_SIZE['tick'], fontname='Arial')
        
        # 刻度样式
        ax.tick_params(axis='x', direction='out', length=4, width=1.0, 
                       labelsize=FONT_SIZE['tick'])
        ax.tick_params(axis='y', direction='out', length=4, width=1.0, 
                       labelsize=FONT_SIZE['tick'])
        
        # y轴整数刻度
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 轴标签
        ax.set_xlabel(cfg['xlabel'], fontsize=FONT_SIZE['label'], 
                      fontname='Arial')
        ax.set_ylabel(cfg['ylabel'], fontsize=FONT_SIZE['label'], 
                      fontname='Arial')
        
        # 刻度字体
        set_tick_font(ax, FONT_SIZE['tick'])
        
        # 图例
        legend_loc = cfg.get('legend_loc', 'upper right')
        legend = ax.legend(fontsize=FONT_SIZE['legend'], 
                          frameon=True, fancybox=False,
                          edgecolor='gray', facecolor='white', 
                          loc=legend_loc)
        for text in legend.get_texts():
            text.set_fontname('Arial')
            text.set_fontsize(FONT_SIZE['legend'])
        
        # 网格
        ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.4, zorder=0)
        ax.grid(axis='x', alpha=0)
        
        # ========== 子图标签：左上角，黑色 ==========
        add_subplot_label(ax, cfg['label'], fontsize=FONT_SIZE['title'])

    
    # 保存
    fig_path = os.path.join(out_dir, "Fig6.tif")
    
    plt.savefig(fig_path,
                format='tiff',
                dpi=DPI,
                bbox_inches='tight',
                pad_inches=0.08,  # 稍微增大边距
                facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    
    plt.close()
    
    file_size_mb = os.path.getsize(fig_path) / 1024 / 1024
    print(f"📈 Fig6 已保存: {fig_path}")
    print(f"   尺寸: {COLUMN_WIDTH_CM}cm × {FIG_HEIGHT_CM}cm, {DPI}dpi")
    print(f"   像素: {int(COLUMN_WIDTH_INCH*DPI)} × {int(FIG_HEIGHT_INCH*DPI)}")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    if file_size_mb > 10:
        print(f"   ⚠️ 超过10MB，投稿时需单独上传")


if __name__ == "__main__":
    main()