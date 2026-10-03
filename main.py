"""入口与批量调度：参数解析、硬件生命周期、批量循环、降温控制、批次报告。"""
import argparse
import os
import time
from datetime import datetime

import config
from src.hardware import hardware
from src.dl_model import model_loader as models
from src.measurement import measurement
from src.utils import io_utils

import signal
import traceback
# 注册信号处理器：Ctrl+C 时打印当前堆栈
def _signal_handler(signum, frame):
    print("\n⚠️ 强制中断! 打印当前堆栈:")
    traceback.print_stack(frame)
    exit(1)

signal.signal(signal.SIGINT, _signal_handler)

def active_cooling():
    """
    主动降温：设定目标温度，循环监测直到温度≤恢复阈值。
    """
    print("\n" + "="*60)
    print("❄️ 启动主动降温")
    print(f"   目标温度: {config.COOLING_TARGET_TEMP}°C | 恢复阈值: ≤{config.COOLING_RESUME_THRESHOLD}°C")
    print("="*60)
    
    hardware.set_target_temp(config.COOLING_TARGET_TEMP)
    hardware.set_heating_rate(40.0)
    hardware.start_heating()
    
    start_time = time.time()
    
    while True:
        time.sleep(config.COOLING_CHECK_INTERVAL)
        current_temp = hardware.get_temp()
        elapsed = time.time() - start_time
        
        if current_temp is not None:
            print(f"   [降温中] T={current_temp:.1f}°C | 已耗时 {elapsed:.0f}s")
            if current_temp <= config.COOLING_RESUME_THRESHOLD:
                print(f"✅ 温度已降至 {current_temp}°C，降温完成")
                hardware.stop_heating()
                return True
        else:
            print(f"   [降温中] 温度读取失败 | 已耗时 {elapsed:.0f}s")
        
        if elapsed > config.COOLING_TIMEOUT:
            print(f"⚠️ 降温超时 ({config.COOLING_TIMEOUT}s)，强制继续")
            hardware.stop_heating()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="流体包裹体均一温度自动批量测量系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --start 9 --end 50
  python main.py -s 1 -e 10
  python main.py --csv path/to/data.csv -s 5 -e 20
        """
    )
    parser.add_argument('--csv', '-c', type=str, default=config.DEFAULT_CSV_PATH,
                       help='包裹体数据库CSV路径')
    parser.add_argument('--start', '-s', type=int, default=config.BATCH_START_RANK,
                       help='起始排名（从1开始）')
    parser.add_argument('--end', '-e', type=int, default=config.BATCH_END_RANK,
                       help='结束排名（包含）')
    parser.add_argument('--output', '-o', type=str, default=config.OUTPUT_BASE,
                       help='输出目录')

    args = parser.parse_args()
    start_rank = args.start
    end_rank = args.end
    csv_path = args.csv
    output_base = args.output

    if start_rank < 1:
        print("起始排名必须 >= 1")
        return
    if end_rank < start_rank:
        print(f"结束排名({end_rank})必须 >= 起始排名({start_rank})")
        return

    print("="*70)
    print("流体包裹体均一温度自动批量测量系统")
    print(f"   排名范围: 第 {start_rank} 名 ~ 第 {end_rank} 名")
    print(f"   CSV路径: {csv_path}")
    print(f"   输出目录: {output_base}")
    print(f"   3D搜索网格: {len(config.XY_SCAN_OFFSETS_UM)}个XY位置 × Z±{config.Z_SCAN_RANGE}μm")
    print(f"   降温目标: {config.COOLING_TARGET_TEMP}°C | 恢复阈值: ≤{config.COOLING_RESUME_THRESHOLD}°C")
    print(f"   直方图bin: {config.HISTOGRAM_BIN_WIDTH}°C")
    print("="*70)

    print("\n加载包裹体数据库...")
    inclusions = io_utils.load_inclusion_list(csv_path)
    if not inclusions or len(inclusions) == 0:
        print("未找到包裹体数据")
        return

    inclusions.sort(key=lambda x: float(x.get('置信度', 0)), reverse=True)
    total_db = len(inclusions)

    if start_rank > total_db:
        print(f"起始排名 {start_rank} 超出数据库总数 {total_db}")
        return

    actual_end = min(end_rank, total_db)
    if actual_end < end_rank:
        print(f"结束排名调整为 {actual_end}（数据库共 {total_db} 个）")

    selected = inclusions[start_rank - 1 : actual_end]

    print(f"数据库共 {total_db} 个，选取排名 [{start_rank} ~ {actual_end}] 共 {len(selected)} 个")
    for i, inc in enumerate(selected[:10]):
        db_rank = start_rank + i
        print(f"   [排名{db_rank}] {inc.get('包裹体名称','?')} (置信度: {float(inc.get('置信度',0)):.4f})")
    if len(selected) > 10:
        print(f"   ... 还有 {len(selected) - 10} 个未显示")

    print("\n初始化系统...")
    if not hardware.init_z_axis():
        print("Z轴初始化失败")
        return
    if not hardware.init_temp():
        print("温度读取失败，继续运行")

    print("\n加载深度学习模型...")
    try:
        models.load_models()
    except Exception as e:
        print(f"模型加载失败: {e}")
        hardware.close_z_axis()
        hardware.close_temp()
        return

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_base, f"batch_rank{start_rank}-{actual_end}_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"\n批次输出目录: {batch_dir}")

    user_input = input("\n确认开始批量测量? (y/n): ").strip().lower()
    if user_input != 'y':
        print("已取消")
        hardware.close_z_axis()
        hardware.close_temp()
        return

    hardware.activate_nexus_once()

    all_results = []
    try:
        for i, inc in enumerate(selected):
            actual_rank = start_rank + i

            print(f"\n{'#'*70}")
            print(f"# 进度: {i+1}/{len(selected)} | 数据库排名: {actual_rank}")
            print(f"{'#'*70}")

            success, report = measurement.measure_single_inclusion(inc, batch_dir, i, len(selected))
            all_results.append((success, report))

            if i < len(selected) - 1:
                print(f"\n包裹体排名 {actual_rank} 完成，准备降温后开始下一个...")
                cooling_ok = active_cooling()
                if not cooling_ok:
                    print("降温异常，但继续执行下一个")
                time.sleep(2.0)

        io_utils.generate_batch_report(batch_dir, all_results)

    except KeyboardInterrupt:
        print("\n用户中断批量测量")
    except Exception as e:
        print(f"\n批量测量错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hardware.stop_heating()
        hardware.close_z_axis()
        hardware.close_temp()
        print("\n批量测量程序结束")


if __name__ == "__main__":
    import pyautogui
    pyautogui.FAILSAFE = True
    main()