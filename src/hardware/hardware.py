"""硬件抽象层：所有物理设备交互（XYZ位移台、冷热台温度、Nexus界面点击）。"""
import time
import ctypes
from ctypes import c_long, c_uint, c_ubyte

import clr
import pyautogui

import config

# 模块级状态（单例硬件）
_x_pos = 0
_y_pos = 0
_z_pos = 0
_z_connected = False
_temp_connected = False
_dt_driver = None
_temp_client = None
_nexus_activated = False


def get_position() -> tuple[int, int, int]:
    """获取当前机械坐标（微米）"""
    return _x_pos, _y_pos, _z_pos


# ========================
# 界面控制
# ========================
def activate_nexus_once():
    global _nexus_activated
    if not _nexus_activated:
        pyautogui.click(config.NEXUS_POS[0], config.NEXUS_POS[1])
        time.sleep(0.3)
        _nexus_activated = True


def simple_click(pos: tuple[int, int]):
    pyautogui.click(pos[0], pos[1])
    time.sleep(0.2)


def wait_ui(delay: float = 0.5):
    time.sleep(delay)


# ========================
# 温度控制
# ========================
def init_temp() -> bool:
    global _temp_client, _temp_connected
    try:
        clr.AddReference(config.LINKAM_DLL)
        from Linkam.IpcExtension import IPCClient
        _temp_client = IPCClient()
        _temp_client.Start()
        time.sleep(1.5)
        if _temp_client.GrabHeader():
            _temp_connected = True
            print("✅ 温度读取已连接")
            return True
    except Exception as e:
        print(f"❌ 温度连接失败：{e}")
    return False


def close_temp():
    global _temp_client, _temp_connected
    if _temp_client:
        try:
            _temp_client.Stop()
            print("🔌 温度读取已断开")
        except:
            pass
    _temp_connected = False


def get_temp():
    if not _temp_connected or _temp_client is None:
        return None
    try:
        header = _temp_client.GrabHeader()
        if not header:
            return None
        data = _temp_client.GrabData()
        titles = [t.strip() for t in header.split('\n')[1].split(',') if t.strip()]
        values = [float(v.strip()) for v in data.split(',') if v.strip()]
        if "Temp" in titles:
            temp = values[titles.index("Temp")]
            if temp == temp:
                return round(temp, 2)
    except Exception as e:
        print(f"⚠️ 温度读取异常：{e}")
    return None


def set_heating_rate(rate: float) -> bool:
    if rate < 0.01 or rate > 150.0:
        print(f"❌ 速率超出范围 [0.01, 150.0]")
        return False
    activate_nexus_once()
    pyautogui.click(config.TEMP_RATE_POS[0], config.TEMP_RATE_POS[1])
    wait_ui(0.3)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.write(str(rate))
    pyautogui.press('enter')
    wait_ui(1)
    print(f"✅ 升温速率 = {rate} °C/min")
    return True


def set_target_temp(limit: float) -> bool:
    if limit < -200 or limit > 600:
        print(f"❌ 温度超出范围 [-200, 600]")
        return False
    activate_nexus_once()
    pyautogui.click(config.TEMP_LIMIT_POS[0], config.TEMP_LIMIT_POS[1])
    wait_ui(0.3)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.write(str(limit))
    pyautogui.press('enter')
    wait_ui(1)
    print(f"✅ 目标温度 = {limit} °C")
    return True


def start_heating():
    activate_nexus_once()
    pyautogui.click(config.TEMP_START_POS[0], config.TEMP_START_POS[1])
    wait_ui(1)
    print("✅ 加热已启动")


def stop_heating():
    activate_nexus_once()
    pyautogui.click(config.TEMP_STOP_POS[0], config.TEMP_STOP_POS[1])
    wait_ui(1)
    print("✅ 加热已停止")


# ========================
# Z轴控制
# ========================
def init_z_axis() -> bool:
    global _dt_driver, _z_connected
    try:
        _dt_driver = ctypes.CDLL(config.Z_AXIS_DLL)
        result = _dt_driver.DTSetComm(c_long(config.COM_PORT))
        if result == 1:
            _dt_driver.SetSpeed(c_ubyte(ord('Z')), c_uint(config.Z_SPEED))
            _z_connected = True
            print(f"✅ Z轴已连接 (COM{config.COM_PORT})")
            return True
        else:
            print("❌ Z轴连接失败")
            return False
    except Exception as e:
        print(f"❌ Z轴加载失败：{e}")
        return False


def close_z_axis():
    global _z_connected
    _z_connected = False
    print("🔌 Z轴已断开")


def move_z_absolute(target_z: int) -> bool:
    global _z_pos
    if not _z_connected:
        return False
    delta = target_z - _z_pos
    if delta == 0:
        return True
    if target_z > config.Z_LIMIT_MAX:
        print(f"❌ Z轴目标 {target_z} 超过上限 {config.Z_LIMIT_MAX}")
        return False
    if target_z < config.Z_LIMIT_MIN:
        print(f"❌ Z轴目标 {target_z} 低于下限 {config.Z_LIMIT_MIN}")
        return False
    dir_val = 1 if delta > 0 else -1
    _dt_driver.MoveStageNoWait(c_ubyte(ord('Z')), c_long(dir_val), c_long(abs(delta)))
    time.sleep(1.5)
    _z_pos = target_z
    print(f"Z = {_z_pos}")
    return True


# ========================
# XY轴控制
# ========================
def _move_time(target: int, current: int) -> float:
    d = abs(target - current)
    if d == 0:
        return 0.5
    return (d / config.XY_MOVE_SPEED) + 0.5


def _set_axis(axis: str, pos: tuple[int, int], target: int, current: int) -> bool:
    global _x_pos, _y_pos
    t = _move_time(target, current)
    for _ in range(2):
        try:
            simple_click(pos)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.write(str(target))
            pyautogui.press('enter')
            time.sleep(t)
            if axis == "X":
                _x_pos = target
            else:
                _y_pos = target
            print(f"{axis} = {target}")
            return True
        except:
            time.sleep(0.5)
    print(f"❌ {axis} 轴移动失败")
    return False


def set_x(x: int) -> bool:
    if not (config.XY_LIMIT_MIN <= x <= config.XY_LIMIT_MAX):
        print(f"❌ X轴 {x} 超出范围")
        return False
    return _set_axis("X", config.X_INPUT_POS, x, _x_pos)


def set_y(y: int) -> bool:
    if not (config.XY_LIMIT_MIN <= y <= config.XY_LIMIT_MAX):
        print(f"❌ Y轴 {y} 超出范围")
        return False
    return _set_axis("Y", config.Y_INPUT_POS, y, _y_pos)


def zero_xy():
    activate_nexus_once()
    simple_click(config.ZERO_BTN_POS)
    time.sleep(6)
    global _x_pos, _y_pos
    _x_pos = _y_pos = 0
    print("✅ XY轴已归零")