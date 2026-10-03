"""图像采集：封装截图实现，对外提供纯业务接口。"""
import numpy as np
import pyautogui
from PIL import Image

import config


def take_screenshot(save_path: str | None = None) -> np.ndarray | None:
    """
    采集当前视野图像。
    
    Returns:
        RGB 格式的 numpy 数组，失败返回 None。
    """
    try:
        img = pyautogui.screenshot(region=config.SCREEN_REGION)
        img_rgb = img.convert("RGB")
        if save_path:
            img_rgb.save(save_path, quality=100)
        return np.array(img_rgb)
    except Exception as e:
        print(f"❌ 截图失败：{e}")
        return None