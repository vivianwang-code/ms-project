# utils/logger_config.py
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    設定統一的日誌系統
    """
    # 確保 logs 資料夾存在
    os.makedirs('data/logs', exist_ok=True)
    
    # 如果沒指定檔案，使用預設檔名
    if log_file is None:
        log_file = f'data/logs/{name}_{datetime.now().strftime("%Y%m%d")}.log'
    
    # 創建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重複添加 handler
    if logger.handlers:
        return logger
    
    # 設定日誌格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 檔案輸出（支援檔案輪轉）
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # 控制台輸出（只顯示 INFO 以上）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger

# 常用的 logger
def get_system_logger():
    """系統運行日誌"""
    return setup_logger('system', 'data/logs/system.log')

def get_error_logger():
    """錯誤日誌"""
    return setup_logger('error', 'data/logs/error.log', logging.ERROR)

def get_phantom_logger():
    """Phantom Load 專用日誌"""
    return setup_logger('phantom', 'data/logs/phantom_monitoring.log')