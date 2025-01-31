import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler


class AsyncLogger:
    def __init__(self, name="api_logger", log_file="logs/api.log", level=logging.INFO):
        """Инициализация асинхронного логгера с ротацией файлов."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Формат логов
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Файловый обработчик (ротация логов, чтобы файл не разрастался)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)

        # Обработчик вывода в консоль
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        # Добавляем обработчики в логгер
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    async def info(self, message: str):
        """Асинхронный лог `INFO`."""
        await self._log_async(logging.INFO, message)

    async def error(self, message: str):
        """Асинхронный лог `ERROR`."""
        await self._log_async(logging.ERROR, message)

    async def _log_async(self, level: int, message: str):
        """Асинхронное логирование через `run_in_executor` для избежания блокировки."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_log, level, message)

    def _sync_log(self, level: int, message: str):
        """Синхронный метод логирования (используется внутри `run_in_executor`)."""
        self.logger.log(level, message)


async def setup_logger():
    """Создает и возвращает асинхронный логгер."""
    return AsyncLogger()
