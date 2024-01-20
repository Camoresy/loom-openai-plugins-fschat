from multiprocessing import Process
from loom_core.openai_plugins.core.control import ControlAdapter

import time
from datetime import datetime
import os
import sys
import logging

logger = logging.getLogger(__name__)
# 为了能使用fastchat_wrapper.py中的函数，需要将当前目录加入到sys.path中
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
import fastchat_process_dict
from fastchat_wrapper import run_model_worker
from fastchat_config import FsChatCfg


class FastChatControlAdapter(ControlAdapter):

    def __init__(self, cfg=None,  state_dict: dict = None):
        self._cfg = FsChatCfg(cfg=cfg)
        super().__init__(state_dict=state_dict)

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    def start_model(self, new_model_name):
        logger.info(f"准备启动新模型进程：{new_model_name}")
        controller_address = self._cfg.get_cfg().get("controller_address", None)
        e = fastchat_process_dict.mp_manager.Event()
        process = Process(
            target=run_model_worker,
            name=f"model_worker - {new_model_name}",
            kwargs=dict(cfg=self._cfg,
                        model_name=new_model_name,
                        controller_address=controller_address,
                        log_level=self.processesInfo.log_level,
                        started_event=e),
            daemon=True,
        )
        process.start()
        process.name = f"{process.name} ({process.pid})"
        fastchat_process_dict.processes["model_worker"][new_model_name] = process
        e.wait()
        logger.info(f"成功启动新模型进程：{new_model_name}")

    def stop_model(self, model_name: str):

        if model_name in fastchat_process_dict.processes["model_worker"]:
            process = fastchat_process_dict.processes["model_worker"].pop(model_name)
            time.sleep(1)
            process.kill()
            logger.info(f"停止模型进程：{model_name}")
        else:
            logger.error(f"未找到模型进程：{model_name}")
            raise Exception(f"未找到模型进程：{model_name}")

    def replace_model(self, model_name: str, new_model_name: str):
        controller_address = self._cfg.get_cfg().get("controller_address", None)
        e = fastchat_process_dict.mp_manager.Event()
        if process := fastchat_process_dict.processes["model_worker"].pop(model_name, None):
            logger.info(f"停止模型进程：{model_name}")
            start_time = datetime.now()
            time.sleep(1)
            process.kill()
            process = Process(
                target=run_model_worker,
                name=f"model_worker - {new_model_name}",
                kwargs=dict(cfg=self._cfg,
                            model_name=new_model_name,
                            controller_address=controller_address,
                            log_level=self.processesInfo.log_level,
                            started_event=e),
                daemon=True,
            )
            process.start()
            process.name = f"{process.name} ({process.pid})"
            fastchat_process_dict.processes["model_worker"][new_model_name] = process
            e.wait()
            timing = datetime.now() - start_time
            logger.info(f"成功启动新模型进程：{new_model_name}。用时：{timing}。")
        else:
            logger.error(f"未找到模型进程：{model_name}")

    @classmethod
    def from_config(cls, cfg=None):
        _state_dict = {
            "controller_name": "fastchat",
            "controller_version": "0.0.1",
            "controller_description": "fastchat controller",
            "controller_author": "fastchat"
        }
        state_dict = cfg.get("state_dict", {})
        if state_dict is not None and _state_dict is not None:
            _state_dict = {**state_dict, **_state_dict}
        else:
            # 处理其中一个或两者都为 None 的情况
            _state_dict = state_dict or _state_dict or {}

        return cls(cfg=cfg, state_dict=_state_dict)
