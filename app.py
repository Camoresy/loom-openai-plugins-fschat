from typing import List

from loom_core.openai_plugins.core.adapter import ProcessesInfo
from loom_core.openai_plugins.core.application import ApplicationAdapter
from multiprocessing import Process
import multiprocessing as mp
import os
import sys
import logging

logger = logging.getLogger(__name__)
# 为了能使用fastchat_wrapper.py中的函数，需要将当前目录加入到sys.path中
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from server_utils import (get_model_worker_config)
from fastchat_wrapper import run_controller, run_model_worker, run_openai_api
import fastchat_process_dict
from fastchat_config import FsChatCfg

class FastChatApplicationAdapter(ApplicationAdapter):
    processesInfo: ProcessesInfo = None
    controller_started: mp.Event = None
    model_worker_started: List[mp.Event] = []

    def __init__(self, cfg=None, state_dict: dict = None):
        self._cfg = FsChatCfg(cfg=cfg)
        super().__init__(state_dict=state_dict)

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    @classmethod
    def from_config(cls, cfg=None):
        _state_dict = {
            "application_name": "fastchat",
            "application_version": "0.0.1",
            "application_description": "fastchat application",
            "application_author": "fastchat"
        }
        state_dict = cfg.get("state_dict", {})
        if state_dict is not None and _state_dict is not None:
            _state_dict = {**state_dict, **_state_dict}
        else:
            # 处理其中一个或两者都为 None 的情况
            _state_dict = state_dict or _state_dict or {}
        return cls(cfg=cfg, state_dict=_state_dict)

    def init_processes(self, processesInfo: ProcessesInfo):

        self.processesInfo = processesInfo
        openai_api = self._cfg.get_cfg().get("openai_api", True)
        model_worker = self._cfg.get_cfg().get("model_worker", True)
        api_worker = self._cfg.get_cfg().get("api_worker", True)
        model_names = self._cfg.get_cfg().get("model_names", [])
        controller_address = self._cfg.get_cfg().get("controller_address", None)
        lite = self._cfg.get_cfg().get("lite", False)

        if lite:
            model_worker = False

        fastchat_process_dict.processes = {"online_api": {}, "model_worker": {}}
        fastchat_process_dict.mp_manager = mp.Manager()
        self.controller_started = fastchat_process_dict.mp_manager.Event()

        try:
            if openai_api:
                process = Process(
                    target=run_controller,
                    name=f"controller",
                    kwargs=dict(cfg=self._cfg,
                                log_level=processesInfo.log_level, started_event=self.controller_started),
                    daemon=True,
                )
                fastchat_process_dict.processes["controller"] = process

                process = Process(
                    target=run_openai_api,
                    name=f"openai_api",
                    kwargs=dict(cfg=self._cfg,
                                log_level=processesInfo.log_level, started_event=self.controller_started),
                    daemon=True,
                )
                fastchat_process_dict.processes["openai_api"] = process

            if model_worker:
                for model_name in model_names:
                    config = get_model_worker_config(cfg=self._cfg, model_name=model_name)
                    if not config.get("online_api"):
                        e = fastchat_process_dict.mp_manager.Event()
                        self.model_worker_started.append(e)
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {model_name}",
                            kwargs=dict(cfg=self._cfg,
                                        model_name=model_name,
                                        controller_address=controller_address,
                                        log_level=processesInfo.log_level,
                                        started_event=e),
                            daemon=True,
                        )
                        fastchat_process_dict.processes["model_worker"][model_name] = process

            if api_worker:
                for model_name in model_names:
                    config = get_model_worker_config(cfg=self._cfg, model_name=model_name)
                    if (config.get("online_api")
                            and config.get("worker_class")
                            and model_name in self._cfg.get_cfg().get("fschat_model_workers", [])):
                        e = None
                        self.model_worker_started.append(e)
                        process = Process(
                            target=run_model_worker,
                            name=f"api_worker - {model_name}",
                            kwargs=dict(cfg=self._cfg,
                                        model_name=model_name,
                                        controller_address=controller_address,
                                        log_level=processesInfo.log_level,
                                        started_event=e),
                            daemon=True,
                        )
                        fastchat_process_dict.processes["online_api"][model_name] = process

        except Exception as e:
            logger.error("Failed to init fastchat", exc_info=True)
            fastchat_process_dict.stop_all()

    def start(self):
        try:
            # 保证任务收到SIGINT后，能够正常退出
            if p := fastchat_process_dict.processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                self.controller_started.wait()  # 等待controller启动完成

            if p := fastchat_process_dict.processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in fastchat_process_dict.processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in fastchat_process_dict.processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 等待所有model_worker启动完成
            for e in self.model_worker_started:
                e.wait()

        except Exception as e:
            logger.error("Failed to start fastchat", exc_info=True)
            fastchat_process_dict.stop_all()

    def stop(self):
        fastchat_process_dict.stop_all()
