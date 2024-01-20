import json
from typing import List

from loom_core.openai_plugins.core.adapter import LLMWorkerInfo
from loom_core.openai_plugins.core.profile_endpoint.core import ProfileEndpointAdapter

import os
import sys
import logging

logger = logging.getLogger(__name__)
# 为了能使用fastchat_wrapper.py中的函数，需要将当前目录加入到sys.path中
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
from fastchat_config import FsChatCfg, log_verbose
from server_utils import (fschat_controller_address, get_model_worker_config, get_httpx_client)


class FastChatProfileEndpointAdapter(ProfileEndpointAdapter):
    """Adapter for the profile endpoint."""

    def __init__(self, cfg=None,  state_dict: dict = None):
        self._cfg = FsChatCfg(cfg=cfg)
        super().__init__(state_dict=state_dict)

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    def list_running_models(self) -> List[LLMWorkerInfo]:
        """从fastchat controller获取已加载模型列表及其配置项"""
        try:
            controller_address = fschat_controller_address(cfg=self._cfg)
            with get_httpx_client(cfg=self._cfg) as client:
                r = client.post(controller_address + "/list_models")
                models = r.json()["models"]
                list_worker = [self.get_model_config(m) for m in models]
                return list_worker
        except Exception as e:
            logger.error(f'{e.__class__.__name__}: {e}',
                         exc_info=e if log_verbose else None)
            return []

    def list_llm_models(self) -> List[LLMWorkerInfo]:
        """获取已配置模型列表"""
        try:
            models = self._cfg.local_llm_model_names()
            models.extend(self._cfg.online_llm_model_names())
            list_worker = [self.get_model_config(m) for m in models]
            return list_worker
        except Exception as e:
            logger.error(f'{e.__class__.__name__}: {e}',
                         exc_info=e if log_verbose else None)
            return []

    def get_model_config(self, model_name) -> LLMWorkerInfo:

        '''
        获取LLM模型配置项（合并后的）
        '''
        model_info = {}
        # 删除ONLINE_MODEL配置中的敏感信息
        for k, v in get_model_worker_config(cfg=self._cfg, model_name=model_name).items():
            if not (k == "worker_class"
                    or "key" in k.lower()
                    or "secret" in k.lower()
                    or k.lower().endswith("id")):
                model_info[k] = v

        info_obj = LLMWorkerInfo(worker_id=model_name,
                                 model_name=model_name,
                                 model_description="",
                                 providers=["model"],
                                 model_extra_info=json.dumps(model_info, ensure_ascii=False, indent=2))

        return info_obj

    @classmethod
    def from_config(cls, cfg=None):
        _state_dict = {
            "profile_name": "fastchat",
            "profile_version": "0.0.1",
            "profile_description": "fastchat profile endpoint",
            "profile_author": "fastchat"
        }
        state_dict = cfg.get("state_dict", {})
        if state_dict is not None and _state_dict is not None:
            _state_dict = {**state_dict, **_state_dict}
        else:
            # 处理其中一个或两者都为 None 的情况
            _state_dict = state_dict or _state_dict or {}

        return cls(cfg=cfg, state_dict=_state_dict)
