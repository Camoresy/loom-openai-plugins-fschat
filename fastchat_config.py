# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
from typing import List

HTTPX_DEFAULT_TIMEOUT = 300.0
log_verbose = True


class FsChatCfg:
    def __init__(self, cfg: dict = None):
        if cfg is None:
            raise RuntimeError("FsChatCfg cfg is None.")
        self._cfg = cfg

    def get_cfg(self):
        return self._cfg

    def get_run_controller_cfg(self):
        return self._cfg.get("run_controller", {})

    def get_run_openai_api_cfg(self):
        return self._cfg.get("run_openai_api", {})

    def get_online_llm_model_by_name(self, name: str):

        online_llm_model_cfg = self._cfg.get("online_llm_model", None)
        if online_llm_model_cfg is None:
            raise RuntimeError("online_llm_model is None.")

        get = lambda model_name: online_llm_model_cfg[
            self.get_online_llm_model_index_by_name(model_name)
        ].get(model_name, {})
        return get(name)

    def get_online_llm_model_index_by_name(self, name) -> int:

        online_llm_model_cfg = self._cfg.get("online_llm_model", None)
        if online_llm_model_cfg is None:
            raise RuntimeError("online_llm_model is None.")

        for cfg in online_llm_model_cfg:
            for key, online_llm_model in cfg.items():
                if key == name:
                    return online_llm_model_cfg.index(cfg)
        return -1

    def get_fschat_model_workers_by_name(self, name: str):

        fschat_model_workers_cfg = self._cfg.get("fschat_model_workers", None)
        if fschat_model_workers_cfg is None:
            raise RuntimeError("fschat_model_workers_cfg is None.")

        get = lambda model_name: fschat_model_workers_cfg[
            self.get_fschat_model_workers_index_by_name(model_name)
        ].get(model_name, {})
        return get(name)

    def get_fschat_model_workers_index_by_name(self, name) -> int:

        fschat_model_workers_cfg = self._cfg.get("fschat_model_workers", None)
        if fschat_model_workers_cfg is None:
            raise RuntimeError("fschat_model_workers_cfg is None.")

        for cfg in fschat_model_workers_cfg:
            for key, fschat_model_workers in cfg.items():
                if key == name:
                    return fschat_model_workers_cfg.index(cfg)
        return -1

    def online_llm_model_names(self) -> List[str]:

        online_llm_model_cfg = self._cfg.get("online_llm_model", None)
        if online_llm_model_cfg is None:
            raise RuntimeError("online_llm_model is None.")
        model_names = []
        for cfg in online_llm_model_cfg:
            model_names.extend(cfg.keys())

        return model_names

    def local_llm_model_names(self) -> List[str]:

        local_llm_model_cfg = self._cfg.get("llm_model", None)
        if local_llm_model_cfg is None:
            raise RuntimeError("local_llm_model_cfg is None.")
        model_names = []
        for model_name in local_llm_model_cfg:
            model_names.append(model_name)

        return model_names

