from multiprocessing import Process
from typing import Dict
import logging

logger = logging.getLogger(__name__)
mp_manager = None
processes: Dict[str, Process] = {}


def stop_all():
    for p in processes.values():
        logger.warning("Sending SIGKILL to %s", p)
        # Queues and other inter-process communication primitives can break when
        # process is killed, but we don't care here

        if isinstance(p, dict):
            for process in p.values():
                try:

                    process.kill()
                except Exception as e:
                    logger.info("Failed to kill process %s", p, exc_info=True)
        else:
            try:
                p.kill()
            except Exception as e:
                logger.info("Failed to kill process %s", p, exc_info=True)

    for p in processes.values():
        logger.info("Process status: %s", p)
