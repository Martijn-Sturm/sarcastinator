import os
import logging
import threading
from logger import RunsInAThread
from logger import config_thr_exc_log
from logger import get_logger

logger = get_logger(filename="test", name=__name__)
config_thr_exc_log()

logger.debug("Test debug")
logger.info("Test info")
logger.warning("Test warning")
logger.error("Test error")

# raise ValueError("we catch this one")



# Test that the logger get exception raised from thread
foo = RunsInAThread("Runs on thread")
thread = threading.Thread(target=foo.run, args=())
thread.daemon = True
thread.start()
thread.join()

logger.warning("Test after thread exception")


foo = RunsInAThread("Runs on thread")
thread = threading.Thread(target=foo.run, args=())
thread.daemon = True
thread.start()
thread.join()