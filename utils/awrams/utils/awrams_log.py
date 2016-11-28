# #FILE_LOGGING_MODE OPTIONS
# APPENDFILE='append'
# TIMESTAMPEDFILE='timestamp'
# ROTATEDSIZEDFILE='rotatedsized'
# DAILYROTATEDFILE='dailyrotated'

import os
import sys
import datetime
from .settings import APPENDFILE,TIMESTAMPEDFILE,ROTATEDSIZEDFILE,DAILYROTATEDFILE

def get_sim_logger():
    '''
    Custom logger for the simulation interface; links simulation_server
    messages to the testing suite
    '''
    import logging
    from awrams.utils.settings import LOG_LEVEL #pylint: disable=no-name-in-module

    sim_logger = logging.getLogger('simulation_server')

    if sim_logger.handlers:
        return sim_logger

    sim_logger.setLevel(LOG_LEVEL)

    return sim_logger

def establish_logging():
    import logging
    import logging.handlers
    # from awrams.utils.settings import LOG_TO_STDOUT, LOG_TO_STDERR, LOG_TO_FILE, FILE_LOGGING_MODE, APPNAME, LOGFILEBASE, LOG_LEVEL, LOGFORMAT, ROTATEDSIZEDFILES, ROTATEDSIZEDBYTES, DAILYROTATEDFILES #pylint: disable=no-name-in-module
    from .settings import LOG_TO_STDOUT,LOG_TO_STDERR,LOG_TO_FILE,FILE_LOGGING_MODE,APPNAME,LOGFILEBASE,LOG_LEVEL,\
                          LOGFORMAT,ROTATEDSIZEDFILES,ROTATEDSIZEDBYTES,DAILYROTATEDFILES

    logger = logging.getLogger(APPNAME)
    if logger.handlers:
        logger.debug("Logger already configured")
        return logger

    formatter = logging.Formatter(LOGFORMAT)
    handlersList = []

    if LOG_TO_FILE:
        if FILE_LOGGING_MODE==TIMESTAMPEDFILE:
            now=datetime.datetime.now()
            timestamp=now.strftime("%Y_%m_%d_%H_%M_%S")
            logfile="%s_%s.log"%(LOGFILEBASE,timestamp)
        else:
            logfile="%s.log"%(LOGFILEBASE)

        folder=os.path.dirname(os.path.realpath(logfile))
        if not os.path.exists(folder):
            os.makedirs(folder)

        if FILE_LOGGING_MODE==TIMESTAMPEDFILE:
            handlersList.append(logging.FileHandler(os.path.expandvars(os.path.expanduser(logfile))))

        elif FILE_LOGGING_MODE==ROTATEDSIZEDFILE:
            handlersList.append(logging.handlers.RotatingFileHandler(logfile, maxBytes=ROTATEDSIZEDBYTES, \
                backupCount=ROTATEDSIZEDFILES))

        elif FILE_LOGGING_MODE==DAILYROTATEDFILE:
            handlersList.append(logging.handlers.TimedRotatingFileHandler(logfile, when='d', \
                interval=1, backupCount=DAILYROTATEDFILES, encoding=None, delay=False, utc=True))

        else:
            #FILE_LOGGING_MODE by default is APPENDFILE:
            handlersList.append(logging.FileHandler(os.path.expandvars(os.path.expanduser(logfile))))

    if LOG_TO_STDOUT:
        handlersList.append(logging.StreamHandler(sys.stdout))

    if LOG_TO_STDERR:
        handlersList.append(logging.StreamHandler(sys.stderr))

    for hdlr in handlersList:
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    logger.setLevel(LOG_LEVEL)
    return logger

PROPAGATE = False

def get_module_logger(module_name="default"):
    establish_logging()
    import logging
    from awrams.utils.settings import APPNAME, DEBUG_MODULES #pylint: disable=no-name-in-module
    logger = logging.getLogger("%s.%s"%(APPNAME,module_name))

    #Prevent doubled log messages in notebook
    logger.parent.propagate = PROPAGATE #False

    if module_name in DEBUG_MODULES:
        logger.setLevel(logging.DEBUG)
    return logger

