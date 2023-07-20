import logging


filenames = dict(
    VALIDATION_LOGS = "./logs/validation_log.txt"    
    
)



class AppLogger:
    """
    Description:
    Simple class to save logs into multiple files
    """
    def __init__(self):
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s]')
        self.handlers = {}

        for filename in filenames.keys():
            self.handlers[filename] = logging.FileHandler(filenames[filename])
            self.handlers[filename].setFormatter(formatter)
        


    def custom_log(self, filename, level=logging.INFO):
        custom_logger = logging.getLogger(filenames[filename])
        custom_logger.setLevel(level)
        custom_logger.addHandler(self.handlers[filename])
        return custom_logger

    def write_log(self, logger, message, level):
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.ERROR:
            logger.error(message)
    
    def log_validation(self, message, level):
        validation_logger = self.custom_log("VALIDATION_LOGS")
        self.write_log(validation_logger, message, level)

    


LOGGER = AppLogger()
