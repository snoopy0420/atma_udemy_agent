import os

# DIR
DIR_HOME = os.path.abspath(os.path.join(".", "..", "..", "..", ".."))
DIR_MODEL = os.path.join(DIR_HOME, 'models')
DIR_DATA = os.path.join(DIR_HOME, 'data')
DIR_LOG = os.path.join(DIR_HOME, 'logs')
DIR_SUBMISSIONS = os.path.join(DIR_DATA, 'submission')
DIR_INTERIM = os.path.join(DIR_DATA, 'interim')
DIR_FEATURE = os.path.join(DIR_DATA, 'features')
DIR_FIGURE = os.path.join(DIR_DATA, 'figures')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_INPUT = os.path.join(DIR_RAW, 'input')

# FILE
FILE_SAMPLE_SUBMISSION = 'sample_submission.csv'
FILE_NAME_TRAIN = 'train.csv'
FILE_NAME_TEST = 'test.csv'
FILE_NAME_UDEMY_ACTIVITY = 'udemy_activity.csv'
FILE_NAME_CAREER = 'career.csv'
FILE_NAME_DX = 'dx.csv'
FILE_NAME_HR = 'hr.csv'
FILE_NAME_ORVER_TIME = 'overtime_work_by_month.csv'
FILE_NAME_POSITION_HISTORY = 'position_history.csv'

# CONFIG
TARGET_COL = 'target'
KEY_COL = ['社員番号', 'category']

# ## ModernBERT 設定
MODEL_CHECKPOINT = 'sbintuitions/modernbert-ja-70m'
MODEL_NAME = 'modernbert-ja-70m'
N_SPLITS = 5
MAX_LENGTH = 1024
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
GRADIENT_ACCUMULATION_STEPS = 4
N_EPOCHS = 10
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = 'cosine'
SEED = 2025
